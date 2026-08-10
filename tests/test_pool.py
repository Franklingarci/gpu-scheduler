"""Reconciling the scheduler's ledger against what the hardware reports."""
from scheduler.gpu import MockGPUMonitor, NVMLGPUMonitor
from scheduler.pool import PlacementTracker


def test_reserving_reduces_free_memory(job):
    tracker = PlacementTracker(MockGPUMonitor(gpu_count=2, memory_per_gpu_mb=16000), headroom_mb=0)
    tracker.reserve(job(1, 6000), 0)

    free = {s.id: s.free_memory_mb for s in tracker.slots()}
    assert free == {0: 10000, 1: 16000}


def test_releasing_returns_the_memory(job):
    tracker = PlacementTracker(MockGPUMonitor(gpu_count=1), headroom_mb=0)
    placed = job(1, 6000)
    placed.gpu_id = 0
    tracker.reserve(placed, 0)
    tracker.release(placed)

    assert tracker.reserved_mb(0) == 0
    assert tracker.slots()[0].free_memory_mb == 16000


def test_headroom_is_withheld(job):
    tracker = PlacementTracker(MockGPUMonitor(gpu_count=1, memory_per_gpu_mb=16000), headroom_mb=512)
    assert tracker.slots()[0].free_memory_mb == 15488


def test_foreign_memory_shrinks_availability_even_though_we_never_reserved_it(job):
    """The case that only exists on real hardware.

    Someone else's process holds 8GB. Our ledger has no idea - we reserved
    2GB and by our own books 14GB is free. Believing the ledger puts a 10GB
    job onto a card with 6GB actually available, and it OOMs.
    """
    monitor = MockGPUMonitor(gpu_count=1, memory_per_gpu_mb=16000, external_used_mb={0: 8000})
    tracker = PlacementTracker(monitor, headroom_mb=0)
    tracker.reserve(job(1, 2000), 0)

    slot = tracker.slots()[0]
    assert slot.reserved_memory_mb == 2000        # what we think we placed
    assert slot.observed_used_mb == 8000          # what the hardware sees
    assert slot.free_memory_mb == 8000            # min(16000-8000, 16000-2000)


def test_our_ledger_wins_when_it_is_the_gloomier_number(job):
    """The simulation case: nothing foreign exists, so bookkeeping decides."""
    tracker = PlacementTracker(MockGPUMonitor(gpu_count=1, memory_per_gpu_mb=16000), headroom_mb=0)
    tracker.reserve(job(1, 9000), 0)

    slot = tracker.slots()[0]
    assert slot.observed_used_mb == 0
    assert slot.free_memory_mb == 7000


def test_free_memory_never_goes_negative():
    """Foreign usage plus headroom can exceed the card; callers must not see
    a negative budget, or a `free >= need` check would start passing."""
    monitor = MockGPUMonitor(gpu_count=1, memory_per_gpu_mb=16000, external_used_mb={0: 16000})
    tracker = PlacementTracker(monitor, headroom_mb=512)
    assert tracker.slots()[0].free_memory_mb == 0


def test_utilization_counts_withheld_memory_as_used(job):
    """Utilization is measured against what we could actually hand out.
    Reporting the raw hardware figure would overstate free capacity by the
    headroom the reconciliation is deliberately holding back."""
    tracker = PlacementTracker(MockGPUMonitor(gpu_count=1, memory_per_gpu_mb=16000), headroom_mb=0)
    tracker.reserve(job(1, 8000), 0)
    assert tracker.slots()[0].utilization_pct == 50.0


def test_release_of_an_unplaced_job_is_a_no_op(job):
    """A job that never started has gpu_id None; releasing it must not explode
    or corrupt another card's books."""
    tracker = PlacementTracker(MockGPUMonitor(gpu_count=1), headroom_mb=0)
    tracker.release(job(99, 1000))
    assert tracker.reserved_mb(0) == 0


def test_nvml_monitor_satisfies_the_same_interface_as_the_mock():
    """The regression test for the bug that caused the rebuild.

    The old code called allocate() on the monitor; only the mock had it, so
    swapping in real hardware raised AttributeError on the first placement.
    Nothing here touches a GPU - it checks that the two implementations
    present the same surface, which is the property that was silently false.
    """
    required = {"snapshot"}
    assert required <= set(dir(MockGPUMonitor))
    assert required <= set(dir(NVMLGPUMonitor))

    forbidden = {"allocate", "release", "reserve"}
    assert not (forbidden & set(vars(MockGPUMonitor)))
    assert not (forbidden & set(vars(NVMLGPUMonitor)))
