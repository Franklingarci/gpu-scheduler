"""The scheduler state machine: queue handling and the guards on a policy's plan."""
import pytest

from scheduler import policy as P
from scheduler.core import Scheduler, SchedulerError
from scheduler.gpu import MockGPUMonitor
from scheduler.models import JobStatus
from scheduler.policy import Placement, Policy
from scheduler.pool import PlacementTracker


class ScriptedPolicy(Policy):
    """Returns a fixed plan, so the guards can be aimed at directly.

    Every guard below fires on a policy bug. Without a deliberately broken
    policy there is no way to prove the guard works, and a guard that has
    never fired is a guard you are trusting on faith.
    """

    ordering = "scripted"

    def __init__(self, placements):
        super().__init__()
        self._placements = placements

    def plan(self, ready, slots, now):
        return self._placements


def build(policy, gpus=2, mb=16000):
    tracker = PlacementTracker(MockGPUMonitor(gpu_count=gpus, memory_per_gpu_mb=mb), headroom_mb=0)
    return Scheduler(policy=policy, tracker=tracker)


# --------------------------------------------------------------- queueing

def test_a_job_is_invisible_until_it_arrives(job):
    sched = build(P.build("priority", "best_fit"))
    sched.submit(job(1, 4000, arrival_time=50.0))

    assert sched.schedule(now=10.0) == []
    assert len(sched.schedule(now=50.0)) == 1


def test_starting_a_job_moves_it_out_of_the_queue(job):
    sched = build(P.build("priority", "best_fit"))
    sched.submit(job(1, 4000))
    started = sched.schedule(now=0.0)

    assert [j.id for j in started] == [1]
    assert sched.queue == []
    assert set(sched.running) == {1}
    assert started[0].status is JobStatus.RUNNING
    assert started[0].start_time == 0.0


def test_a_job_that_does_not_fit_stays_queued(job):
    sched = build(P.build("priority", "best_fit"), gpus=1, mb=8000)
    sched.submit(job(1, 6000))
    sched.submit(job(2, 6000))

    sched.schedule(now=0.0)
    assert len(sched.running) == 1
    assert len(sched.queue) == 1
    assert sched.pending


def test_completing_a_job_frees_its_memory(job):
    sched = build(P.build("priority", "best_fit"), gpus=1, mb=8000)
    sched.submit(job(1, 6000))
    sched.schedule(now=0.0)

    assert sched.tracker.slots()[0].free_memory_mb == 2000
    done = sched.complete(1, now=12.0)

    assert sched.tracker.slots()[0].free_memory_mb == 8000
    assert done.status is JobStatus.COMPLETED
    assert done.completion_time == 12.0
    assert done.wait_time == 0.0
    assert done.run_time == 12.0
    assert not sched.pending


def test_pool_utilization_sums_bytes_rather_than_averaging_percentages(job):
    """A mixed pool would be misreported by averaging per-GPU percentages:
    a full 4GB card and an empty 32GB card is 11% utilized, not 50%."""
    tracker = PlacementTracker(MockGPUMonitor(gpu_count=2, memory_per_gpu_mb=16000), headroom_mb=0)
    sched = Scheduler(policy=P.build("priority", "best_fit"), tracker=tracker)
    sched.submit(job(1, 8000))
    sched.schedule(now=0.0)

    assert sched.utilization_pct() == pytest.approx(25.0)  # 8000 of 32000


# ----------------------------------------------------------------- guards

def test_placing_a_job_that_has_not_arrived_is_rejected(job):
    sched = build(ScriptedPolicy([Placement(job_id=99, gpu_id=0)]))
    sched.submit(job(1, 4000))

    with pytest.raises(SchedulerError, match="not ready"):
        sched.schedule(now=0.0)


def test_placing_onto_a_gpu_that_does_not_exist_is_rejected(job):
    sched = build(ScriptedPolicy([Placement(job_id=1, gpu_id=7)]), gpus=2)
    sched.submit(job(1, 4000))

    with pytest.raises(SchedulerError, match="unknown gpu"):
        sched.schedule(now=0.0)


def test_overcommitting_a_gpu_in_one_pass_is_rejected(job):
    """This is the guard that matters most.

    Without it the tracker records an impossible placement, the run finishes,
    and the numbers look entirely reasonable while being wrong. A crash is
    strictly better than a plausible lie.
    """
    plan = [Placement(job_id=1, gpu_id=0), Placement(job_id=2, gpu_id=0)]
    sched = build(ScriptedPolicy(plan), gpus=1, mb=16000)
    sched.submit(job(1, 12000))
    sched.submit(job(2, 12000))

    with pytest.raises(SchedulerError, match="over-committed"):
        sched.schedule(now=0.0)


def test_completing_a_job_that_is_not_running_is_rejected(job):
    sched = build(P.build("priority", "best_fit"))
    with pytest.raises(SchedulerError, match="not running"):
        sched.complete(42, now=1.0)


def test_a_job_cannot_be_completed_twice(job):
    sched = build(P.build("priority", "best_fit"))
    sched.submit(job(1, 4000))
    sched.schedule(now=0.0)
    sched.complete(1, now=5.0)

    with pytest.raises(SchedulerError, match="not running"):
        sched.complete(1, now=6.0)


# ------------------------------------------------------- clock-independence

def test_the_core_never_reads_a_clock():
    """Structural test with a real point behind it.

    If core.py starts reading a clock, the simulator and the live driver stop
    agreeing about what 'now' means, and the benchmark quietly stops measuring
    the thing that ships.

    Parsed rather than grepped: the first version of this test searched the
    raw source and tripped over the word "time.monotonic" inside a docstring
    explaining why the clock is absent.
    """
    import ast
    from pathlib import Path

    import scheduler.core as core_module

    tree = ast.parse(Path(core_module.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not ({"time", "datetime", "asyncio"} & imported), (
        f"core.py must stay clock-free; it imports {imported}"
    )
