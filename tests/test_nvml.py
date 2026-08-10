"""NVMLGPUMonitor against a fake driver.

This path cannot run on a laptop and rented GPU time is metered, so the parts
that are easy to get wrong and expensive to discover late - unit conversion,
field mapping, device enumeration - are pinned here against a stub that
mimics pynvml's shape. It is not a substitute for a real hardware run, but it
means the first real run is debugging the driver rather than debugging us.
"""
import sys

import pytest

from scheduler.gpu import NVMLGPUMonitor
from scheduler.pool import PlacementTracker

MIB = 1024 * 1024


class _MemInfo:
    def __init__(self, total_bytes, used_bytes):
        self.total = total_bytes
        self.used = used_bytes
        self.free = total_bytes - used_bytes


class _Util:
    def __init__(self, gpu_pct):
        self.gpu = gpu_pct
        self.memory = 0


class FakePynvml:
    """Mimics the handful of pynvml calls NVMLGPUMonitor makes."""

    def __init__(self, devices):
        # devices: list of (total_bytes, used_bytes, compute_pct)
        self._devices = devices
        self.init_calls = 0

    def nvmlInit(self):
        self.init_calls += 1

    def nvmlDeviceGetCount(self):
        return len(self._devices)

    def nvmlDeviceGetHandleByIndex(self, index):
        return index  # handles are opaque; an int is a fine stand-in

    def nvmlDeviceGetMemoryInfo(self, handle):
        total, used, _ = self._devices[handle]
        return _MemInfo(total, used)

    def nvmlDeviceGetUtilizationRates(self, handle):
        return _Util(self._devices[handle][2])


@pytest.fixture
def fake_nvml(monkeypatch):
    """Inject a fake pynvml. NVMLGPUMonitor imports it inside __init__ rather
    than at module scope, which is what makes this possible - and what keeps
    the whole project importable on a machine with no GPU."""
    def _install(devices):
        fake = FakePynvml(devices)
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        return fake
    return _install


def test_bytes_are_converted_to_mib(fake_nvml):
    """NVML reports bytes; the rest of the project speaks MiB.

    A missed conversion here would report a 40GB A100 as 42 949 672 960 'MB'
    of free memory, and the scheduler would place every job on card 0.
    """
    fake_nvml([(40 * 1024 * MIB, 8 * 1024 * MIB, 55)])
    state = NVMLGPUMonitor().snapshot()[0]

    assert state.total_memory_mb == 40 * 1024
    assert state.used_memory_mb == 8 * 1024
    assert state.free_memory_mb == 32 * 1024


def test_every_device_is_enumerated_with_its_index_as_id(fake_nvml):
    """Slot ids must match NVML device indices, because those are the numbers
    that go into CUDA_VISIBLE_DEVICES when the live driver launches a job."""
    fake_nvml([(16 * 1024 * MIB, 0, 0)] * 4)
    states = NVMLGPUMonitor().snapshot()

    assert [s.id for s in states] == [0, 1, 2, 3]


def test_compute_utilization_is_carried_through_separately_from_memory(fake_nvml):
    """A card can be 90% busy on 5% of its memory. Memory occupancy is a proxy
    for load, not a measurement of it, and conflating them is the blind spot
    the simulator already has."""
    fake_nvml([(16 * 1024 * MIB, 800 * MIB, 90)])
    state = NVMLGPUMonitor().snapshot()[0]

    assert state.compute_util_pct == 90.0
    assert state.used_memory_mb == 800


def test_nvml_is_initialised_once_not_per_snapshot(fake_nvml):
    """nvmlInit on every read would be a syscall per scheduling decision."""
    fake = fake_nvml([(16 * 1024 * MIB, 0, 0)])
    monitor = NVMLGPUMonitor()
    monitor.snapshot()
    monitor.snapshot()

    assert fake.init_calls == 1


def test_a_foreign_process_on_real_hardware_shrinks_what_we_hand_out(fake_nvml):
    """The end-to-end case the whole reconciliation exists for.

    The card reports 12GB used by somebody else. Our ledger is empty, so by
    our own books all 16GB is free. Believing the ledger puts a 10GB job onto
    4GB of actual room.
    """
    fake_nvml([(16 * 1024 * MIB, 12 * 1024 * MIB, 40)])
    tracker = PlacementTracker(NVMLGPUMonitor(), headroom_mb=512)
    slot = tracker.slots()[0]

    assert slot.reserved_memory_mb == 0
    assert slot.observed_used_mb == 12 * 1024
    assert slot.free_memory_mb == 4 * 1024 - 512


def test_an_empty_pool_does_not_crash(fake_nvml):
    """nvmlDeviceGetCount can legitimately return 0 - wrong instance type,
    driver not loaded, devices masked by CUDA_VISIBLE_DEVICES."""
    fake_nvml([])
    assert NVMLGPUMonitor().snapshot() == []
