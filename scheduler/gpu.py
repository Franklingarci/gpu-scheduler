"""
Reading GPU state. Read-only, on purpose.

This module answers exactly one question: what does the hardware say right
now? It never records a scheduling decision - that belongs to pool.py.

Keeping those apart is the whole reason NVMLGPUMonitor can stand in for
MockGPUMonitor without the scheduler noticing. NVML genuinely cannot
"allocate" anything; it is a sensor, not a control interface. An interface
that asks a monitor to allocate is an interface no real driver can implement.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceState:
    """One GPU as the hardware reports it, at one instant.

    Frozen because it is a reading, not a record you mutate. Every snapshot
    produces fresh instances, so a stale reading can never be silently
    edited into looking current.
    """
    id: int
    total_memory_mb: int
    used_memory_mb: int
    compute_util_pct: float = 0.0

    @property
    def free_memory_mb(self) -> int:
        return max(0, self.total_memory_mb - self.used_memory_mb)


class GPUMonitor(ABC):
    """A source of truth about hardware. One method, deliberately."""

    @abstractmethod
    def snapshot(self) -> list[DeviceState]:
        """Current state of every GPU in the pool."""
        ...


class MockGPUMonitor(GPUMonitor):
    """A fixed pool that reports no external memory use.

    In simulation the hardware holds nothing we did not put there, so this
    always reports used=0 and lets pool.py derive occupancy from what the
    scheduler reserved. That is what makes a simulation a simulation: the
    two sources of truth agree by construction.

    `external_used_mb` deliberately breaks that agreement so tests can cover
    the case that only happens on real hardware - somebody else's process
    holding memory we never reserved.
    """

    def __init__(
        self,
        gpu_count: int = 2,
        memory_per_gpu_mb: int = 16000,
        external_used_mb: dict[int, int] | None = None,
    ):
        self.gpu_count = gpu_count
        self.memory_per_gpu_mb = memory_per_gpu_mb
        self._external = external_used_mb or {}

    def snapshot(self) -> list[DeviceState]:
        return [
            DeviceState(
                id=i,
                total_memory_mb=self.memory_per_gpu_mb,
                used_memory_mb=self._external.get(i, 0),
            )
            for i in range(self.gpu_count)
        ]


class NVMLGPUMonitor(GPUMonitor):
    """Real hardware, via pynvml. Requires an NVIDIA GPU and `pip install pynvml`.

    Imports pynvml inside __init__ rather than at module scope so the whole
    project stays importable, testable and runnable on a laptop with no GPU.
    """

    def __init__(self):
        import pynvml  # noqa: PLC0415 - deliberate, see docstring

        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handles = {
            i: pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        }

    def snapshot(self) -> list[DeviceState]:
        states = []
        for gpu_id, handle in self._handles.items():
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
            states.append(
                DeviceState(
                    id=gpu_id,
                    total_memory_mb=mem.total // (1024 * 1024),
                    used_memory_mb=mem.used // (1024 * 1024),
                    compute_util_pct=float(util.gpu),
                )
            )
        return states
