"""
GPU state monitoring, abstracted behind a common interface.

Two implementations:
  - MockGPUMonitor: in-memory, no hardware needed. Use this to develop and
    test the scheduler locally.
  - NVMLGPUMonitor: reads real hardware state via pynvml. Swap this in once
    you're running on an actual GPU instance (Lambda, EC2, etc). Same
    interface, so the scheduler code doesn't change.
"""
from abc import ABC, abstractmethod
from .models import GPUDevice


class GPUMonitor(ABC):
    @abstractmethod
    def get_devices(self) -> list[GPUDevice]:
        """Return current state of all GPUs in the pool."""
        ...


class MockGPUMonitor(GPUMonitor):
    """Simulates a fixed pool of GPUs entirely in memory. No hardware,
    no dependencies — this is what you develop and test the scheduler
    logic against before touching real infra."""

    def __init__(self, gpu_count: int = 2, memory_per_gpu_mb: int = 24000):
        self._devices = {
            i: GPUDevice(id=i, total_memory_mb=memory_per_gpu_mb)
            for i in range(gpu_count)
        }

    def get_devices(self) -> list[GPUDevice]:
        return list(self._devices.values())

    def get_device(self, gpu_id: int) -> GPUDevice:
        return self._devices[gpu_id]

    def allocate(self, gpu_id: int, memory_mb: int, job_id: int) -> None:
        dev = self._devices[gpu_id]
        dev.used_memory_mb += memory_mb
        dev.running_jobs.append(job_id)

    def release(self, gpu_id: int, memory_mb: int, job_id: int) -> None:
        dev = self._devices[gpu_id]
        dev.used_memory_mb = max(0, dev.used_memory_mb - memory_mb)
        if job_id in dev.running_jobs:
            dev.running_jobs.remove(job_id)


class NVMLGPUMonitor(GPUMonitor):
    """Reads real GPU state via pynvml. Requires an actual NVIDIA GPU
    and the pynvml package (`pip install pynvml`). Use this once you're
    running on real hardware — the scheduler code that consumes this
    class doesn't need to change at all.

    NOTE: this reads real hardware state for monitoring/placement
    decisions. It does not itself launch or pin jobs to GPUs — that's
    done separately via CUDA_VISIBLE_DEVICES or `docker run --gpus`.
    """

    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handles = {
            i: pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        }

    def get_devices(self) -> list[GPUDevice]:
        devices = []
        for gpu_id, handle in self._handles.items():
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            devices.append(GPUDevice(
                id=gpu_id,
                total_memory_mb=mem.total // (1024 * 1024),
                used_memory_mb=mem.used // (1024 * 1024),
            ))
        return devices

    def get_compute_utilization_pct(self, gpu_id: int) -> float:
        """Real compute utilization (separate from memory) — useful as
        a second signal alongside memory-based placement."""
        util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handles[gpu_id])
        return float(util.gpu)
