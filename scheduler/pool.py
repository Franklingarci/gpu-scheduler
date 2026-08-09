"""
The scheduler's own book of what it placed where, reconciled against what
the hardware actually reports.

Why both. The scheduler knows what it reserved. The hardware knows what is
really resident. On a dedicated simulated pool those agree. On a real box
they diverge constantly:

  - another user's process is holding 4GB we never reserved
  - a job we launched has not allocated its memory yet, so NVML under-reports
  - a crashed job leaked memory the driver has not reclaimed
  - CUDA context overhead, a few hundred MB per process, that nobody budgets

Trusting either number alone puts you on a GPU that OOMs. So we take the
pessimistic view of both and keep a headroom margin on top.
"""
from dataclasses import dataclass

from .gpu import DeviceState, GPUMonitor
from .models import Job


@dataclass(frozen=True)
class Slot:
    """What a policy is allowed to see about one GPU when deciding."""
    id: int
    total_memory_mb: int
    free_memory_mb: int      # reconciled: safe to hand out
    reserved_memory_mb: int  # what we believe we placed
    observed_used_mb: int    # what the hardware reports
    compute_util_pct: float

    @property
    def utilization_pct(self) -> float:
        if self.total_memory_mb == 0:
            return 0.0
        return 100.0 * (self.total_memory_mb - self.free_memory_mb) / self.total_memory_mb


class PlacementTracker:
    """Records scheduler decisions and reconciles them with hardware readings.

    This is the piece that used to live on the monitor as allocate()/release().
    Moving it here is what unbroke the NVML swap.
    """

    def __init__(self, monitor: GPUMonitor, headroom_mb: int = 512):
        self.monitor = monitor
        self.headroom_mb = headroom_mb
        # gpu_id -> {job_id: memory_mb}
        self._reserved: dict[int, dict[int, int]] = {}

    def reserve(self, job: Job, gpu_id: int) -> None:
        self._reserved.setdefault(gpu_id, {})[job.id] = job.memory_mb

    def release(self, job: Job) -> None:
        if job.gpu_id is None:
            return
        self._reserved.get(job.gpu_id, {}).pop(job.id, None)

    def reserved_mb(self, gpu_id: int) -> int:
        return sum(self._reserved.get(gpu_id, {}).values())

    def job_ids_on(self, gpu_id: int) -> list[int]:
        return list(self._reserved.get(gpu_id, {}))

    def slots(self) -> list[Slot]:
        """Reconciled view of the pool. This is what policies decide against.

        The reconciliation is one line and it is the important line:

            free = min(hardware_says_free, total - we_reserved) - headroom

        Take whichever source is more pessimistic. In simulation the hardware
        term is the whole GPU (nothing external), so the reservation term
        wins and this reduces to ordinary bookkeeping. On real hardware,
        whichever number is scarier is the one that keeps you off a GPU that
        is about to OOM.
        """
        slots = []
        for dev in self.monitor.snapshot():
            slots.append(self._reconcile(dev))
        return slots

    def _reconcile(self, dev: DeviceState) -> Slot:
        reserved = self.reserved_mb(dev.id)
        by_hardware = dev.free_memory_mb
        by_book = dev.total_memory_mb - reserved
        free = max(0, min(by_hardware, by_book) - self.headroom_mb)
        return Slot(
            id=dev.id,
            total_memory_mb=dev.total_memory_mb,
            free_memory_mb=free,
            reserved_memory_mb=reserved,
            observed_used_mb=dev.used_memory_mb,
            compute_util_pct=dev.compute_util_pct,
        )
