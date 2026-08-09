"""
Scheduler core: decides which pending job goes on which GPU.

Two strategies:
  - BestFitScheduler: the actual project — packs jobs onto the GPU with
    the tightest sufficient free memory, processed in priority order.
  - FIFOScheduler: the naive baseline — first job in line takes the
    first GPU with enough room, no packing intelligence. This is what
    you benchmark BestFitScheduler against.
"""
from abc import ABC, abstractmethod
from .models import Job, JobStatus
from .gpu_monitor import MockGPUMonitor


class BaseScheduler(ABC):
    def __init__(self, monitor: MockGPUMonitor):
        self.monitor = monitor
        self.queue: list[Job] = []
        self.running: list[Job] = []
        self.completed: list[Job] = []

    def submit(self, job: Job) -> None:
        self.queue.append(job)

    @abstractmethod
    def _select_gpu(self, job: Job) -> int | None:
        """Return a gpu_id this job fits on, or None if nothing fits."""
        ...

    def tick(self, now: float) -> None:
        """One scheduling cycle: finish jobs whose duration has elapsed,
        then try to place pending jobs. Call this on a loop with your
        simulation clock (or a real wall-clock timer in production)."""
        self._complete_finished_jobs(now)
        self._place_pending_jobs(now)

    def _complete_finished_jobs(self, now: float) -> None:
        still_running = []
        for job in self.running:
            if job.start_time is not None and now - job.start_time >= job.duration_s:
                job.completion_time = now
                job.status = JobStatus.COMPLETED
                self.monitor.release(job.gpu_id, job.memory_mb, job.id)
                self.completed.append(job)
            else:
                still_running.append(job)
        self.running = still_running

    def _place_pending_jobs(self, now: float) -> None:
        still_queued = []
        # only jobs that have actually arrived are eligible this tick
        eligible = [j for j in self.queue if j.arrival_time <= now]
        not_yet_arrived = [j for j in self.queue if j.arrival_time > now]
        # priority order: highest priority first, then earliest arrival
        ordered = sorted(eligible, key=lambda j: (-j.priority, j.arrival_time))
        for job in ordered:
            gpu_id = self._select_gpu(job)
            if gpu_id is not None:
                self.monitor.allocate(gpu_id, job.memory_mb, job.id)
                job.gpu_id = gpu_id
                job.start_time = now
                job.status = JobStatus.RUNNING
                self.running.append(job)
            else:
                still_queued.append(job)
        self.queue = still_queued + not_yet_arrived


class BestFitScheduler(BaseScheduler):
    """Tightest-fit placement: pick the GPU with the LEAST free memory
    that still satisfies the job. This packs small jobs together and
    preserves whole GPUs for larger ones — the same idea as best-fit
    memory allocators."""

    def _select_gpu(self, job: Job) -> int | None:
        candidates = [
            d for d in self.monitor.get_devices()
            if d.free_memory_mb >= job.memory_mb
        ]
        if not candidates:
            return None
        best = min(candidates, key=lambda d: d.free_memory_mb)
        return best.id


class FIFOScheduler(BaseScheduler):
    """Naive baseline: first GPU with enough room, in device-id order.
    No packing intelligence — this is the thing BestFitScheduler should
    beat on utilization."""

    def _select_gpu(self, job: Job) -> int | None:
        for d in sorted(self.monitor.get_devices(), key=lambda d: d.id):
            if d.free_memory_mb >= job.memory_mb:
                return d.id
        return None
