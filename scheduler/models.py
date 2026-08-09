"""Core data models for the scheduler."""
from dataclasses import dataclass, field
from enum import Enum
import itertools


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"


_job_id_counter = itertools.count(1)


@dataclass
class Job:
    memory_mb: int          # GPU memory required, in MB
    duration_s: float       # how long the job runs once placed
    priority: int = 0       # higher = more important, scheduled first
    arrival_time: float = 0.0
    id: int = field(default_factory=lambda: next(_job_id_counter))
    status: JobStatus = JobStatus.QUEUED

    # filled in once the scheduler places it
    gpu_id: int | None = None
    start_time: float | None = None
    completion_time: float | None = None

    @property
    def wait_time(self) -> float | None:
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time

    @property
    def run_time(self) -> float | None:
        if self.start_time is None or self.completion_time is None:
            return None
        return self.completion_time - self.start_time


@dataclass
class GPUDevice:
    id: int
    total_memory_mb: int
    used_memory_mb: int = 0
    running_jobs: list[int] = field(default_factory=list)

    @property
    def free_memory_mb(self) -> int:
        return self.total_memory_mb - self.used_memory_mb

    @property
    def utilization_pct(self) -> float:
        """Memory-based utilization as a proxy — real deployment reads
        compute utilization from NVML too (see gpu_monitor.py)."""
        if self.total_memory_mb == 0:
            return 0.0
        return 100.0 * self.used_memory_mb / self.total_memory_mb
