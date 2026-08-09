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


# GPUDevice used to live here. It carried both hardware facts
# (total/used memory) and scheduler bookkeeping (running_jobs), which is the
# conflation that made NVML impossible to swap in. It is now two types:
# gpu.DeviceState for what the hardware reports, pool.Slot for the
# reconciled view a policy decides against.
