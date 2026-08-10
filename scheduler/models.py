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
    duration_s: float       # simulation only: how long the job runs once placed
    priority: int = 0       # higher = more important, scheduled first
    arrival_time: float = 0.0
    id: int = field(default_factory=lambda: next(_job_id_counter))
    status: JobStatus = JobStatus.QUEUED

    name: str = ""
    # What to actually execute. Required by the live runner, ignored by the
    # simulator. duration_s is the mirror image: the simulator needs it, and
    # on real hardware it is at best a hint - the process decides when it is
    # done, and it can crash at 2s or hang past any estimate.
    command: list[str] | None = None

    # filled in once the scheduler places it
    gpu_id: int | None = None
    start_time: float | None = None
    completion_time: float | None = None
    exit_code: int | None = None

    @property
    def label(self) -> str:
        return self.name or f"job-{self.id}"

    @property
    def failed(self) -> bool:
        """A non-zero exit still frees the GPU; it just did not do the work.

        Keeping these separate matters for the live path: a fleet where every
        job crashes instantly has excellent makespan and has accomplished
        nothing.
        """
        return self.exit_code is not None and self.exit_code != 0

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
