"""
The scheduler itself: holds the queue, asks the policy what to start, and
applies the answer.

Two things it deliberately does NOT do, and both are what let the same class
serve the benchmark and real hardware:

  It never reads a clock. `now` arrives as an argument. Whoever calls
  schedule() decides what time it is - a simulated event clock or
  time.monotonic().

  It never decides that a job has finished. A driver calls complete() when
  it knows. In simulation that is duration_s elapsing; on real hardware it
  is a process exiting, which can happen early on a crash or late on a
  hang. A core that inferred completion from duration_s would be wrong on
  every real job that did not run exactly as long as predicted.
"""
from dataclasses import dataclass

from .logger import _NullLogger
from .models import Job, JobStatus
from .policy import Policy
from .pool import PlacementTracker


class SchedulerError(RuntimeError):
    """A policy returned a placement the scheduler cannot honour."""


@dataclass
class Scheduler:
    policy: Policy
    tracker: PlacementTracker
    logger: object = None

    def __post_init__(self):
        self.queue: list[Job] = []
        self.running: dict[int, Job] = {}
        self.completed: list[Job] = []
        if self.logger is None:
            self.logger = _NullLogger()

    # -- queue management ---------------------------------------------------

    def submit(self, job: Job) -> None:
        self.queue.append(job)

    @property
    def pending(self) -> bool:
        """Is there anything left to do? Drivers loop on this."""
        return bool(self.queue or self.running)

    # -- the one interesting method ----------------------------------------

    def schedule(self, now: float) -> list[Job]:
        """Ask the policy what should start, apply it, return what started.

        Only jobs that have actually arrived are offered. A job with
        arrival_time in the future is invisible to the policy, which is what
        stops a simulation from placing work before it was submitted.
        """
        ready = [j for j in self.queue if j.arrival_time <= now]
        if not ready:
            return []

        slots = self.tracker.slots()
        placements = self.policy.plan(ready, slots, now)
        if not placements:
            return []

        by_id = {j.id: j for j in ready}
        budget = {s.id: s.free_memory_mb for s in slots}
        started: list[Job] = []

        for p in placements:
            job = by_id.get(p.job_id)
            # These checks exist because a policy bug is otherwise silent: the
            # tracker would happily record an impossible placement and the
            # run would produce plausible, wrong numbers. Loud beats subtle.
            if job is None:
                raise SchedulerError(
                    f"{self.policy.name} placed job {p.job_id}, which is not ready"
                )
            if p.gpu_id not in budget:
                raise SchedulerError(
                    f"{self.policy.name} placed job {p.job_id} on unknown gpu {p.gpu_id}"
                )
            if job.memory_mb > budget[p.gpu_id]:
                raise SchedulerError(
                    f"{self.policy.name} over-committed gpu {p.gpu_id}: "
                    f"job {p.job_id} needs {job.memory_mb}MB, "
                    f"{budget[p.gpu_id]}MB left in this pass"
                )

            budget[p.gpu_id] -= job.memory_mb
            self.tracker.reserve(job, p.gpu_id)
            job.gpu_id = p.gpu_id
            job.start_time = now
            job.status = JobStatus.RUNNING
            self.running[job.id] = job
            started.append(job)

        placed = {j.id for j in started}
        self.queue = [j for j in self.queue if j.id not in placed]
        return started

    def complete(self, job_id: int, now: float) -> Job:
        """A driver telling us a job has finished. Frees its reservation."""
        job = self.running.pop(job_id, None)
        if job is None:
            raise SchedulerError(f"job {job_id} is not running")
        job.completion_time = now
        job.status = JobStatus.COMPLETED
        self.tracker.release(job)
        self.completed.append(job)
        self.logger.log_job(job)
        return job

    # -- observation --------------------------------------------------------

    def utilization_pct(self) -> float:
        """Pool-wide memory utilization right now, as a share of total capacity.

        Averaging the per-GPU percentages instead would weight a small card
        the same as a big one. Summing bytes is the honest form and stays
        correct on a mixed pool.
        """
        slots = self.tracker.slots()
        total = sum(s.total_memory_mb for s in slots)
        if total == 0:
            return 0.0
        used = sum(s.total_memory_mb - s.free_memory_mb for s in slots)
        return 100.0 * used / total
