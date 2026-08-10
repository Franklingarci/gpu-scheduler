"""
The live driver: same policies, same core, real processes on real GPUs.

What changes from the simulator is exactly two things, which is the claim the
whole sensor/ledger split was made to support:

  Time comes from the wall clock instead of an event heap.
  Placing a job launches a process instead of scheduling a completion event.

Nothing in policy.py or core.py is aware of the difference.

Three things here have no counterpart in simulation, and each one is a way
real hardware misbehaves:

  A job finishes when its PROCESS exits, not when duration_s elapses. It can
  crash in two seconds or hang past any estimate. The core is told by
  complete(); it never infers.

  A launch can fail outright - bad binary, missing file, permission denied.
  The reservation is already recorded at that point, so it has to be undone
  or the ledger leaks memory that nothing will ever free.

  Somebody else's process can be holding memory we never reserved, which is
  what PlacementTracker's reconciliation is for. Headroom defaults to 512MB
  here rather than the simulator's 0, because CUDA contexts alone cost a few
  hundred MB that nobody budgets for.
"""
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field

from .core import Scheduler
from .gpu import GPUMonitor
from .logger import _NullLogger
from .models import Job
from .policy import Policy
from .pool import PlacementTracker


class LaunchError(RuntimeError):
    """A job's command could not be started at all."""


class StalledError(RuntimeError):
    """Work is queued, nothing is running, and nothing can be placed.

    The live counterpart of the simulator's unplaceable-job guard. Without it
    the poll loop spins forever: a job larger than any card, or a reservation
    that leaked because a rollback was missed, leaves the pool permanently
    unable to satisfy the queue and the runner never notices.

    Found by mutation testing - removing the launch-failure rollback made the
    suite hang rather than fail, which is the worse of the two outcomes.
    """


@dataclass
class LiveResult:
    completed: list[Job] = field(default_factory=list)
    failed: list[Job] = field(default_factory=list)
    launch_failures: list[tuple[Job, str]] = field(default_factory=list)
    wall_time_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failed and not self.launch_failures


class LiveRunner:
    """Runs real jobs on a real pool under a chosen policy.

    poll_interval is the loop's granularity. Unlike the simulator's old fixed
    tick this introduces no measurement bias, because nothing here is being
    measured against a model - the clock is the clock. It only bounds how
    quickly a freed GPU gets reused.
    """

    def __init__(
        self,
        policy: Policy,
        monitor: GPUMonitor,
        headroom_mb: int = 512,
        poll_interval: float = 0.5,
        logger=None,
        on_event=None,
    ):
        self.tracker = PlacementTracker(monitor, headroom_mb=headroom_mb)
        self.scheduler = Scheduler(
            policy=policy, tracker=self.tracker, logger=logger or _NullLogger()
        )
        self.poll_interval = poll_interval
        self.on_event = on_event or (lambda *a: None)
        self._procs: dict[int, subprocess.Popen] = {}

    def run(self, jobs: list[Job]) -> LiveResult:
        missing = [j for j in jobs if not j.command]
        if missing:
            raise LaunchError(
                f"{len(missing)} job(s) have no command: "
                f"{', '.join(j.label for j in missing[:3])}"
            )

        result = LiveResult()
        for job in jobs:
            self.scheduler.submit(job)

        started_at = time.monotonic()
        try:
            while self.scheduler.pending:
                now = time.monotonic() - started_at
                self._reap(now, result)
                self._launch(now, result)
                self._check_stalled(now)

                if self.scheduler.pending:
                    time.sleep(self.poll_interval)
        except BaseException:
            # Ctrl-C, or anything else: never leave orphaned GPU processes
            # holding memory after the scheduler that placed them is gone.
            self._terminate_all()
            raise

        result.wall_time_s = time.monotonic() - started_at
        result.completed = list(self.scheduler.completed)
        result.failed = [j for j in self.scheduler.completed if j.failed]
        return result

    # -- the two halves of one poll ----------------------------------------

    def _reap(self, now: float, result: LiveResult) -> None:
        for job_id, proc in list(self._procs.items()):
            code = proc.poll()
            if code is None:
                continue
            del self._procs[job_id]
            job = self.scheduler.running.get(job_id)
            if job is not None:
                job.exit_code = code
            self.scheduler.complete(job_id, now)
            self.on_event("finished", job, code)

    def _launch(self, now: float, result: LiveResult) -> None:
        for job in self.scheduler.schedule(now):
            env = dict(os.environ)
            # The scheduler's gpu_id is an NVML device index, which is exactly
            # what CUDA_VISIBLE_DEVICES takes. The job sees it as device 0.
            env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)
            try:
                self._procs[job.id] = subprocess.Popen(job.command, env=env)
            except OSError as exc:
                # The reservation is already on the books. Roll it back, or
                # this GPU loses memory permanently for the rest of the run.
                self.scheduler.running.pop(job.id, None)
                self.tracker.release(job)
                job.gpu_id = None
                job.start_time = None
                result.launch_failures.append((job, str(exc)))
                self.on_event("launch_failed", job, str(exc))
                continue
            self.on_event("started", job, job.gpu_id)

    def _check_stalled(self, now: float) -> None:
        """Nothing running, nothing launchable, and arrived work still queued.

        Jobs whose arrival_time is in the future are excluded: waiting for one
        of those is the loop working correctly, not a deadlock.
        """
        if self._procs:
            return
        arrived = [j for j in self.scheduler.queue if j.arrival_time <= now]
        if not arrived:
            return

        slots = self.tracker.slots()
        largest = max((s.free_memory_mb for s in slots), default=0)
        smallest = min(j.memory_mb for j in arrived)
        if smallest <= largest:
            # Idle right now but something fits, so the next poll will place
            # it. This happens legitimately after a launch failure rolls back
            # a reservation partway through a pass.
            return

        raise StalledError(
            f"{len(arrived)} job(s) queued, none running, and nothing fits: "
            f"smallest needs {smallest}MB, largest free is {largest}MB. "
            "Either a job is bigger than any GPU, or a reservation leaked."
        )

    def _terminate_all(self) -> None:
        for proc in self._procs.values():
            try:
                proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
        for proc in self._procs.values():
            try:
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass
        self._procs.clear()
