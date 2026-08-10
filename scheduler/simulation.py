"""
Discrete-event simulation driver.

The old simulator advanced a fixed 1-second tick and asked "anything happen?"
at each step. That quantises everything: a job with duration 5.3s was released
at tick 6, holding memory it was not using for 0.7s. The error is systematic,
not random, so it inflated every utilization number in the same direction.

This driver keeps a heap of future events and jumps the clock straight to the
next one. Nothing is rounded, and empty stretches of time cost nothing to skip.
The analogy: a fixed tick checks the oven every minute; an event clock sets a
timer for exactly when the bread is done.

Between two consecutive events the pool cannot change - no job starts, none
finishes - so utilization is piecewise constant, and the time-weighted average
over any window is an exact integral rather than a sample mean. That matters:
event timestamps are irregular, so a plain mean over samples would silently
weight a busy millisecond the same as an idle minute.
"""
import heapq
import itertools
import math
from dataclasses import dataclass, field

from .core import Scheduler
from .gpu import MockGPUMonitor
from .logger import _NullLogger
from .models import Job
from .policy import Policy
from .pool import PlacementTracker

_ARRIVAL = 0
_COMPLETION = 1


class SimulationError(RuntimeError):
    """The trace cannot complete - usually a job larger than any GPU."""


@dataclass
class SimResult:
    policy: str
    seed: int | None
    makespan: float
    jobs_completed: int
    wait_times: list[float]
    # (time, utilization_pct) breakpoints; the value holds until the next entry
    timeline: list[tuple[float, float]] = field(default_factory=list)

    def utilization_over(self, start: float, end: float) -> float:
        """Time-weighted mean utilization across [start, end].

        Comparing policies on their own makespans is the trap the original
        project fell into: a run that finishes sooner has fewer trailing idle
        moments dragging its average down, so it looks better without having
        packed anything more tightly. Always integrate over a window that is
        the same for every policy being compared.
        """
        if end <= start or not self.timeline:
            return 0.0
        area = 0.0
        for i, (t, util) in enumerate(self.timeline):
            t_next = self.timeline[i + 1][0] if i + 1 < len(self.timeline) else end
            lo, hi = max(t, start), min(t_next, end)
            if hi > lo:
                area += util * (hi - lo)
        return area / (end - start)

    def wait_percentile(self, p: float) -> float:
        return percentile(self.wait_times, p)


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile.

    The original code did `sorted[int(len(sorted) * 0.99)]`, which for 60
    samples is index 59 - the maximum, not the 99th percentile. Any p above
    about 98 collapsed to max(), and a length where int(n*p) == n raised
    IndexError outright.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def simulate(
    policy: Policy,
    jobs: list[Job],
    gpu_count: int = 4,
    memory_per_gpu_mb: int = 16000,
    headroom_mb: int = 0,
    seed: int | None = None,
    logger=None,
) -> SimResult:
    """Run one trace through one policy and return the measurements.

    headroom_mb defaults to 0 here, unlike the live path: reserving safety
    margin against foreign processes is meaningless when no foreign processes
    exist, and a non-zero default would quietly shrink every simulated GPU.
    """
    monitor = MockGPUMonitor(gpu_count=gpu_count, memory_per_gpu_mb=memory_per_gpu_mb)
    tracker = PlacementTracker(monitor, headroom_mb=headroom_mb)
    logger = logger or _NullLogger()
    sched = Scheduler(policy=policy, tracker=tracker, logger=logger)

    capacity = memory_per_gpu_mb - headroom_mb
    too_big = [j for j in jobs if j.memory_mb > capacity]
    if too_big:
        raise SimulationError(
            f"{len(too_big)} job(s) need more than a whole GPU "
            f"({too_big[0].memory_mb}MB > {capacity}MB usable); the trace can never drain"
        )

    for job in jobs:
        sched.submit(job)

    # (time, tiebreak, kind, job_id) - the counter keeps ordering deterministic
    # when two events land on the same timestamp, which float durations make
    # rarer than you would think but never impossible.
    seq = itertools.count()
    events: list[tuple[float, int, int, int]] = [
        (job.arrival_time, next(seq), _ARRIVAL, job.id) for job in jobs
    ]
    heapq.heapify(events)

    timeline: list[tuple[float, float]] = [(0.0, 0.0)]
    prev_t = 0.0
    current_util = 0.0
    last_completion = 0.0

    while events:
        now = events[0][0]

        # everything landing on this exact timestamp resolves before we schedule
        while events and events[0][0] == now:
            _, _, kind, job_id = heapq.heappop(events)
            if kind is _COMPLETION:
                sched.complete(job_id, now)
                last_completion = now

        for job in sched.schedule(now):
            heapq.heappush(
                events, (now + job.duration_s, next(seq), _COMPLETION, job.id)
            )

        current_util = sched.utilization_pct()
        if timeline[-1][0] == now:
            timeline[-1] = (now, current_util)
        else:
            timeline.append((now, current_util))
        logger.log_snapshot(now, tracker.slots())
        prev_t = now

    if sched.queue or sched.running:
        stuck = len(sched.queue) + len(sched.running)
        raise SimulationError(
            f"{stuck} job(s) never completed under {policy.name}; "
            "the event loop drained with work outstanding"
        )

    return SimResult(
        policy=policy.name,
        seed=seed,
        makespan=last_completion,
        jobs_completed=len(sched.completed),
        wait_times=[j.wait_time for j in sched.completed if j.wait_time is not None],
        timeline=timeline,
    )
