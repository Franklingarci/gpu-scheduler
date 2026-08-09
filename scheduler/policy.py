"""
Placement policies, split along the two axes that actually vary independently:

  ORDERING - who gets considered next (FIFO, priority, backfill)
  FIT      - which GPU a chosen job lands on (first, best, worst)

Keeping them separate matters for more than tidiness. Compare "FIFO with
first-fit" against "priority with best-fit" and you have changed two things
at once, so a difference in the result tells you nothing about which one
caused it. With the axes split you can hold one fixed and sweep the other,
and the benchmark can finally answer whether the gain comes from ordering
or from packing.

Note the plan() interface shape. The obvious one is

    select_gpu(job) -> gpu_id | None

and it is a trap, because it can only answer "where does THIS job go".
Backfill has to reason about the queue as a whole - it needs to know which
job is at the head in order to decide who may jump the line. An interface
that sees one job at a time cannot express it. So policies take the whole
ready list and return a batch of placements.

Policies are pure: they read state and return decisions, never mutating the
pool. The caller applies the plan. That is what makes them trivial to test -
build a list of jobs, build a list of slots, assert on the placements, no
clock and no hardware involved.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .models import Job
from .pool import Slot


@dataclass(frozen=True)
class Placement:
    job_id: int
    gpu_id: int


# --------------------------------------------------------------------------
# Axis 1: fit - given a job that we have decided to run, which GPU?
# --------------------------------------------------------------------------

class FitStrategy(ABC):
    """Chooses a GPU for one job from the currently available room.

    `remaining` is a mutable gpu_id -> free_mb budget for a single planning
    pass, not the live pool. See Policy._remaining for why.
    """

    name: str = "fit"

    @abstractmethod
    def choose(
        self, need_mb: int, remaining: dict[int, int], exclude: int | None = None
    ) -> int | None:
        ...

    @staticmethod
    def _candidates(need_mb: int, remaining: dict[int, int], exclude: int | None) -> list[int]:
        return [
            gid for gid, free in remaining.items()
            if free >= need_mb and gid != exclude
        ]


class FirstFit(FitStrategy):
    """Lowest device id with enough room. Stop looking once it fits.

    The cheap one: O(n) with an early exit, no scan of the whole pool. Its
    real weakness is that it hammers low-numbered GPUs, so gpu0 stays hot
    and the tail of the pool sits cold.
    """

    name = "first_fit"

    def choose(self, need_mb, remaining, exclude=None):
        for gpu_id in sorted(remaining):
            if remaining[gpu_id] >= need_mb and gpu_id != exclude:
                return gpu_id
        return None


class BestFit(FitStrategy):
    """Tightest sufficient GPU - the one with the least free memory that still fits.

    Packs small jobs together so whole GPUs stay open for large ones. Same
    idea as a best-fit memory allocator. The classic failure mode is
    fragmentation: it leaves lots of slivers too small to be useful.
    """

    name = "best_fit"

    def choose(self, need_mb, remaining, exclude=None):
        candidates = self._candidates(need_mb, remaining, exclude)
        if not candidates:
            return None
        # gpu id breaks ties so placement stays deterministic across runs
        return min(candidates, key=lambda gid: (remaining[gid], gid))


class WorstFit(FitStrategy):
    """Emptiest sufficient GPU. Spreads load instead of packing it.

    In classic memory allocation worst-fit is the punchline, because leaving
    the largest possible remainder is supposed to keep it usable and mostly
    does not. GPUs change the argument, and this is worth understanding
    because it is where the analogy to memory allocators breaks down.

    Memory is one resource. A GPU is at least two: memory AND compute. Two
    jobs whose memory both fit on one card still contend for the same SMs,
    so both run slower than they would alone. Best-fit maximises that
    contention by design - packing tightly is the goal. Worst-fit spreads
    jobs across cards and avoids it.

    So the expectation is that best-fit wins on memory utilization and
    worst-fit wins on per-job runtime. Our simulator models memory only, so
    it will flatter best-fit. Naming that up front is the honest thing to
    do; measuring it needs the real-hardware phase.
    """

    name = "worst_fit"

    def choose(self, need_mb, remaining, exclude=None):
        candidates = self._candidates(need_mb, remaining, exclude)
        if not candidates:
            return None
        return max(candidates, key=lambda gid: (remaining[gid], -gid))


FITS: dict[str, type[FitStrategy]] = {
    f.name: f for f in (FirstFit, BestFit, WorstFit)
}


# --------------------------------------------------------------------------
# Axis 2: ordering - who gets considered, and in what order
# --------------------------------------------------------------------------

def effective_priority(job: Job, now: float, aging_rate: float) -> float:
    """Static priority plus `aging_rate` points per second spent waiting.

    Without this, priority scheduling starves. A steady arrival of priority-3
    work means a priority-0 job is never the best candidate and waits forever.
    Aging bounds that wait: queue long enough and any job eventually outranks
    anything. `aging_rate=0` gives strict priority, useful mainly for
    demonstrating the starvation it causes.
    """
    waited = max(0.0, now - job.arrival_time)
    return job.priority + aging_rate * waited


class Policy(ABC):
    """Decides placements. Pure - reads state, returns decisions."""

    ordering: str = "policy"

    def __init__(self, fit: FitStrategy | None = None):
        self.fit = fit or BestFit()

    @property
    def name(self) -> str:
        return f"{self.ordering}+{self.fit.name}"

    @abstractmethod
    def plan(self, ready: list[Job], slots: list[Slot], now: float) -> list[Placement]:
        """Jobs that have arrived and are still queued, plus current pool state."""
        ...

    @staticmethod
    def _remaining(slots: list[Slot]) -> dict[int, int]:
        """Mutable free-memory budget for one planning pass.

        A plan can place several jobs at once, so we decrement as we go.
        Reading `slot.free_memory_mb` fresh for every job in the same pass
        would happily put four 12GB jobs on one 16GB GPU.
        """
        return {s.id: s.free_memory_mb for s in slots}


class FIFOPolicy(Policy):
    """Strict arrival order. The head of the queue blocks everything behind it.

    The honest baseline, and stricter than it looks. If the oldest job needs
    12GB and no GPU has 12GB free, the pool sits idle even with a 2GB job
    waiting right behind it. That stall is head-of-line blocking and it is
    the specific waste the other policies exist to recover.

    Worth flagging: the previous version of this project called its baseline
    FIFO but skipped past jobs that did not fit. That skip is already a crude
    backfill, which is a large part of why the "smarter" policy could not
    beat it.
    """

    ordering = "fifo"

    def plan(self, ready, slots, now):
        remaining = self._remaining(slots)
        placements = []
        for job in sorted(ready, key=lambda j: (j.arrival_time, j.id)):
            gpu_id = self.fit.choose(job.memory_mb, remaining)
            if gpu_id is None:
                break  # the block: nothing behind this job is considered
            remaining[gpu_id] -= job.memory_mb
            placements.append(Placement(job.id, gpu_id))
        return placements


class PriorityPolicy(Policy):
    """Priority order, skipping any job that does not currently fit.

    Two changes from FIFO, and the benchmark should separate them: it orders
    by priority rather than arrival, and it skips rather than stalls. The
    skip is what recovers most of the idle time.
    """

    ordering = "priority"

    def __init__(self, fit: FitStrategy | None = None, aging_rate: float = 0.0):
        super().__init__(fit)
        self.aging_rate = aging_rate

    def plan(self, ready, slots, now):
        remaining = self._remaining(slots)
        placements = []
        for job in _by_priority(ready, now, self.aging_rate):
            gpu_id = self.fit.choose(job.memory_mb, remaining)
            if gpu_id is None:
                continue  # skip, do not block
            remaining[gpu_id] -= job.memory_mb
            placements.append(Placement(job.id, gpu_id))
        return placements


class BackfillPolicy(Policy):
    """Priority order, but a job may jump the line only if it does not delay
    the current head job.

    What production schedulers actually do; Slurm calls it EASY backfill.
    The first job that does not fit becomes the reservation holder, and one
    GPU is set aside to accumulate room for it. Everything behind it may run
    right now on any OTHER card.

    The guarantee is the point: the head job starts no later than it would
    have under strict priority, so backfilling costs it nothing. You are
    filling holes that would have been idle, not stealing turns.
    """

    ordering = "backfill"

    def __init__(self, fit: FitStrategy | None = None, aging_rate: float = 0.0):
        super().__init__(fit)
        self.aging_rate = aging_rate

    def plan(self, ready, slots, now):
        remaining = self._remaining(slots)
        placements = []
        reserved_for_blocked: int | None = None

        for job in _by_priority(ready, now, self.aging_rate):
            if reserved_for_blocked is None:
                gpu_id = self.fit.choose(job.memory_mb, remaining)
                if gpu_id is not None:
                    remaining[gpu_id] -= job.memory_mb
                    placements.append(Placement(job.id, gpu_id))
                else:
                    reserved_for_blocked = _reservation_target(remaining)
                continue

            gpu_id = self.fit.choose(job.memory_mb, remaining, exclude=reserved_for_blocked)
            if gpu_id is not None:
                remaining[gpu_id] -= job.memory_mb
                placements.append(Placement(job.id, gpu_id))
        return placements


ORDERINGS: dict[str, type[Policy]] = {
    p.ordering: p for p in (FIFOPolicy, PriorityPolicy, BackfillPolicy)
}


def build(ordering: str, fit: str, aging_rate: float = 0.0) -> Policy:
    """Construct one cell of the ordering x fit matrix by name."""
    cls = ORDERINGS[ordering]
    fit_obj = FITS[fit]()
    if cls is FIFOPolicy:
        return cls(fit_obj)
    return cls(fit_obj, aging_rate=aging_rate)


def _by_priority(ready: list[Job], now: float, aging_rate: float) -> list[Job]:
    return sorted(
        ready,
        key=lambda j: (-effective_priority(j, now, aging_rate), j.arrival_time, j.id),
    )


def _reservation_target(remaining: dict[int, int]) -> int | None:
    """Which GPU we save for the blocked job.

    Simplification worth naming: we pick the emptiest card, assuming it is
    closest to fitting. Proper EASY backfill computes the earliest time each
    GPU will have enough room from the running jobs' remaining durations, and
    reserves that one. That needs a runtime estimate per job, which is a real
    operational burden - Slurm makes users declare a time limit precisely for
    this. Revisit once the benchmark shows what the approximation costs.
    """
    if not remaining:
        return None
    return max(remaining, key=lambda gid: (remaining[gid], -gid))
