"""Placement policy behaviour. No clock, no hardware, no scheduler."""
import pytest

from scheduler import policy as P


# ---------------------------------------------------------------- fit axis

def test_the_three_fits_pick_three_different_gpus():
    """The defining property of the fit axis.

    gpu0 is deliberately neither the tightest nor the emptiest, so a bug that
    collapsed two strategies into one would show up here. With only two GPUs
    first-fit and best-fit frequently agree by coincidence, which is why this
    uses three.
    """
    remaining = {0: 10000, 1: 6000, 2: 16000}
    assert P.FirstFit().choose(4000, dict(remaining)) == 0   # lowest id
    assert P.BestFit().choose(4000, dict(remaining)) == 1    # tightest
    assert P.WorstFit().choose(4000, dict(remaining)) == 2   # emptiest


@pytest.mark.parametrize("fit", [P.FirstFit(), P.BestFit(), P.WorstFit()])
def test_fit_returns_none_when_nothing_is_big_enough(fit):
    assert fit.choose(20000, {0: 10000, 1: 6000}) is None


@pytest.mark.parametrize("fit", [P.FirstFit(), P.BestFit(), P.WorstFit()])
def test_fit_honours_exclude(fit):
    """Backfill's reservation depends on this: the held card must be untouchable."""
    assert fit.choose(1000, {0: 8000}, exclude=0) is None


@pytest.mark.parametrize("fit", [P.FirstFit(), P.BestFit(), P.WorstFit()])
def test_a_job_that_exactly_fills_a_gpu_still_fits(fit):
    """Off-by-one guard: `free >= need`, not `free > need`."""
    assert fit.choose(8000, {0: 8000}) == 0


# ----------------------------------------------------------- ordering axis

def test_fifo_blocks_behind_a_job_that_does_not_fit(job, slot):
    """The defining property of FIFO, and the bug in the original project.

    The old baseline used `continue` here instead of `break`, so it skipped
    past the stuck job. That skip is already a crude backfill, which is much
    of why the 'smarter' policy could not beat it.
    """
    slots = [slot(0, 4000), slot(1, 4000)]
    jobs = [
        job(1, 12000, arrival_time=0.0),   # fits nowhere
        job(2, 2000, arrival_time=1.0),    # would fit, but is behind the block
        job(3, 2000, arrival_time=2.0),
    ]
    assert P.FIFOPolicy().plan(jobs, slots, now=10.0) == []


def test_priority_skips_past_a_job_that_does_not_fit(job, slot):
    slots = [slot(0, 4000), slot(1, 4000)]
    jobs = [
        job(1, 12000, priority=3),
        job(2, 2000, priority=1),
        job(3, 2000, priority=1),
    ]
    placed = {p.job_id for p in P.PriorityPolicy().plan(jobs, slots, now=10.0)}
    assert placed == {2, 3}


def test_backfill_holds_a_gpu_for_the_blocked_job(job, slot):
    """Small jobs run, but never on the reserved card.

    That is what makes backfilling free: the head job's card is accumulating
    room untouched, so it starts no later than it would have under strict
    priority.
    """
    slots = [slot(0, 4000), slot(1, 4000)]
    jobs = [
        job(1, 12000, priority=3),
        job(2, 2000, priority=1),
        job(3, 2000, priority=1),
    ]
    placements = P.BackfillPolicy().plan(jobs, slots, now=10.0)

    assert {p.job_id for p in placements} == {2, 3}
    used = {p.gpu_id for p in placements}
    assert len(used) == 1, "backfilled jobs must share the un-reserved card"
    assert 1 not in {p.job_id for p in placements}


def test_backfill_places_normally_when_nothing_is_blocked(job, slot):
    slots = [slot(0, 16000), slot(1, 16000)]
    jobs = [job(1, 4000), job(2, 4000)]
    assert len({p.job_id for p in P.BackfillPolicy().plan(jobs, slots, now=0.0)}) == 2


# ----------------------------------------------------- cross-cutting rules

@pytest.mark.parametrize("ordering", sorted(P.ORDERINGS))
@pytest.mark.parametrize("fit", sorted(P.FITS))
def test_one_pass_never_overcommits_a_gpu(ordering, fit, job, slot):
    """The `_remaining` budget exists for exactly this.

    A plan places several jobs at once. Re-reading each slot's free memory per
    job would cheerfully put four 12GB jobs on one 16GB card, and the run would
    then produce entirely plausible, entirely wrong numbers.
    """
    slots = [slot(0, 16000)]
    jobs = [job(i, 12000, arrival_time=0.0) for i in range(1, 5)]
    placements = P.build(ordering, fit).plan(jobs, slots, now=0.0)
    assert len(placements) <= 1


@pytest.mark.parametrize("ordering", sorted(P.ORDERINGS))
@pytest.mark.parametrize("fit", sorted(P.FITS))
def test_policies_never_place_a_job_twice(ordering, fit, job, slot):
    slots = [slot(0, 16000), slot(1, 16000)]
    jobs = [job(i, 2000) for i in range(1, 6)]
    ids = [p.job_id for p in P.build(ordering, fit).plan(jobs, slots, now=0.0)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("ordering", sorted(P.ORDERINGS))
@pytest.mark.parametrize("fit", sorted(P.FITS))
def test_planning_is_deterministic(ordering, fit, job, slot):
    """Same inputs, same plan. Ties break on gpu id and job id for this reason -
    a set-iteration order leaking into placements would make seeded runs
    irreproducible in a way that is miserable to debug."""
    slots = [slot(0, 8000), slot(1, 8000), slot(2, 8000)]
    jobs = [job(i, 3000, priority=i % 2) for i in range(1, 7)]
    pol = P.build(ordering, fit)
    first = pol.plan(jobs, slots, now=5.0)
    assert first == pol.plan(jobs, slots, now=5.0)


@pytest.mark.parametrize("ordering", sorted(P.ORDERINGS))
@pytest.mark.parametrize("fit", sorted(P.FITS))
def test_policies_do_not_mutate_their_inputs(ordering, fit, job, slot):
    """Purity is the property the whole design leans on."""
    slots = [slot(0, 8000), slot(1, 4000)]
    jobs = [job(1, 3000), job(2, 3000)]
    before_slots = [(s.id, s.free_memory_mb) for s in slots]
    before_jobs = [(j.id, j.gpu_id, j.start_time, j.status) for j in jobs]

    P.build(ordering, fit).plan(jobs, slots, now=0.0)

    assert [(s.id, s.free_memory_mb) for s in slots] == before_slots
    assert [(j.id, j.gpu_id, j.start_time, j.status) for j in jobs] == before_jobs


# ------------------------------------------------------------------ aging

def test_without_aging_a_low_priority_job_never_wins(job):
    """Strict priority starves, by construction, no matter how long you wait."""
    waiting = job(1, 1000, priority=0, arrival_time=0.0)
    fresh = job(2, 1000, priority=3, arrival_time=9_999.0)
    assert P.effective_priority(waiting, 10_000.0, 0.0) < P.effective_priority(
        fresh, 10_000.0, 0.0
    )


def test_aging_lets_a_waiting_job_overtake_a_higher_priority_newcomer(job):
    """Which is what bounds the wait rather than merely shortening it."""
    waiting = job(1, 1000, priority=0, arrival_time=0.0)
    fresh = job(2, 1000, priority=3, arrival_time=100.0)
    rate = 0.1  # one priority point per 10s waited
    assert P.effective_priority(waiting, 100.0, rate) > P.effective_priority(
        fresh, 100.0, rate
    )


def test_aging_changes_which_job_gets_the_last_slot(job, slot):
    slots = [slot(0, 4000)]
    old_low = job(1, 4000, priority=0, arrival_time=0.0)
    new_high = job(2, 4000, priority=3, arrival_time=99.0)

    strict = P.PriorityPolicy(aging_rate=0.0).plan([old_low, new_high], slots, now=100.0)
    aged = P.PriorityPolicy(aging_rate=0.1).plan([old_low, new_high], slots, now=100.0)

    assert strict[0].job_id == 2
    assert aged[0].job_id == 1


# ------------------------------------------------------------- construction

def test_build_covers_the_whole_matrix():
    names = {P.build(o, f).name for o in P.ORDERINGS for f in P.FITS}
    assert len(names) == 9
    assert "backfill+worst_fit" in names
