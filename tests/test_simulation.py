"""The discrete-event driver, and the measurement properties the project rests on."""
import pytest

from scheduler import policy as P
from scheduler.job_generator import generate_jobs, load_factor
from scheduler.simulation import SimulationError, percentile, simulate

MATRIX = [(o, f) for o in sorted(P.ORDERINGS) for f in sorted(P.FITS)]


def run(ordering="priority", fit="best_fit", seed=42, jobs=40, gpus=4, **kw):
    trace = generate_jobs(count=jobs, seed=seed)
    return simulate(P.build(ordering, fit), trace, gpu_count=gpus, seed=seed, **kw)


# ------------------------------------------------------------- correctness

@pytest.mark.parametrize("ordering,fit", MATRIX)
def test_every_policy_drains_the_whole_trace(ordering, fit):
    result = run(ordering, fit)
    assert result.jobs_completed == 40
    assert result.makespan > 0


@pytest.mark.parametrize("ordering,fit", MATRIX)
def test_no_job_ever_starts_before_it_arrives(ordering, fit):
    """Wait time is start minus arrival, so a negative wait means the
    simulation placed work that had not been submitted yet."""
    assert all(w >= 0 for w in run(ordering, fit).wait_times)


def test_a_job_larger_than_any_gpu_raises_instead_of_hanging():
    """Otherwise the event loop drains and the job silently never runs."""
    trace = generate_jobs(count=5, seed=1, memory_range_mb=(20000, 20000))
    with pytest.raises(SimulationError, match="more than a whole GPU"):
        simulate(P.build("fifo", "best_fit"), trace, gpu_count=2, memory_per_gpu_mb=16000)


# ---------------------------------------------------------- reproducibility

@pytest.mark.parametrize("ordering,fit", MATRIX)
def test_the_same_seed_gives_the_same_answer(ordering, fit):
    assert run(ordering, fit, seed=7).makespan == run(ordering, fit, seed=7).makespan


def test_results_do_not_depend_on_what_ran_before_them():
    """Regression test for a reproducibility bug that would have poisoned the sweep.

    models._job_id_counter is process-global, so the second trace generated in
    a run once received ids 61..120 where the first got 1..60 - same seed,
    different ids. Policies break ties on job id, so identical seeds scheduled
    differently depending on execution order. Seeding is worthless if results
    still depend on what came first.
    """
    alone = run(seed=42).makespan

    for other in (1, 2, 3, 99):
        run(seed=other)
    after_others = run(seed=42).makespan

    assert alone == after_others


def test_generated_ids_are_one_through_n_every_time():
    assert [j.id for j in generate_jobs(20, seed=5)] == list(range(1, 21))
    assert [j.id for j in generate_jobs(20, seed=6)] == list(range(1, 21))


# -------------------------------------------------------------- event clock

def test_the_clock_is_not_quantised_to_whole_seconds():
    """Regression test against reintroducing a fixed tick.

    The old simulator advanced 1.0s per step, so a 5.3s job was released at
    tick 6 and held memory it was not using for 0.7s. The bias was systematic
    and inflated every utilization figure. Job durations here are continuous,
    so an event-driven makespan should essentially never land on a whole
    second - if it does, something is rounding.
    """
    makespan = run(jobs=60).makespan
    assert makespan != round(makespan)


def test_time_weighted_utilization_ignores_sample_spacing():
    """Event timestamps are irregular, so a plain mean over the timeline would
    weight a busy millisecond the same as an idle minute. Integrating over a
    window that contains no activity must give zero regardless of how many
    breakpoints happen to sit nearby."""
    result = run()
    assert result.utilization_over(result.makespan + 100, result.makespan + 200) == 0.0
    assert result.utilization_over(5.0, 5.0) == 0.0


# ------------------------------------------------------ the headline finding

@pytest.mark.parametrize("ordering,fit", MATRIX)
def test_full_window_utilization_is_identical_for_every_policy(ordering, fit):
    """Memory utilization cannot distinguish these policies. It is an identity.

    A trace holds a fixed sum(memory_mb * duration_s). Integrated over any
    window containing every completion, utilization is

        work / (capacity * window)

    with no policy term in it. The original project reported mean utilization
    as its headline result; the apparent 3.4pp win came entirely from
    averaging each policy over its own makespan, so whoever finished first
    scored highest.

    Pinning this as a test means anyone who later "improves" utilization has
    either changed the trace or made a measurement error.
    """
    trace = generate_jobs(count=40, seed=42)
    work = sum(j.memory_mb * j.duration_s for j in trace)
    capacity = 4 * 16000
    window = 400.0  # comfortably past every policy's makespan

    result = run(ordering, fit, seed=42)
    assert result.makespan < window
    expected = 100.0 * work / (capacity * window)
    assert result.utilization_over(0, window) == pytest.approx(expected, rel=1e-9)


def test_makespan_and_wait_times_do_vary_between_policies():
    """The counterpart: the metrics the benchmark should actually report."""
    makespans = {f"{o}+{f}": run(o, f, seed=42).makespan for o, f in MATRIX}
    assert len(set(makespans.values())) > 1, "ordering must change the outcome"


def test_fifo_drains_slower_than_priority():
    """Head-of-line blocking has a measurable cost; if this ever inverts, the
    baseline has started skipping again like the original one did."""
    assert run("fifo", "best_fit").makespan > run("priority", "best_fit").makespan


# -------------------------------------------------------------- percentiles

def test_percentile_uses_nearest_rank():
    """Regression test for the original p99.

    `sorted[int(len(sorted) * 0.99)]` on 60 samples is index 59 - the maximum.
    Any p above about 98 collapsed to max(), and a length where int(n*p) == n
    raised IndexError outright.
    """
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 50) == 50
    assert percentile(values, 99) == 99
    assert percentile(values, 100) == 100
    assert percentile(values, 1) == 1


def test_percentile_of_an_empty_list_is_zero_not_a_crash():
    assert percentile([], 95) == 0.0


def test_p99_is_not_silently_the_maximum():
    values = [1.0] * 99 + [1000.0]
    assert percentile(values, 99) == 1.0
    assert percentile(values, 100) == 1000.0


# ------------------------------------------------------------- load factor

def test_load_factor_flags_a_pool_with_no_contention():
    """A benchmark that cannot create contention measures nothing - every
    policy places everything immediately and all nine tie. The original was
    run at 8 GPUs, where exactly that happened."""
    trace = generate_jobs(count=60, seed=42)
    assert load_factor(trace, gpu_count=4, memory_per_gpu_mb=16000, span_s=60.0) > 1.0
    assert load_factor(trace, gpu_count=64, memory_per_gpu_mb=16000, span_s=60.0) < 0.5


def test_with_no_contention_every_policy_ties():
    """The property that makes the load-factor check worth running."""
    makespans = {run(o, f, jobs=6, gpus=32).makespan for o, f in MATRIX}
    assert len(makespans) == 1
