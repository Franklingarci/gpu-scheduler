"""Summary statistics, especially the pairing that the sweep depends on."""
import pytest

from scheduler.stats import (
    Estimate,
    estimate,
    paired_difference,
    t_critical_95,
    win_rate,
)


def test_mean_and_interval_on_a_known_sample():
    est = estimate([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert est.mean == pytest.approx(5.0)
    assert est.n == 8
    # stdev 2.138, stderr 0.756, t(7) 2.365
    assert est.half_width == pytest.approx(2.365 * 0.7559, rel=1e-3)


def test_identical_values_give_a_zero_width_interval():
    est = estimate([3.0] * 10)
    assert est.mean == 3.0
    assert est.half_width == pytest.approx(0.0)
    # [3.0, 3.0] genuinely excludes zero: every observation agreed on a
    # non-zero value, which is as certain as a sample can be
    assert est.excludes_zero


def test_a_zero_difference_on_every_seed_does_not_exclude_zero():
    """The companion case: policies that tie on every trace, which happens
    whenever the pool is big enough that nothing ever contends."""
    est = estimate([0.0] * 10)
    assert est.mean == 0.0
    assert not est.excludes_zero


def test_a_single_observation_has_no_usable_interval():
    """One sample cannot bound its own error, and reporting a tight interval
    from n=1 is worse than reporting none."""
    est = estimate([42.0])
    assert est.mean == 42.0
    assert est.half_width == float("inf")
    assert not est.excludes_zero


def test_empty_input_does_not_crash():
    assert estimate([]).n == 0


def test_smaller_samples_get_wider_intervals():
    """The t correction. Using 1.96 regardless would understate uncertainty at
    small n, which is exactly where a sweep is most likely to mislead."""
    assert t_critical_95(1) > t_critical_95(10) > t_critical_95(29) > t_critical_95(500)
    assert t_critical_95(500) == pytest.approx(1.96)


# ---------------------------------------------------------------- pairing

def test_pairing_finds_an_effect_that_unpaired_comparison_would_miss():
    """The reason the sweep runs every policy on the same trace.

    Trace difficulty swamps the policy effect: these runs range over 100s
    while the policy is worth a steady 5s. Unpaired, the effect drowns in
    that spread. Paired, it is unmistakable.
    """
    baseline = [100.0, 150.0, 200.0, 120.0, 180.0, 160.0, 140.0, 190.0]
    treatment = [b - 5.0 for b in baseline]

    unpaired_spread = estimate(baseline).half_width
    paired = paired_difference(treatment, baseline)

    assert paired.mean == pytest.approx(-5.0)
    assert paired.half_width == pytest.approx(0.0)
    assert paired.excludes_zero
    assert unpaired_spread > 20.0, "unpaired uncertainty dwarfs the real effect"


def test_pairing_requires_aligned_inputs():
    """Misaligned lists would silently compare seed 3 against seed 7."""
    with pytest.raises(ValueError, match="unpaired"):
        paired_difference([1.0, 2.0, 3.0], [1.0, 2.0])


def test_a_difference_that_is_pure_noise_spans_zero():
    baseline = [10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 7.0, 10.0]
    treatment = [12.0, 10.0, 9.0, 8.0, 13.0, 7.0, 11.0, 10.0]
    assert not paired_difference(treatment, baseline).excludes_zero


def test_excludes_zero_is_about_significance_not_importance():
    """A tiny effect measured precisely still excludes zero. The sweep prints
    the magnitude alongside so nobody reads this as 'matters'."""
    tiny = Estimate(mean=-0.2, half_width=0.05, n=30)
    assert tiny.excludes_zero


# --------------------------------------------------------------- win rate

def test_win_rate_counts_seeds_not_magnitude():
    """The distribution-free cross-check. If a policy wins on 2 of 30 seeds
    while its mean looks good, one outlier is carrying the result."""
    baseline = [10.0] * 10
    treatment = [9.0] * 9 + [100.0]

    wins, total = win_rate(treatment, baseline)
    assert (wins, total) == (9, 10)

    # the mean says the treatment is far worse; the win rate says otherwise,
    # and the disagreement is the signal
    assert paired_difference(treatment, baseline).mean > 0


def test_win_rate_can_be_inverted_for_higher_is_better_metrics():
    baseline = [1.0, 2.0, 3.0]
    treatment = [2.0, 3.0, 4.0]
    assert win_rate(treatment, baseline, lower_is_better=False) == (3, 3)
    assert win_rate(treatment, baseline, lower_is_better=True) == (0, 3)


def test_ties_are_not_counted_as_wins():
    """Nine of the policies frequently tie on easy traces; counting ties as
    wins would inflate every row of the sweep."""
    assert win_rate([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]) == (0, 3)
