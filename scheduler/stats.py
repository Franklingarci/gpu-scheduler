"""
Summary statistics for the sweep. Stdlib only.

The important idea here is pairing. Every policy in a sweep runs the SAME
generated trace for a given seed, so the runs are matched, not independent.
Comparing mean(A) against mean(B) and eyeing whether the error bars overlap
throws that away: most of the variance between runs comes from the trace
being easy or hard, and that component is identical for both policies on the
same seed.

Take the per-seed difference first and the trace difficulty cancels out. The
confidence interval then describes the difference itself, which is the thing
the question is actually about - "is backfill faster than priority" - rather
than two separate estimates you have to compare by eye.
"""
import math
import statistics
from dataclasses import dataclass

# Two-sided 95% critical values of Student's t. A table rather than scipy,
# because the project deliberately has no third-party runtime dependency.
# Beyond df=30 the t distribution is within ~2% of the normal, so anything
# larger falls through to 1.96.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_critical_95(df: int) -> float:
    if df < 1:
        return float("inf")
    return _T95.get(df, 1.960)


@dataclass(frozen=True)
class Estimate:
    """A mean with a 95% confidence interval around it."""
    mean: float
    half_width: float
    n: int

    @property
    def low(self) -> float:
        return self.mean - self.half_width

    @property
    def high(self) -> float:
        return self.mean + self.half_width

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely on one side of zero.

        For a paired difference this is the readable form of "the effect is
        distinguishable from noise at this sample size". It is not proof the
        effect matters - an interval of [-0.9, -0.1] seconds excludes zero and
        is still worth nothing operationally.
        """
        return self.low > 0 or self.high < 0

    def __str__(self) -> str:
        return f"{self.mean:+.1f} [{self.low:+.1f}, {self.high:+.1f}]"


def estimate(values: list[float]) -> Estimate:
    """Mean and 95% CI of the mean, via the t distribution."""
    n = len(values)
    if n == 0:
        return Estimate(0.0, float("inf"), 0)
    if n == 1:
        return Estimate(values[0], float("inf"), 1)

    mean = statistics.fmean(values)
    stderr = statistics.stdev(values) / math.sqrt(n)
    return Estimate(mean, t_critical_95(n - 1) * stderr, n)


def paired_difference(treatment: list[float], baseline: list[float]) -> Estimate:
    """CI on the per-seed difference (treatment - baseline).

    Requires the two lists to be aligned by seed. Differencing first is what
    removes trace difficulty as a source of variance; do it the other way and
    a real effect can hide inside the spread of how hard the traces were.
    """
    if len(treatment) != len(baseline):
        raise ValueError(
            f"unpaired inputs: {len(treatment)} vs {len(baseline)} observations"
        )
    return estimate([t - b for t, b in zip(treatment, baseline)])


def win_rate(treatment: list[float], baseline: list[float], lower_is_better: bool = True) -> tuple[int, int]:
    """How many seeds the treatment won, as (wins, total).

    A distribution-free companion to the CI. If a policy wins on 29 of 30
    seeds the effect is real regardless of what the residuals look like, and
    if it wins on 16 of 30 while the CI excludes zero, one big outlier is
    doing the work and the mean is the wrong summary.
    """
    if len(treatment) != len(baseline):
        raise ValueError("unpaired inputs")
    wins = sum(
        (t < b) if lower_is_better else (t > b)
        for t, b in zip(treatment, baseline)
    )
    return wins, len(treatment)
