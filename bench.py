"""
Multi-seed sweep across the ordering x fit matrix.

    python bench.py                      # 30 seeds, 4 GPUs
    python bench.py --seeds 100 --gpus 2
    python bench.py --aging 0.05

Every policy sees the SAME trace for a given seed, so comparisons are paired
and the confidence intervals are on the per-seed difference rather than on
two independent means. See scheduler/stats.py for why that matters.

Reported metrics are makespan and wait-time percentiles. Mean memory
utilization is deliberately absent: a trace contains a fixed
sum(memory_mb * duration_s), so utilization over any window containing every
completion is work/(capacity*window), which has no policy term in it. It is
an identity, not a measurement, and test_simulation.py pins that fact.
"""
import argparse
import statistics
from dataclasses import dataclass, field

from scheduler import policy as policies
from scheduler.job_generator import generate_jobs, load_factor
from scheduler.simulation import simulate
from scheduler.stats import estimate, paired_difference, win_rate

BASELINE = "fifo+first_fit"


@dataclass
class Column:
    """One policy's results across every seed, aligned by seed order."""
    makespan: list[float] = field(default_factory=list)
    wait_p50: list[float] = field(default_factory=list)
    wait_p95: list[float] = field(default_factory=list)


def run_sweep(args) -> tuple[dict[str, Column], float]:
    columns: dict[str, Column] = {}
    load_factors = []

    for seed in range(args.seed_start, args.seed_start + args.seeds):
        trace = generate_jobs(
            count=args.jobs, seed=seed, arrival_span_s=args.arrival_span
        )
        load_factors.append(
            load_factor(trace, args.gpus, args.gpu_mb, args.arrival_span)
        )

        for ordering in policies.ORDERINGS:
            for fit in policies.FITS:
                pol = policies.build(ordering, fit, aging_rate=args.aging)
                # a fresh trace per policy: Job objects carry mutable run state
                # (gpu_id, start_time, status), so reusing one would leak the
                # previous policy's placements into the next run
                jobs = generate_jobs(
                    count=args.jobs, seed=seed, arrival_span_s=args.arrival_span
                )
                result = simulate(
                    pol, jobs,
                    gpu_count=args.gpus,
                    memory_per_gpu_mb=args.gpu_mb,
                    seed=seed,
                )
                col = columns.setdefault(pol.name, Column())
                col.makespan.append(result.makespan)
                col.wait_p50.append(result.wait_percentile(50))
                col.wait_p95.append(result.wait_percentile(95))

    return columns, statistics.fmean(load_factors)


def report_metric(columns: dict[str, Column], attr: str, label: str) -> None:
    baseline = getattr(columns[BASELINE], attr)
    baseline_mean = statistics.fmean(baseline)

    print(f"\n{label} — lower is better, paired against {BASELINE}")
    print(f"{'policy':24s} {'mean':>9s}  {'vs baseline':>28s}  {'wins':>7s}")
    print("-" * 76)

    ordered = sorted(columns, key=lambda name: statistics.fmean(getattr(columns[name], attr)))
    for name in ordered:
        values = getattr(columns[name], attr)
        mean = estimate(values)
        if name == BASELINE:
            print(f"{name:24s} {mean.mean:8.1f}s  {'— baseline —':>28s}  {'':>7s}")
            continue
        diff = paired_difference(values, baseline)
        wins, total = win_rate(values, baseline)
        # relative to the BASELINE, not to this policy's own mean - dividing by
        # the treatment mean reported a 31s saving off a 37s baseline as -491%
        pct = 100.0 * diff.mean / baseline_mean if baseline_mean else 0.0
        flag = "" if diff.excludes_zero else "  (CI spans 0)"
        print(
            f"{name:24s} {mean.mean:8.1f}s  "
            f"{diff.mean:+7.1f}s ({pct:+5.1f}%) [{diff.low:+6.1f},{diff.high:+6.1f}]  "
            f"{wins:3d}/{total:<3d}{flag}"
        )


def report_axes(columns: dict[str, Column]) -> None:
    """Which knob actually drives the result.

    This is the payoff of splitting ordering from fit. The old design fused
    them, so a difference between its two strategies could not be attributed
    to either one.
    """
    print("\naxis effects on makespan — each level averaged over the other axis")
    print("-" * 76)

    spreads = {}
    for axis, levels in (("ordering", policies.ORDERINGS), ("fit", policies.FITS)):
        means = {}
        for level in levels:
            matching = [
                statistics.fmean(col.makespan)
                for name, col in columns.items()
                if (name.split("+")[0] if axis == "ordering" else name.split("+")[1]) == level
            ]
            means[level] = statistics.fmean(matching)
        spreads[axis] = max(means.values()) - min(means.values())
        rendered = "   ".join(f"{k} {v:.1f}s" for k, v in sorted(means.items(), key=lambda kv: kv[1]))
        print(f"{axis:9s} {rendered}")

    ratio = spreads["ordering"] / spreads["fit"] if spreads["fit"] else float("inf")
    print(
        f"\nspread: ordering {spreads['ordering']:.1f}s   fit {spreads['fit']:.1f}s"
        f"   → ordering matters {ratio:.0f}x more than packing"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=60)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-mb", type=int, default=16000)
    ap.add_argument("--arrival-span", type=float, default=60.0)
    ap.add_argument("--aging", type=float, default=0.0)
    args = ap.parse_args()

    columns, mean_lf = run_sweep(args)

    print(f"sweep: {args.seeds} seeds x 9 policies x {args.jobs} jobs "
          f"on {args.gpus}x{args.gpu_mb}MB, aging={args.aging}")
    print(f"mean load factor {mean_lf:.2f}", end="")
    if mean_lf < 0.8:
        print("  ** BELOW 0.8: no contention, every policy will tie **")
    else:
        print("  (>1 means demand exceeds the pool over the arrival window)")

    report_metric(columns, "makespan", "MAKESPAN")
    report_metric(columns, "wait_p95", "WAIT p95")
    report_metric(columns, "wait_p50", "WAIT p50")
    report_axes(columns)

    print("\nintervals are 95% CI on the per-seed paired difference")
    print("wins = seeds where the policy beat the baseline outright")


if __name__ == "__main__":
    main()
