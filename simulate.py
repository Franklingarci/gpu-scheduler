"""
Run one job trace through every policy and print a comparison.

    python simulate.py                 # default pool, seed 42
    python simulate.py --seed 7 --gpus 2

No GPU required - this runs against MockGPUMonitor.

A note on why mean utilization is not in this table. A trace contains a fixed
amount of work: sum(memory_mb * duration_s). Integrate utilization over any
window long enough to contain every completion and you get

    work / (pool_capacity * window)

which has no policy term in it. Every policy moves the same bytes for the same
durations and differs only in WHEN. Measured fairly, mean utilization is
constant across all nine strategies - it cannot distinguish them even in
principle. The original project reported it as the headline result; the
apparent difference came entirely from averaging each policy over its own
makespan, so whoever finished first scored highest.

What does vary: how fast the queue drains (makespan), how long jobs wait
(percentiles), and how much work you front-load while jobs are still arriving
(early-window utilization).
"""
import argparse

from scheduler import policy as policies
from scheduler.job_generator import generate_jobs, load_factor
from scheduler.simulation import simulate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=60)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--gpu-mb", type=int, default=16000)
    ap.add_argument("--arrival-span", type=float, default=60.0)
    ap.add_argument("--aging", type=float, default=0.0)
    args = ap.parse_args()

    jobs = generate_jobs(
        count=args.jobs, seed=args.seed, arrival_span_s=args.arrival_span
    )
    lf = load_factor(jobs, args.gpus, args.gpu_mb, args.arrival_span)

    print(f"seed {args.seed} · {args.jobs} jobs · {args.gpus}x{args.gpu_mb}MB")
    print(f"load factor {lf:.2f}", end="  ")
    print("(under 0.8 means no contention; every policy will tie)" if lf < 0.8 else "")
    print()

    results = []
    for ordering in policies.ORDERINGS:
        for fit in policies.FITS:
            pol = policies.build(ordering, fit, aging_rate=args.aging)
            trace = generate_jobs(
                count=args.jobs, seed=args.seed, arrival_span_s=args.arrival_span
            )
            results.append(
                simulate(
                    pol,
                    trace,
                    gpu_count=args.gpus,
                    memory_per_gpu_mb=args.gpu_mb,
                    seed=args.seed,
                )
            )

    window = max(r.makespan for r in results)
    early = args.arrival_span  # while work is still arriving

    print(f"{'policy':26s} {'makespan':>9s} {'early util':>11s} "
          f"{'wait p50':>9s} {'wait p95':>9s} {'wait max':>9s}")
    print("-" * 80)
    for r in sorted(results, key=lambda r: r.makespan):
        print(
            f"{r.policy:26s} {r.makespan:8.1f}s "
            f"{r.utilization_over(0, early):10.1f}% "
            f"{r.wait_percentile(50):8.1f}s {r.wait_percentile(95):8.1f}s "
            f"{r.wait_percentile(100):8.1f}s"
        )

    flat = {round(r.utilization_over(0, window), 4) for r in results}
    print(f"\nearly util = first {early:.0f}s, while jobs are still arriving")
    print(
        f"full-window util over 0-{window:.1f}s: "
        + (f"{flat.pop()*1:.1f}% for ALL nine policies - conserved, see module docstring"
           if len(flat) == 1 else f"{sorted(flat)}")
    )


if __name__ == "__main__":
    main()
