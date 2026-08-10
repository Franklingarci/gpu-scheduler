"""
Run real jobs on a real GPU pool.

    python run.py --jobs jobs.json
    python run.py --jobs jobs.json --policy priority+best_fit
    python run.py --jobs jobs.json --dry-run --gpus 4

jobs.json is a list of objects:

    [
      {"name": "train-a", "memory_mb": 8000, "priority": 2,
       "command": ["python", "train.py", "--lr", "3e-4"]},
      {"name": "eval-b",  "memory_mb": 2000,
       "command": ["python", "eval.py"]}
    ]

--dry-run swaps NVML for a simulated pool while still launching real
processes. The scheduling, the pinning and the process lifecycle are all
exercised; only the GPU readings are fake. Rehearse a job file this way
before spending money on hardware.
"""
import argparse
import json
import sys
from pathlib import Path

from scheduler import policy as policies
from scheduler.gpu import MockGPUMonitor, NVMLGPUMonitor
from scheduler.live import LaunchError, LiveRunner
from scheduler.logger import RunLogger
from scheduler.models import Job


def load_jobs(path: str) -> list[Job]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: expected a JSON list of jobs")

    jobs = []
    for i, spec in enumerate(raw, start=1):
        missing = {"memory_mb", "command"} - set(spec)
        if missing:
            raise SystemExit(f"{path}: job {i} is missing {', '.join(sorted(missing))}")
        jobs.append(Job(
            id=i,
            name=spec.get("name", ""),
            memory_mb=int(spec["memory_mb"]),
            command=list(spec["command"]),
            priority=int(spec.get("priority", 0)),
            arrival_time=float(spec.get("arrival_time", 0.0)),
            duration_s=float(spec.get("duration_s", 0.0)),  # unused live
        ))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="path to a JSON job file")
    ap.add_argument("--policy", default="backfill+best_fit",
                    help="ordering+fit, e.g. priority+worst_fit")
    ap.add_argument("--aging", type=float, default=0.01,
                    help="priority gained per second waited; 0 disables and "
                         "allows indefinite starvation")
    ap.add_argument("--headroom-mb", type=int, default=512,
                    help="memory withheld per GPU for CUDA context overhead")
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--log")
    ap.add_argument("--dry-run", action="store_true",
                    help="simulated GPUs, real processes")
    ap.add_argument("--gpus", type=int, default=2, help="dry-run only")
    ap.add_argument("--gpu-mb", type=int, default=16000, help="dry-run only")
    args = ap.parse_args()

    try:
        ordering, fit = args.policy.split("+", 1)
        pol = policies.build(ordering, fit, aging_rate=args.aging)
    except (ValueError, KeyError):
        raise SystemExit(
            f"unknown policy {args.policy!r}; expected one of: "
            + ", ".join(f"{o}+{f}" for o in policies.ORDERINGS for f in policies.FITS)
        )

    if args.dry_run:
        monitor = MockGPUMonitor(gpu_count=args.gpus, memory_per_gpu_mb=args.gpu_mb)
        print(f"DRY RUN: {args.gpus} simulated GPUs, real processes")
    else:
        try:
            monitor = NVMLGPUMonitor()
        except Exception as exc:
            raise SystemExit(
                f"could not initialise NVML ({exc.__class__.__name__}: {exc}).\n"
                "Needs an NVIDIA GPU and a loaded driver. To rehearse without "
                "one, add --dry-run."
            )

    devices = monitor.snapshot()
    if not devices:
        raise SystemExit("no GPUs visible; check CUDA_VISIBLE_DEVICES")

    jobs = load_jobs(args.jobs)
    print(f"policy {pol.name} · {len(jobs)} jobs · {len(devices)} GPUs")
    for d in devices:
        print(f"  gpu{d.id}  {d.free_memory_mb:>6d}MB free of {d.total_memory_mb}MB")
    print()

    def on_event(kind, job, detail):
        if kind == "started":
            print(f"[start ] {job.label:20s} gpu{detail}  {job.memory_mb}MB")
        elif kind == "finished":
            mark = "ok" if detail == 0 else f"EXIT {detail}"
            print(f"[done  ] {job.label:20s} gpu{job.gpu_id}  {job.run_time:6.1f}s  {mark}")
        elif kind == "launch_failed":
            print(f"[FAILED] {job.label:20s} could not start: {detail}")

    log = RunLogger(args.log, policy=pol.name) if args.log else None
    runner = LiveRunner(
        pol, monitor,
        headroom_mb=args.headroom_mb,
        poll_interval=args.poll,
        logger=log,
        on_event=on_event,
    )

    try:
        result = runner.run(jobs)
    except LaunchError as exc:
        raise SystemExit(str(exc))
    except KeyboardInterrupt:
        print("\ninterrupted; running jobs terminated", file=sys.stderr)
        return 130
    finally:
        if log:
            log.close()

    print(f"\n{len(result.completed)}/{len(jobs)} finished in {result.wall_time_s:.1f}s")
    if result.failed:
        print(f"{len(result.failed)} exited non-zero: "
              + ", ".join(j.label for j in result.failed))
    for job, err in result.launch_failures:
        print(f"never started: {job.label} ({err})")

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
