"""
Runs a scheduling simulation end to end using the mock GPU monitor —
no real hardware needed. This is what proves the scheduling LOGIC
works before you touch a real GPU instance.

Runs the same job trace through both BestFitScheduler and
FIFOScheduler so you get a direct before/after comparison.

Usage: python simulate.py
"""
from scheduler.gpu_monitor import MockGPUMonitor
from scheduler.scheduler import BestFitScheduler, FIFOScheduler
from scheduler.job_generator import generate_jobs
from scheduler.benchmark_logger import BenchmarkLogger


def run_simulation(scheduler_cls, label: str, gpu_count=2, memory_per_gpu_mb=16000,
                    job_count=60, sim_duration_s=120.0, tick_s=1.0):
    monitor = MockGPUMonitor(gpu_count=gpu_count, memory_per_gpu_mb=memory_per_gpu_mb)
    scheduler = scheduler_cls(monitor)
    logger = BenchmarkLogger(f"logs/{label}.jsonl")

    jobs = generate_jobs(count=job_count)
    for job in jobs:
        scheduler.submit(job)

    now = 0.0
    while now <= sim_duration_s:
        scheduler.tick(now)
        logger.log_snapshot(now, monitor)
        now += tick_s

    # drain: let remaining running jobs finish even past sim_duration_s
    while scheduler.running:
        now += tick_s
        scheduler.tick(now)
        logger.log_snapshot(now, monitor)

    for job in scheduler.completed:
        logger.log_job(job)
    logger.close()

    completed = scheduler.completed
    wait_times = [j.wait_time for j in completed if j.wait_time is not None]
    print(f"\n=== {label} ===")
    print(f"jobs completed: {len(completed)} / {job_count}")
    if wait_times:
        sorted_waits = sorted(wait_times)
        p50 = sorted_waits[len(sorted_waits) // 2]
        p99 = sorted_waits[int(len(sorted_waits) * 0.99)]
        print(f"wait time  p50: {p50:.1f}s   p99: {p99:.1f}s   max: {max(wait_times):.1f}s")
    return completed


if __name__ == "__main__":
    run_simulation(FIFOScheduler, "fifo_baseline")
    run_simulation(BestFitScheduler, "best_fit")
    print("\nLogs written to logs/fifo_baseline.jsonl and logs/best_fit.jsonl")
