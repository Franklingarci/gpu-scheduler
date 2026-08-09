# GPU-Aware Cluster Scheduler

A lightweight job scheduler for a shared GPU pool. Makes placement decisions
using real hardware state (via NVML) rather than static assumptions, and
benchmarks a best-fit packing strategy against a naive FIFO baseline.

## Why

GPU pools sit underutilized under naive scheduling — jobs queue unnecessarily
while hardware idles, or one large job starves everything else. This project
builds a smarter placement layer and measures the utilization gain against a
FIFO baseline, using real GPU memory/utilization data via NVML rather than
synthetic numbers.

## Architecture

```
Job queue -> Scheduler core -> GPU pool
                 ^                |
                 |                v
           NVML monitor <---------+
                 |
           Benchmark logger
```

- **Job queue** — synthetic or real workloads (`scheduler/job_generator.py`)
- **Scheduler core** — best-fit bin-packing placement, priority-aware (`scheduler/scheduler.py`)
- **GPU pool** — abstracted behind `GPUMonitor`; swap `MockGPUMonitor` (local dev,
  no hardware needed) for `NVMLGPUMonitor` (real hardware) with no other code changes
- **Benchmark logger** — JSONL logs of every scheduling decision and periodic
  GPU snapshots (`scheduler/benchmark_logger.py`)

## Results (local simulation, mock GPU pool)

| Strategy | Mean utilization | Median utilization |
|---|---|---|
| FIFO (baseline) | 76.6% | 80.1% |
| Best-fit | 80.0% | 85.3% |

Best-fit improves mean utilization by ~3.4 percentage points through smarter
packing alone — no additional hardware. Wait-time was roughly a wash between
strategies (best-fit optimizes for packing density, not queue latency — a
real, worth-noting tradeoff, not something to paper over).

## Running it

```bash
pip install -r requirements.txt
python simulate.py    # runs both strategies, writes logs/*.jsonl
python analyze.py     # prints utilization comparison
```

No GPU required for the above — it runs against `MockGPUMonitor`. To validate
against real hardware, swap in `NVMLGPUMonitor` from `scheduler/gpu_monitor.py`
(requires `pynvml` and an actual NVIDIA GPU).

## Roadmap / stretch goal

- LLM-based placement advisor for ambiguous scheduling decisions, benchmarked
  against the deterministic best-fit algorithm on the same test cases.

## Status

Early — scheduling logic validated in simulation; real-hardware validation
and the LLM advisor stretch goal are in progress.
