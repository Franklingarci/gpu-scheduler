# GPU-Aware Cluster Scheduler

A job scheduler for a shared GPU pool, plus a benchmark harness that measures
whether its placement strategies actually beat a naive baseline.

Nine strategies, thirty seeds, paired confidence intervals. The core has zero
third-party dependencies and needs no GPU to run.

```bash
python simulate.py                              # one trace, all nine policies
python bench.py                                 # 30-seed sweep with CIs
python run.py --jobs examples/jobs.json --dry-run   # run real processes
python -m pytest                                # 163 tests, ~1.8s
```

## Results

30 seeds, 60 jobs, 4x16GB pool, mean load factor 1.90. Every policy sees the
same trace per seed, so comparisons are paired and the intervals are on the
per-seed difference. Baseline is `fifo+first_fit`.

| policy | makespan | vs baseline | 95% CI | seeds won |
|---|---|---|---|---|
| `priority+first_fit` | 156.0s | **-4.4%** | [-10.6, -3.8] | 23/30 |
| `priority+best_fit` | 156.0s | -4.4% | [-10.3, -4.1] | 23/30 |
| `priority+worst_fit` | 157.2s | -3.7% | [-9.2, -2.9] | 23/30 |
| `backfill+first_fit` | 160.9s | -1.4% | [-5.7, **+1.1**] | 19/30 |
| `fifo+best_fit` | 161.1s | -1.3% | [-3.7, -0.4] | 16/30 |
| `fifo+first_fit` | 163.2s | baseline | | |

### The finding is a tradeoff, not a winner

The interesting result is not that one policy wins. It is that the policies
which drain the queue fastest are the ones that treat the unluckiest job
worst, and the effect is large and consistent enough to state precisely.

| policy | median wait | p95 wait |
|---|---|---|
| `priority+first_fit` | 7.0s (**-81%**, 30/30 seeds) | 88.5s (**+13%**, won 5/30) |
| `backfill+first_fit` | 10.5s (-72%, 30/30 seeds) | 92.3s (+18%, won 2/30) |
| `fifo+first_fit` | 37.5s | 77.9s |

Priority ordering cuts the median job's wait by 81% on every single seed, and
makes the 95th-percentile wait 13% worse while winning only 5 of 30. FIFO is
slow and fair: nobody is ever skipped, so the tail is bounded. Priority is
fast for the typical job and punishes the unlucky one.

Which of those you want is a policy question, not an engineering one. A
research cluster where a starved job means a blocked PhD student should
probably run FIFO. A CI fleet where throughput is everything should not.

### Ordering matters ~6x more than packing

Averaging each level over the other axis:

```
ordering   priority 156.4s    backfill 161.4s    fifo 162.7s     spread 6.3s
fit        best_fit 159.7s    first_fit 160.0s   worst_fit 160.8s  spread 1.1s
```

Which job runs next dominates which GPU it lands on. Most of the fit-strategy
differences have confidence intervals spanning zero, meaning they are not
distinguishable from noise at 30 seeds.

That is a limitation of the simulator, not a fact about GPUs, and it was
predicted before the sweep ran (see `WorstFit`'s docstring). This simulator
models memory only. The cost of packing tightly is *compute* contention: two
jobs that both fit in memory on one card still fight over the same SMs. A
memory-only model cannot see that, so it cannot separate packing strategies.
Measuring it needs real hardware.

### Backfill does not help here

`backfill`'s makespan interval spans zero. It is not distinguishable from the
baseline.

The cause is known and documented in the code: this implementation reserves a
whole GPU for the blocked head job and lets nothing else touch it. Real EASY
backfill lets a short job use the reserved card if it would finish before the
reservation comes due, which requires a per-job runtime estimate. That is
exactly why Slurm makes users declare a time limit. Without one, this version
is strictly more conservative than it needs to be and idles capacity.

## A correction to an earlier version of this README

This project previously reported that best-fit improved mean GPU utilization
by 3.4 percentage points over a FIFO baseline. **That result was an artifact
and has been withdrawn.** Two independent problems:

**The runs were averaged over different windows.** Utilization was computed
per run over that run's own length. The runs were not the same length, so a
policy that finished sooner had fewer trailing idle moments dragging its
average down. Measured over a window common to both, the gap disappeared and
slightly reversed.

**The metric cannot work, even in principle.** A trace contains a fixed
amount of work, `sum(memory_mb * duration_s)`. Integrated over any window
containing every completion, utilization is:

```
work / (pool_capacity * window)
```

There is no policy term in that expression. Every policy moves the same bytes
for the same durations and differs only in *when*. All nine strategies report
identical utilization when measured honestly. It is an identity, not a
measurement. `tests/test_simulation.py` pins this so nobody "improves" it
later.

Two smaller problems in the same measurement path: the reported p99 was
`sorted[int(n*0.99)]`, which for 60 samples is index 59, the maximum. And the
old baseline labelled FIFO used `continue` where strict FIFO uses `break`, so
it skipped past jobs that did not fit. That skip is already a crude backfill,
which is much of why the "smarter" policy could not beat it.

The benchmark now reports makespan and wait percentiles, which do vary.

## Design

Three concerns, kept in three files, because merging any two of them is what
broke the previous version.

```
gpu.py      SENSOR       what the hardware reports, read-only
pool.py     LEDGER       what we placed, reconciled against the sensor
policy.py   DECISIONS    who runs next and where, pure functions
core.py     ENGINE       holds the queue, applies the plan
```

**The sensor is read-only.** An earlier design put `allocate()` on the GPU
monitor, so the scheduler claimed memory by calling it. `NVMLGPUMonitor` did
not have that method and never could: NVML is telemetry, with no allocate
call in its API. You claim memory on a real GPU by launching a process, and
NVML then *observes* it. Asking a monitor to allocate is asking a thermometer
to heat the room. The README used to promise you could swap in real hardware
with no other code changes; that promise raised `AttributeError` on the first
placement.

**The ledger reconciles two disagreeing sources of truth.**

```python
free = min(hardware_says_free, total - we_reserved) - headroom
```

The scheduler knows what it reserved. The hardware knows what is resident.
They diverge on a real box: another user's process, a launched job that has
not allocated yet, a crashed job whose memory the driver has not reclaimed,
CUDA context overhead. Trust either alone and you place onto a card that
OOMs, so take the more pessimistic one. In simulation nothing foreign exists,
the hardware term is the whole GPU, and this collapses to plain bookkeeping.
The simulator is a strict special case of reality rather than a second
implementation that drifts.

**Policies see the whole queue.** The obvious interface is
`select_gpu(job) -> gpu_id`, and it cannot express backfill, which has to
know which job is stuck at the front and which jobs behind it could slip
past. Policies take the ready list and return a batch:

```python
def plan(self, ready: list[Job], slots: list[Slot], now: float) -> list[Placement]
```

**The core never reads a clock.** `now` arrives as an argument, and a driver
calls `complete()` when a job finishes. That is what lets one implementation
serve both a simulated event clock and a wall clock on real hardware, where a
process can exit early on a crash or late on a hang. A test enforces it by
parsing the module's imports.

### Strategies

Ordering and fit are independent axes, so they vary independently. Fusing
them is why the old comparison could not attribute a difference to either.

**Ordering** decides who is considered next:

- `fifo` strict arrival order; a job that does not fit blocks everything behind it
- `priority` highest priority first, skipping past anything that does not fit
- `backfill` priority order, but a card is held for the blocked head job so
  the jobs that jump the line cannot delay it

**Fit** decides which GPU the chosen job lands on:

- `first_fit` lowest device id that fits; cheap, but keeps gpu0 hot and the tail cold
- `best_fit` tightest that fits; packs small jobs together to keep whole cards free
- `worst_fit` emptiest that fits; spreads load instead of packing it

Any of the nine combinations builds by name:

```python
policy.build("backfill", "worst_fit")   # -> "backfill+worst_fit"
```

`priority` and `backfill` accept `aging_rate`, which adds priority per second
waited. Without it, strict priority starves: a steady arrival of high-priority
work means a low-priority job is never the best candidate and waits forever.
Aging bounds the wait rather than merely shortening it. `aging_rate=0` keeps
the starvation, which is useful for demonstrating it.

## Simulation

Discrete-event, over a `heapq` of future events: pop the earliest, jump the
clock to it. The previous simulator advanced a fixed 1-second tick, so a 5.3s
job was released at tick 6 and held memory it was not using for 0.7s. That
error was systematic and inflated every utilization figure in the same
direction.

Between two events the pool cannot change, so utilization is piecewise
constant and the time-weighted average over any window is an exact integral
rather than a sample mean over irregular timestamps.

`load_factor()` reports how hard a trace pushes the pool. Below about 0.8
there is no contention, every policy places everything immediately, and all
nine tie. A benchmark that cannot create contention measures nothing, which
is what happened when the original was run with 8 GPUs.

## Running real jobs

`run.py` drives real processes on a real pool using the same nine policies.
Jobs come from a JSON file:

```json
[
  {"name": "train-a", "memory_mb": 8000, "priority": 2,
   "command": ["python", "train.py", "--lr", "3e-4"]}
]
```

```bash
python run.py --jobs examples/jobs.json --policy priority+best_fit
```

Each job is launched with `CUDA_VISIBLE_DEVICES` set to the card the
scheduler picked, so it sees exactly one GPU, numbered 0.

**`--dry-run` swaps NVML for a simulated pool while still launching real
processes.** Scheduling, pinning, reaping, exit codes and rollback are all
exercised; only the GPU readings are fake. Rehearse a job file this way
before spending money on hardware:

```
$ python run.py --jobs examples/jobs.json --dry-run --gpus 2
[start ] train-large          gpu0  12000MB
[start ] sweep-shard-1        gpu1  6000MB
[start ] sweep-shard-2        gpu1  6000MB
[start ] eval-a               gpu0  3000MB
[done  ] eval-a               gpu0     2.0s  ok
[start ] preprocess           gpu0  2000MB
[done  ] train-large          gpu0     4.1s  ok

6/6 finished in 4.1s
```

Three things exist here that have no counterpart in simulation, each one a
way real hardware misbehaves:

- **A job finishes when its process exits**, not when an estimate elapses. It
  can crash in two seconds or hang past any prediction. The core is told; it
  never infers. Non-zero exits are recorded separately from completions,
  because a fleet where everything crashes has excellent makespan and has
  accomplished nothing.
- **A launch can fail outright** on a bad binary or a missing file. The
  reservation is already on the books by then, so it is rolled back. Without
  that, the memory is gone for the rest of the run and every later job is
  scheduled against a pool that is quietly smaller.
- **The run can stall.** If arrived work is queued, nothing is running, and
  nothing fits, `StalledError` is raised rather than polling forever. That
  guard exists because mutation testing removed the rollback above and the
  test suite *hung* instead of failing, which is the worse outcome.

Interrupting with Ctrl-C terminates running children rather than orphaning
processes that still hold GPU memory.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest              # 163 tests, ~1.8s, no GPU required
```

Policy tests need no clock, no hardware and no scheduler, which is the payoff
of keeping policies pure. `test_nvml.py` exercises the real-hardware path
against a stub driver, pinning unit conversion and device-index mapping
before any GPU time gets spent on them.

The suite is verified by mutation rather than assumed. Reintroducing the
original `continue`-instead-of-`break` fails 2 tests including the one named
for it; removing the per-pass memory budget fails 17. Mutating the live
driver's launch rollback made the suite hang rather than fail, which is how
`StalledError` came to exist.

A pre-commit hook runs the suite against the staged snapshot, not the working
tree, so staging a subset is validated as what it actually is. Enable it on a
fresh clone:

```bash
git config core.hooksPath .githooks
```

## Install

The core has no third-party dependencies. `simulate.py`, `bench.py` and the
whole scheduler run on a clean checkout with nothing installed.

```
requirements.txt       nvidia-ml-py   real-hardware monitoring only
requirements-dev.txt   pytest
```

`nvidia-ml-py` is NVIDIA's own package and provides the `pynvml` module. Do
not swap it for the `pynvml` distribution on PyPI: that one is deprecated and
just pulls this in anyway. It installs fine without a GPU, since it is a
ctypes wrapper that only fails at `nvmlInit()`.

## Status

Working: simulator, nine policies, paired benchmark harness, live driver,
JSONL run logging, test suite.

The live driver required no changes to `policy.py` or `core.py`, which was
the entire point of the sensor/ledger split and is now demonstrated rather
than claimed. A test runs all nine policies against real processes.

Not yet built:

- **Real-hardware validation.** The open question is whether packing
  strategies separate once compute contention is real. This simulator cannot
  answer it, and `--dry-run` cannot either.
- **Runtime estimates for backfill**, to test whether proper EASY backfill
  beats the conservative reservation this version uses.
