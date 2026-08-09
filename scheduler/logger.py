"""
Run logger - writes structured, timestamped records so utilization over time
and latency stats can be reconstructed after a run.

Two record types, one file, JSON Lines (one JSON object per line):
  - "snapshot": pool state at an instant (the utilization time series)
  - "job":      a completed job's full lifecycle (queue wait, run time)

Snapshots record the RECONCILED view (pool.Slot), not the raw hardware
reading, because that is what the scheduler actually had available to hand
out. Logging raw hardware would overstate capacity by whatever headroom and
foreign usage the reconciliation held back.

Records carry the run's identity - policy name and seed - on every line, so
files from a multi-seed sweep can be concatenated and still be separable.
"""
import json
from contextlib import contextmanager
from pathlib import Path

from .models import Job
from .pool import Slot


class RunLogger:
    def __init__(self, path: str, policy: str = "", seed: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.seed = seed
        self._fh = open(self.path, "w")

    def log_snapshot(self, now: float, slots: list[Slot]) -> None:
        for slot in slots:
            self._write({
                "type": "snapshot",
                "time": now,
                "gpu_id": slot.id,
                "utilization_pct": slot.utilization_pct,
                "free_memory_mb": slot.free_memory_mb,
                "reserved_memory_mb": slot.reserved_memory_mb,
                "observed_used_mb": slot.observed_used_mb,
            })

    def log_job(self, job: Job) -> None:
        self._write({
            "type": "job",
            "job_id": job.id,
            "gpu_id": job.gpu_id,
            "priority": job.priority,
            "memory_mb": job.memory_mb,
            "arrival_time": job.arrival_time,
            "start_time": job.start_time,
            "completion_time": job.completion_time,
            "wait_time": job.wait_time,
            "run_time": job.run_time,
        })

    def _write(self, record: dict) -> None:
        record["policy"] = self.policy
        record["seed"] = self.seed
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@contextmanager
def null_logger():
    """A logger that discards everything.

    The multi-seed sweep runs hundreds of simulations and only needs the
    summary numbers; writing a JSONL file per run would produce tens of
    thousands of files nobody reads.
    """
    yield _NullLogger()


class _NullLogger:
    policy = ""
    seed = None

    def log_snapshot(self, now: float, slots: list[Slot]) -> None: ...
    def log_job(self, job: Job) -> None: ...
    def close(self) -> None: ...
