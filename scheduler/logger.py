"""
Benchmark logger — writes structured, timestamped records so you can
reconstruct utilization over time and latency stats after a run.

Two record types, one file, JSON Lines format (one JSON object per line):
  - "snapshot": periodic GPU pool state (utilization time series)
  - "job": a completed job's full lifecycle (queue wait, run time)
"""
import json
from pathlib import Path
from .models import Job
from .gpu_monitor import GPUMonitor


class BenchmarkLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")

    def log_snapshot(self, now: float, monitor: GPUMonitor) -> None:
        for device in monitor.get_devices():
            self._write({
                "type": "snapshot",
                "time": now,
                "gpu_id": device.id,
                "utilization_pct": device.utilization_pct,
                "free_memory_mb": device.free_memory_mb,
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
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._fh.close()
