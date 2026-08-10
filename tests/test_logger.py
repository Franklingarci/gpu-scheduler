"""The JSONL run logger."""
import json

import pytest

from scheduler import policy as P
from scheduler.gpu import MockGPUMonitor
from scheduler.job_generator import generate_jobs
from scheduler.logger import RunLogger, _NullLogger
from scheduler.pool import PlacementTracker
from scheduler.simulation import simulate


def read(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def test_snapshots_record_the_reconciled_view_not_the_raw_hardware(tmp_path, job):
    """Logging what NVML reports would overstate capacity by exactly the
    headroom and foreign usage the reconciliation is holding back, so a plot
    built from these logs would disagree with what the scheduler could
    actually hand out."""
    monitor = MockGPUMonitor(gpu_count=1, memory_per_gpu_mb=16000, external_used_mb={0: 4000})
    tracker = PlacementTracker(monitor, headroom_mb=512)
    tracker.reserve(job(1, 2000), 0)

    path = tmp_path / "run.jsonl"
    with RunLogger(str(path), policy="test", seed=1) as log:
        log.log_snapshot(3.5, tracker.slots())

    (record,) = read(path)
    assert record["observed_used_mb"] == 4000          # hardware
    assert record["reserved_memory_mb"] == 2000        # our books
    assert record["free_memory_mb"] == 16000 - 4000 - 512


def test_every_record_carries_the_run_identity(tmp_path):
    """Sweep output gets concatenated; without policy and seed on each line
    the rows are unattributable once the files are merged."""
    path = tmp_path / "run.jsonl"
    trace = generate_jobs(6, seed=99)
    with RunLogger(str(path), policy="priority+best_fit", seed=99) as log:
        simulate(P.build("priority", "best_fit"), trace, gpu_count=2, logger=log)

    records = read(path)
    assert records
    assert all(r["policy"] == "priority+best_fit" and r["seed"] == 99 for r in records)


def test_both_record_types_are_written(tmp_path):
    path = tmp_path / "run.jsonl"
    trace = generate_jobs(8, seed=3)
    with RunLogger(str(path), policy="p", seed=3) as log:
        simulate(P.build("fifo", "first_fit"), trace, gpu_count=2, logger=log)

    kinds = {r["type"] for r in read(path)}
    assert kinds == {"snapshot", "job"}


def test_one_job_record_per_completed_job(tmp_path):
    path = tmp_path / "run.jsonl"
    trace = generate_jobs(10, seed=4)
    with RunLogger(str(path), policy="p", seed=4) as log:
        result = simulate(P.build("priority", "best_fit"), trace, gpu_count=2, logger=log)

    jobs = [r for r in read(path) if r["type"] == "job"]
    assert len(jobs) == result.jobs_completed == 10
    assert {j["job_id"] for j in jobs} == set(range(1, 11))


def test_job_records_carry_the_full_lifecycle(tmp_path):
    path = tmp_path / "run.jsonl"
    with RunLogger(str(path), policy="p", seed=1) as log:
        simulate(P.build("fifo", "first_fit"), generate_jobs(4, seed=1), gpu_count=2, logger=log)

    for rec in (r for r in read(path) if r["type"] == "job"):
        assert rec["start_time"] >= rec["arrival_time"]
        assert rec["completion_time"] > rec["start_time"]
        assert rec["wait_time"] == pytest.approx(rec["start_time"] - rec["arrival_time"])
        assert rec["run_time"] == pytest.approx(rec["completion_time"] - rec["start_time"])
        assert rec["gpu_id"] is not None


def test_the_parent_directory_is_created(tmp_path):
    path = tmp_path / "deep" / "nested" / "run.jsonl"
    with RunLogger(str(path), policy="p") as log:
        log.log_snapshot(0.0, [])
    assert path.exists()


def test_utilization_series_is_reconstructable_from_the_log(tmp_path):
    """The point of keeping the logger: a plot of pool utilization over time,
    which is the one visual the project lacks."""
    path = tmp_path / "run.jsonl"
    trace = generate_jobs(20, seed=7)
    with RunLogger(str(path), policy="p", seed=7) as log:
        simulate(P.build("priority", "best_fit"), trace, gpu_count=2, logger=log)

    snaps = [r for r in read(path) if r["type"] == "snapshot"]
    times = sorted({r["time"] for r in snaps})

    assert times == sorted(times)
    assert len(times) > 5
    assert max(r["utilization_pct"] for r in snaps) > 0


def test_the_null_logger_swallows_everything(tmp_path):
    """The default, so the sweep does not emit thousands of files."""
    null = _NullLogger()
    null.log_snapshot(1.0, [])
    null.log_job(None)
    null.close()
    assert list(tmp_path.iterdir()) == []
