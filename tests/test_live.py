"""The live driver, exercised with real subprocesses against a simulated pool.

Only the GPU readings are fake here. Process launching, CUDA_VISIBLE_DEVICES
pinning, reaping, exit codes and rollback are all real, which is most of what
can go wrong. That matters because the alternative is discovering these on
metered hardware.
"""
import sys

import pytest

from scheduler import policy as P
from scheduler.gpu import MockGPUMonitor
from scheduler.live import LaunchError, LiveRunner
from scheduler.models import Job

PY = sys.executable


def sleeper(seconds: float) -> list[str]:
    return [PY, "-c", f"import time; time.sleep({seconds})"]


def exiter(code: int) -> list[str]:
    return [PY, "-c", f"raise SystemExit({code})"]


def writer(path, text: str) -> list[str]:
    return [PY, "-c", f"open({str(path)!r}, 'w').write({text})"]


def runner(gpus=2, mb=16000, headroom=0, **kw):
    return LiveRunner(
        P.build("priority", "best_fit"),
        MockGPUMonitor(gpu_count=gpus, memory_per_gpu_mb=mb),
        headroom_mb=headroom,
        poll_interval=0.02,
        **kw,
    )


def make(job_id, memory_mb, command, **kw):
    return Job(id=job_id, memory_mb=memory_mb, duration_s=0.0, command=command, **kw)


# ------------------------------------------------------------ happy path

def test_every_job_runs_to_completion():
    jobs = [make(i, 4000, sleeper(0.05)) for i in range(1, 5)]
    result = runner().run(jobs)

    assert len(result.completed) == 4
    assert result.ok
    assert all(j.exit_code == 0 for j in result.completed)
    assert result.wall_time_s > 0


def test_the_job_is_pinned_to_the_gpu_the_scheduler_chose(tmp_path):
    """The one line that connects a scheduling decision to real hardware.

    scheduler gpu_id -> CUDA_VISIBLE_DEVICES -> what the process can see. An
    off-by-one here runs every job on the wrong card while the ledger insists
    otherwise, and nothing would look wrong until the pool started OOMing.
    """
    out = tmp_path / "seen.txt"
    job = make(1, 4000, writer(out, "__import__('os').environ['CUDA_VISIBLE_DEVICES']"))

    result = runner(gpus=4).run([job])
    placed_on = result.completed[0].gpu_id

    assert out.read_text() == str(placed_on)


def test_each_job_sees_only_its_own_card(tmp_path):
    outs = [tmp_path / f"{i}.txt" for i in range(1, 3)]
    jobs = [
        make(i, 12000, writer(out, "__import__('os').environ['CUDA_VISIBLE_DEVICES']"))
        for i, out in enumerate(outs, start=1)
    ]

    runner(gpus=2).run(jobs)
    seen = {out.read_text() for out in outs}

    assert seen == {"0", "1"}, "two 12GB jobs must land on different cards"


# ------------------------------------------------------- memory is respected

def test_a_job_waits_when_the_pool_is_full(tmp_path):
    """Three 12GB jobs on two 16GB cards: one has to wait for a completion."""
    jobs = [make(i, 12000, sleeper(0.05)) for i in range(1, 4)]
    result = runner(gpus=2).run(jobs)

    assert len(result.completed) == 3
    starts = sorted(j.start_time for j in result.completed)
    assert starts[2] > starts[0], "the third job cannot have started immediately"


def test_headroom_is_withheld_from_real_placements():
    """A 15.9GB job does not fit a 16GB card once CUDA context overhead is
    reserved, and must not be placed as though it does.

    Driven through the scheduler rather than run(), because a job that can
    never be placed would spin the poll loop forever - which is itself worth
    knowing about, and is why the simulator has an explicit guard for it.
    """
    run = runner(gpus=1, headroom=512)
    run.scheduler.submit(make(1, 15_900, sleeper(0.05)))

    assert run.scheduler.schedule(0.0) == []
    assert run.tracker.slots()[0].free_memory_mb == 15_488

    # 15.4GB does fit within the same headroom, so the limit is the headroom
    # and not some other accidental cap
    run.scheduler.submit(make(2, 15_400, sleeper(0.05)))
    assert len(run.scheduler.schedule(0.0)) == 1


# ------------------------------------------------------------- failure modes

def test_a_crashing_job_still_frees_its_gpu():
    """Exit code is recorded, memory comes back. A fleet where everything
    crashes has excellent makespan and has accomplished nothing, so the two
    are tracked separately."""
    jobs = [make(1, 12000, exiter(3)), make(2, 12000, sleeper(0.05))]
    result = runner(gpus=1).run(jobs)

    assert len(result.completed) == 2
    assert not result.ok
    failed = [j for j in result.completed if j.failed]
    assert [j.id for j in failed] == [1]
    assert failed[0].exit_code == 3


def test_a_command_that_cannot_start_releases_its_reservation():
    """The rollback that has no counterpart in simulation.

    The reservation is on the books before Popen runs. If the launch throws
    and nothing undoes it, that memory is gone for the rest of the run and
    every later job is scheduled against a pool that is quietly smaller.
    """
    jobs = [
        make(1, 12000, ["/definitely/not/a/binary"]),
        make(2, 12000, sleeper(0.05)),
    ]
    run = runner(gpus=1)
    result = run.run(jobs)

    assert len(result.launch_failures) == 1
    assert result.launch_failures[0][0].id == 1
    # the second job needs the full card, so it only runs if rollback worked
    assert [j.id for j in result.completed] == [2]
    assert run.tracker.reserved_mb(0) == 0


def test_a_job_with_no_command_is_rejected_before_anything_starts():
    """Fail on the whole batch rather than launching half of it and then
    discovering the rest is unrunnable."""
    jobs = [make(1, 4000, sleeper(0.05)), make(2, 4000, None)]

    with pytest.raises(LaunchError, match="no command"):
        runner().run(jobs)


def test_the_pool_is_left_clean_after_a_run():
    jobs = [make(i, 4000, sleeper(0.05)) for i in range(1, 4)]
    run = runner()
    run.run(jobs)

    assert all(s.free_memory_mb == s.total_memory_mb for s in run.tracker.slots())
    assert run._procs == {}


# ------------------------------------------------------------------ events

def test_events_are_reported_in_lifecycle_order():
    seen = []
    jobs = [make(1, 4000, sleeper(0.05))]
    runner(on_event=lambda kind, job, detail: seen.append(kind)).run(jobs)

    assert seen == ["started", "finished"]


def test_policies_are_interchangeable_on_the_live_path():
    """The claim the whole sensor/ledger split exists to support: the same
    nine policies drive real processes with no changes."""
    for ordering in P.ORDERINGS:
        for fit in P.FITS:
            run = LiveRunner(
                P.build(ordering, fit),
                MockGPUMonitor(gpu_count=2, memory_per_gpu_mb=16000),
                headroom_mb=0,
                poll_interval=0.02,
            )
            result = run.run([make(i, 8000, sleeper(0.02)) for i in range(1, 4)])
            assert len(result.completed) == 3, f"{ordering}+{fit} did not drain"


# ------------------------------------------------------------ stall guard

def test_a_job_bigger_than_any_gpu_raises_instead_of_spinning():
    """Without this the poll loop runs forever with an idle pool.

    Found by mutation testing: removing the launch-failure rollback made the
    suite hang rather than fail, which is strictly worse than a crash.
    """
    from scheduler.live import StalledError

    run = runner(gpus=1, mb=8000)
    with pytest.raises(StalledError, match="nothing fits"):
        run.run([make(1, 20000, sleeper(0.05))])


def test_a_leaked_reservation_is_caught_rather_than_hung():
    from scheduler.live import StalledError

    run = runner(gpus=1, mb=16000)
    run.tracker.reserve(make(99, 16000, sleeper(0.01)), 0)   # simulate a leak

    with pytest.raises(StalledError, match="reservation leaked|nothing fits"):
        run.run([make(1, 8000, sleeper(0.05))])


def test_waiting_for_a_future_arrival_is_not_a_stall():
    """A job scheduled to arrive later is the loop working, not a deadlock."""
    jobs = [make(1, 4000, sleeper(0.05), arrival_time=0.3)]
    result = runner().run(jobs)
    assert len(result.completed) == 1
