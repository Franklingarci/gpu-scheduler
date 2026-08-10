"""Shared builders for tests.

Policy tests construct Slots directly rather than driving a PlacementTracker.
That is the payoff of keeping policies pure: a placement test needs no clock,
no monitor and no scheduler, so it states exactly the situation it cares about
and nothing else.
"""
import pytest

from scheduler.models import Job
from scheduler.pool import Slot


@pytest.fixture
def slot():
    def _slot(gpu_id: int, free_mb: int, total_mb: int = 16000) -> Slot:
        return Slot(
            id=gpu_id,
            total_memory_mb=total_mb,
            free_memory_mb=free_mb,
            reserved_memory_mb=total_mb - free_mb,
            observed_used_mb=total_mb - free_mb,
            compute_util_pct=0.0,
        )
    return _slot


@pytest.fixture
def job():
    def _job(
        job_id: int,
        memory_mb: int,
        *,
        duration_s: float = 10.0,
        priority: int = 0,
        arrival_time: float = 0.0,
    ) -> Job:
        return Job(
            id=job_id,
            memory_mb=memory_mb,
            duration_s=duration_s,
            priority=priority,
            arrival_time=arrival_time,
        )
    return _job
