"""Generates synthetic job traces to drive the scheduler simulation."""
import random

from .models import Job


def generate_jobs(
    count: int,
    seed: int = 42,
    memory_range_mb: tuple[int, int] = (2000, 12000),
    duration_range_s: tuple[float, float] = (5.0, 30.0),
    arrival_span_s: float = 60.0,
    priority_range: tuple[int, int] = (0, 3),
) -> list[Job]:
    """A heterogeneous multi-tenant workload mix, randomized but reproducible.

    Job ids are assigned 1..count per trace rather than from the global
    counter in models.py. That counter keeps incrementing for the life of the
    process, so the second trace generated in a sweep would get ids 61..120
    while the first got 1..60 - same seed, different ids. Since policies break
    ties on job id, identical seeds would then schedule differently depending
    on what ran before them. Seeding is worthless if the results still depend
    on execution order.
    """
    rng = random.Random(seed)
    jobs = [
        Job(
            memory_mb=rng.randint(*memory_range_mb),
            duration_s=rng.uniform(*duration_range_s),
            priority=rng.randint(*priority_range),
            arrival_time=rng.uniform(0, arrival_span_s),
        )
        for _ in range(count)
    ]
    jobs.sort(key=lambda j: j.arrival_time)
    for i, job in enumerate(jobs, start=1):
        job.id = i
    return jobs


def load_factor(
    jobs: list[Job], gpu_count: int, memory_per_gpu_mb: int, span_s: float
) -> float:
    """How hard this trace pushes the pool. 1.0 means exactly saturating.

    Below ~0.8 there is no contention, every policy places everything
    immediately, and all nine strategies tie - which is exactly what happened
    when the old benchmark was run with 8 GPUs. A scheduling benchmark that
    cannot create contention is measuring nothing, so the sweep should assert
    on this before trusting a result.
    """
    demand = sum(j.memory_mb * j.duration_s for j in jobs)
    supply = gpu_count * memory_per_gpu_mb * span_s
    return demand / supply if supply else 0.0
