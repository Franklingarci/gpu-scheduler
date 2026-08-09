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
    """Produce a list of jobs with randomized but reproducible (seeded)
    memory needs, durations, arrival times, and priorities — mimicking
    a heterogeneous multi-tenant workload mix."""
    rng = random.Random(seed)
    jobs = []
    for _ in range(count):
        jobs.append(Job(
            memory_mb=rng.randint(*memory_range_mb),
            duration_s=rng.uniform(*duration_range_s),
            priority=rng.randint(*priority_range),
            arrival_time=rng.uniform(0, arrival_span_s),
        ))
    jobs.sort(key=lambda j: j.arrival_time)
    return jobs
