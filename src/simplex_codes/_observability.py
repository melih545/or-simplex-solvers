"""Measure HTTP solver attempts without recording model data."""

import json
import logging
from time import perf_counter
from typing import get_args

from prometheus_client import CollectorRegistry, Counter, Histogram

from ._service import SolveOutcome, Status, execute_model


REGISTRY = CollectorRegistry()
ATTEMPTS = Counter(
    "simplex_solve_attempts", "Solve attempts by solver and outcome.",
    ("solver", "status"), registry=REGISTRY,
)
DURATION = Histogram(
    "simplex_solve_duration_seconds", "Time spent executing an LP model.",
    ("solver",), buckets=(.0005, .001, .0025, .005, .01, .025, .05, .1, .25, .5, 1, 5),
    registry=REGISTRY,
)
SOLVER = "two_phase"
LOG = logging.getLogger("simplex.solve")

for status in get_args(Status):
    ATTEMPTS.labels(SOLVER, status)
DURATION.labels(SOLVER)


def measure_model(model: dict, *, trace: bool = False) -> SolveOutcome:
    start = perf_counter()
    outcome = execute_model(model, trace=trace)
    duration = perf_counter() - start
    ATTEMPTS.labels(SOLVER, outcome.status).inc()
    DURATION.labels(SOLVER).observe(duration)
    LOG.info(json.dumps({
        "event": "solve_finished",
        "solver": SOLVER,
        "status": outcome.status,
        "duration_seconds": duration,
        "iterations": outcome.iterations,
    }))
    return outcome
