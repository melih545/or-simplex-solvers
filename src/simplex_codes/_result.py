"""Shared result model for solver outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Normalized solver output."""

    solution: np.ndarray
    objective_value: float
    iterations: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "solution", np.asarray(self.solution, dtype=float))
        object.__setattr__(self, "objective_value", float(self.objective_value))
        object.__setattr__(self, "iterations", int(self.iterations))

