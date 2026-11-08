"""Scaling and feasibility checks for the two-phase solver."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


# Scaling puts nonzero row and objective norms near one. Use a tighter
# roundoff threshold for pivot decisions than for checking primal feasibility.
ROUNDOFF_TOLERANCE = 32 * np.finfo(float).eps
FEASIBILITY_TOLERANCE = 1e-9


def scale_rows(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize rows by powers of two without changing their feasible set."""
    magnitudes = np.max(np.abs(A), axis=1, initial=0)
    # For constant rows, normalize by the RHS so even 0 = tiny is infeasible.
    magnitudes = np.where(magnitudes > 0, magnitudes, np.abs(b))
    exponents = np.where(magnitudes > 0, 1 - np.frexp(magnitudes)[1], 0)
    return _rescale(A, exponents[:, None]), _rescale(b, exponents)


def scale_costs(costs: np.ndarray) -> np.ndarray:
    magnitude = np.max(np.abs(costs), initial=0)
    return _rescale(costs, 1 - np.frexp(magnitude)[1]) if magnitude else costs


def _rescale(values: np.ndarray, exponents: np.ndarray | int) -> np.ndarray:
    scaled = np.ldexp(values, exponents)
    if np.any((values != 0) & (scaled == 0)):
        raise RuntimeError("Numerical failure: scaling erased a nonzero coefficient.")
    return scaled


def verify_solution(
    A: np.ndarray, b: np.ndarray, relations: Sequence[str],
    bounds: Sequence[tuple[float, float]], solution: np.ndarray,
) -> None:
    """Check primal feasibility in the original coordinates and row senses."""
    if not np.isfinite(solution).all():
        raise RuntimeError("Numerical failure: the solution is not finite.")

    # Normalize independently of the tableau to avoid overflow and make the
    # residual test insensitive to a change in a constraint's units.
    A, b = scale_rows(A, b)
    residuals = A @ solution - b
    activity = np.abs(A) @ np.abs(solution)
    tolerances = FEASIBILITY_TOLERANCE * np.maximum(np.abs(b), activity)
    for row, (residual, relation, tolerance) in enumerate(zip(residuals, relations, tolerances), start=1):
        violation = residual if relation == "<=" else -residual if relation == ">=" else abs(residual)
        if violation > tolerance:
            raise RuntimeError(f"Numerical failure: solution violates original constraint {row}.")

    for index, (value, (lower, upper)) in enumerate(zip(solution, bounds), start=1):
        below_lower = np.isfinite(lower) and lower - value > _bound_tolerance(value, lower)
        above_upper = np.isfinite(upper) and value - upper > _bound_tolerance(value, upper)
        if below_lower or above_upper:
            raise RuntimeError(f"Numerical failure: solution violates original bounds for variable {index}.")


def _bound_tolerance(value: float, bound: float) -> float:
    # At zero, allow absolute roundoff. Nonzero bounds use their own scale.
    return FEASIBILITY_TOLERANCE * max(abs(value), abs(bound) if bound else 1.0)
