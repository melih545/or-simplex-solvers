"""Primal simplex implementation."""

from __future__ import annotations

import numpy as np

from ._result import SolverResult
from ._utils import (
    EPSILON,
    as_float_matrix,
    as_float_vector,
    build_tableau,
    compute_objective_value,
    extract_basic_solution,
    find_unit_basis,
    normalize_sense,
    pivot,
    validate_dimensions,
)


def simplex(
    c: list[float] | np.ndarray,
    A: list[list[float]] | np.ndarray,
    b: list[float] | np.ndarray,
    sense: str = "max",
) -> SolverResult:
    """Solve a tableau-ready LP with the primal simplex method."""

    result, _ = _solve_simplex(c, A, b, sense=sense, record_trajectory=False)
    return result


def _solve_simplex(
    c: list[float] | np.ndarray,
    A: list[list[float]] | np.ndarray,
    b: list[float] | np.ndarray,
    *,
    sense: str,
    record_trajectory: bool,
) -> tuple[SolverResult, list[np.ndarray]]:
    c_array = as_float_vector(c, "c")
    A_array = as_float_matrix(A, "A")
    b_array = as_float_vector(b, "b")
    validate_dimensions(c_array, A_array, b_array)
    sense = normalize_sense(sense)

    tableau = build_tableau(c_array, A_array, b_array)
    basis = find_unit_basis(A_array, np.flatnonzero(np.abs(c_array) <= EPSILON))
    trajectory = [extract_basic_solution(tableau, basis)[:2]] if record_trajectory else []
    iterations = 0

    while _can_be_improved(tableau, sense):
        pivot_row, pivot_column = _get_pivot_position(tableau, sense)
        tableau = pivot(tableau, pivot_row, pivot_column)
        basis[pivot_row - 1] = pivot_column
        iterations += 1

        if record_trajectory:
            trajectory.append(extract_basic_solution(tableau, basis)[:2])

    solution = extract_basic_solution(tableau, basis)
    result = SolverResult(
        solution=solution,
        objective_value=compute_objective_value(c_array, solution, sense=sense),
        iterations=iterations,
    )
    return result, trajectory


def _can_be_improved(tableau: np.ndarray, sense: str) -> bool:
    objective_row = tableau[0, :-1]
    if sense == "max":
        return bool(np.any(objective_row < -EPSILON))
    return bool(np.any(objective_row > EPSILON))


def _get_pivot_position(tableau: np.ndarray, sense: str) -> tuple[int, int]:
    objective_row = tableau[0, :-1]
    if sense == "max":
        pivot_column = int(np.argmin(objective_row))
    else:
        pivot_column = int(np.argmax(objective_row))

    return _get_leaving_row(tableau, pivot_column), pivot_column


def _get_leaving_row(
    tableau: np.ndarray, pivot_column: int, *, basis: np.ndarray | None = None,
    pivot_tolerance: float = EPSILON,
) -> int:
    column = tableau[1:, pivot_column]
    rhs = tableau[1:, -1]
    ratios = np.full_like(rhs, np.inf)
    np.divide(rhs, column, out=ratios, where=column > pivot_tolerance)
    if np.isinf(ratios).all():
        raise ValueError("The linear program is unbounded for the current tableau.")

    row = int(np.argmin(ratios))
    if basis is not None:
        tied_rows = np.flatnonzero(ratios == ratios[row])
        row = int(tied_rows[np.argmin(basis[tied_rows])])
    return row + 1

