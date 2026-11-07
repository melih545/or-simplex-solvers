"""Dual simplex implementation."""

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
    pivot,
    validate_dimensions,
)


def dual_simplex(
    c: list[float] | np.ndarray,
    A: list[list[float]] | np.ndarray,
    b: list[float] | np.ndarray,
    objective_offset: float = 0.0,
) -> SolverResult:
    """Solve a tableau-ready LP with the dual simplex method."""

    c_array = as_float_vector(c, "c")
    A_array = as_float_matrix(A, "A")
    b_array = as_float_vector(b, "b")
    validate_dimensions(c_array, A_array, b_array)

    tableau = build_tableau(c_array, A_array, b_array, objective_offset=objective_offset)
    basis = find_unit_basis(A_array, np.flatnonzero(np.abs(c_array) <= EPSILON))
    iterations = 0

    while np.any(tableau[1:, -1] < -EPSILON):
        pivot_row, pivot_column = _get_pivot_position(tableau)
        tableau = pivot(tableau, pivot_row, pivot_column)
        basis[pivot_row - 1] = pivot_column
        iterations += 1

    solution = extract_basic_solution(tableau, basis)
    return SolverResult(
        solution=solution,
        objective_value=compute_objective_value(c_array, solution, offset=objective_offset),
        iterations=iterations,
    )


def _get_pivot_position(tableau: np.ndarray) -> tuple[int, int]:
    rhs = tableau[1:, -1]
    pivot_row = int(np.argmin(rhs)) + 1

    candidate_columns = np.where(tableau[pivot_row, :-1] < -EPSILON)[0]
    if candidate_columns.size == 0:
        raise ValueError("Dual simplex found no valid entering column.")

    ratios = np.abs(tableau[0, candidate_columns] / tableau[pivot_row, candidate_columns])
    pivot_column = int(candidate_columns[np.argmin(ratios)])
    return pivot_row, pivot_column

