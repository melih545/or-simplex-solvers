"""Big-M simplex implementation."""

from __future__ import annotations

import numpy as np

from ._result import SolverResult
from ._utils import (
    BIG_M,
    EPSILON,
    as_float_matrix,
    as_float_vector,
    build_tableau,
    compute_objective_value,
    extract_basic_solution,
    normalize_sense,
    pivot,
    validate_dimensions,
)
from .simplex import _can_be_improved, _get_pivot_position


def big_m_method(
    c: list[float] | np.ndarray,
    A: list[list[float]] | np.ndarray,
    b: list[float] | np.ndarray,
    sense: str,
    artificial_count: int,
) -> SolverResult:
    """Solve a tableau-ready LP with a Big-M style initialization."""

    c_array = as_float_vector(c, "c")
    A_array = as_float_matrix(A, "A")
    b_array = as_float_vector(b, "b")
    validate_dimensions(c_array, A_array, b_array)
    sense = normalize_sense(sense)

    if artificial_count <= 0 or artificial_count > c_array.size:
        raise ValueError("artificial_count must be between 1 and the number of variables.")

    tableau = _add_penalties(c_array, A_array, b_array, sense=sense, artificial_count=artificial_count)
    artificial_indices = np.arange(c_array.size - artificial_count, c_array.size)
    iterations = 0

    for _ in range(artificial_count):
        pivot_row, pivot_column = _get_cleanup_pivot_position(tableau, artificial_indices)
        tableau = pivot(tableau, pivot_row, pivot_column)
        iterations += 1

    while _can_be_improved(tableau, sense):
        pivot_row, pivot_column = _get_pivot_position(tableau, sense)
        tableau = pivot(tableau, pivot_row, pivot_column)
        iterations += 1

    solution = extract_basic_solution(tableau, c_array.size)
    return SolverResult(
        solution=solution,
        objective_value=compute_objective_value(c_array, solution, sense=sense),
        iterations=iterations,
    )


def _add_penalties(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    *,
    sense: str,
    artificial_count: int,
) -> np.ndarray:
    tableau = build_tableau(c, A, b)
    penalty = BIG_M if sense == "max" else -BIG_M
    tableau[0, -artificial_count - 1 : -1] = penalty
    return tableau


def _get_cleanup_pivot_position(tableau: np.ndarray, artificial_indices: np.ndarray) -> tuple[int, int]:
    objective_row = tableau[0, :-1].copy()

    for _ in range(objective_row.size):
        pivot_column = int(np.argmin(objective_row))
        if pivot_column not in artificial_indices:
            break
        objective_row[pivot_column] = 0.0
    else:
        raise ValueError("Big-M cleanup found no non-artificial entering column.")

    column = tableau[1:, pivot_column]
    rhs = tableau[1:, -1]
    ratios = np.where(column > EPSILON, rhs / column, np.inf)
    if np.isinf(ratios).all():
        raise ValueError("Big-M cleanup found no valid leaving row.")

    pivot_row = int(np.argmin(ratios)) + 1
    return pivot_row, pivot_column

