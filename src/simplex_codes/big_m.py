"""Big-M simplex implementation."""

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
from .simplex import _get_leaving_row


def big_m_method(
    c: list[float] | np.ndarray,
    A: list[list[float]] | np.ndarray,
    b: list[float] | np.ndarray,
    sense: str,
    artificial_count: int,
) -> SolverResult:
    """Solve a tableau-ready LP with a symbolic Big-M penalty."""

    c_array = as_float_vector(c, "c")
    A_array = as_float_matrix(A, "A")
    b_array = as_float_vector(b, "b")
    validate_dimensions(c_array, A_array, b_array)
    sense = normalize_sense(sense)

    if artificial_count <= 0 or artificial_count > c_array.size:
        raise ValueError("artificial_count must be between 1 and the number of variables.")

    if np.any(b_array < -EPSILON):
        raise ValueError("Big-M requires a primal feasible initial tableau.")

    first_artificial = c_array.size - artificial_count
    column_order = [*range(first_artificial, c_array.size), *range(first_artificial)]
    basis = find_unit_basis(A_array, column_order)

    # The leading row stores the coefficient of the symbolic penalty M.
    # Minimize that coefficient before improving the ordinary objective.
    penalty_row = np.zeros(c_array.size + 1)
    penalty_row[first_artificial:-1] = 1.0
    tableau = np.vstack([penalty_row, build_tableau(c_array, A_array, b_array)])
    tableau[:2] -= tableau[:2, basis] @ tableau[2:]

    iterations = 0
    while True:
        penalties = tableau[0, :-1]
        if np.any(penalties < -EPSILON):
            pivot_column = int(np.argmin(penalties))
        else:
            if tableau[0, -1] < -EPSILON:
                raise ValueError("The linear program is infeasible: artificial variables remain positive.")

            costs = tableau[1, :-1] if sense == "max" else -tableau[1, :-1]
            costs = np.where(np.abs(penalties) <= EPSILON, costs, np.inf)
            if not np.any(costs < -EPSILON):
                break
            pivot_column = int(np.argmin(costs))

        pivot_row = _get_leaving_row(tableau[1:], pivot_column)
        tableau = pivot(tableau, pivot_row + 1, pivot_column)
        basis[pivot_row - 1] = pivot_column
        iterations += 1

    solution = extract_basic_solution(tableau[1:], basis)
    return SolverResult(
        solution=solution,
        objective_value=compute_objective_value(c_array, solution, sense=sense),
        iterations=iterations,
    )

