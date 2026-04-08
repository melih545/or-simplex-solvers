"""Revised simplex implementation."""

from __future__ import annotations

import numpy as np

from ._result import SolverResult
from ._utils import EPSILON, as_float_matrix, as_float_vector, compute_objective_value, validate_dimensions


def revised_simplex(
    c: list[float] | np.ndarray,
    A: list[list[float]] | np.ndarray,
    b: list[float] | np.ndarray,
    basic_indices: list[int] | np.ndarray,
    nonbasic_indices: list[int] | np.ndarray,
) -> SolverResult:
    """Solve an LP with the revised simplex method."""

    c_array = as_float_vector(c, "c")
    A_array = as_float_matrix(A, "A")
    b_array = as_float_vector(b, "b")
    validate_dimensions(c_array, A_array, b_array)

    basis = [int(index) for index in basic_indices]
    nonbasis = [int(index) for index in nonbasic_indices]
    _validate_partition(c_array.size, basis, nonbasis)

    iterations = 0
    while True:
        basis_matrix = A_array[:, basis]
        nonbasis_matrix = A_array[:, nonbasis]
        basis_costs = c_array[basis]
        nonbasis_costs = c_array[nonbasis]

        basis_inverse = np.linalg.inv(basis_matrix)
        reduced_costs = basis_costs @ basis_inverse @ nonbasis_matrix - nonbasis_costs

        if np.all(reduced_costs >= -EPSILON):
            basic_values = basis_inverse @ b_array
            solution = np.zeros(c_array.size, dtype=float)
            solution[basis] = basic_values
            solution[np.isclose(solution, 0.0, atol=EPSILON)] = 0.0
            return SolverResult(
                solution=solution,
                objective_value=compute_objective_value(c_array, solution),
                iterations=iterations,
            )

        entering_position = int(np.argmin(reduced_costs))
        entering_variable = nonbasis[entering_position]

        direction = basis_inverse @ A_array[:, entering_variable]
        basic_values = basis_inverse @ b_array
        ratios = np.where(direction > EPSILON, basic_values / direction, np.inf)
        if np.isinf(ratios).all():
            raise ValueError("Revised simplex found an unbounded entering direction.")

        leaving_row = int(np.argmin(ratios))
        leaving_variable = basis[leaving_row]
        basis[leaving_row] = entering_variable
        nonbasis[entering_position] = leaving_variable
        iterations += 1


def _validate_partition(variable_count: int, basis: list[int], nonbasis: list[int]) -> None:
    combined = sorted(basis + nonbasis)
    if combined != list(range(variable_count)):
        raise ValueError("basic_indices and nonbasic_indices must partition every variable exactly once.")

