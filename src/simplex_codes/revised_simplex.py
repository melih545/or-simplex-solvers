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
    if len(basis) != A_array.shape[0]:
        raise ValueError("The basis must contain one column per constraint.")

    iterations = 0
    while True:
        basis_matrix = A_array[:, basis]
        nonbasis_matrix = A_array[:, nonbasis]
        basis_costs = c_array[basis]
        nonbasis_costs = c_array[nonbasis]

        try:
            basic_values = np.linalg.solve(basis_matrix, b_array)
        except np.linalg.LinAlgError as exc:
            raise ValueError("The basis matrix is singular.") from exc
        if np.any(basic_values < -EPSILON):
            if iterations == 0:
                raise ValueError("The initial basis must be primal feasible.")
            raise ValueError("The basis lost primal feasibility during iteration.")

        dual_values = np.linalg.solve(basis_matrix.T, basis_costs)
        reduced_costs = dual_values @ nonbasis_matrix - nonbasis_costs

        if np.all(reduced_costs >= -EPSILON):
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

        direction = np.linalg.solve(basis_matrix, A_array[:, entering_variable])
        ratios = np.full_like(basic_values, np.inf)
        np.divide(basic_values, direction, out=ratios, where=direction > EPSILON)
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

