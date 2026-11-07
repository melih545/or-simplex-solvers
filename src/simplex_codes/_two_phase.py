"""Two-phase simplex for linear programs in their original variables."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from ._result import SolverResult
from ._utils import (
    EPSILON,
    as_float_matrix,
    as_float_vector,
    build_tableau,
    extract_basic_solution,
    normalize_sense,
    pivot,
    validate_dimensions,
)
from .simplex import _get_leaving_row


@dataclass(frozen=True, slots=True)
class PivotStep:
    """An immutable snapshot of a two-phase simplex basis."""

    iteration: int
    phase: int
    entering: int | None
    leaving: int | None
    solution: tuple[float, ...]
    objective_value: float


_Trace = Callable[[np.ndarray, np.ndarray, int, int, int | None, int | None], None]


def solve(
    c: ArrayLike,
    A: ArrayLike,
    b: ArrayLike,
    constraint_senses: Sequence[str] | None = None,
    *,
    sense: str = "min",
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    max_iterations: int = 10_000,
    on_step: Callable[[PivotStep], None] | None = None,
) -> SolverResult:
    """Optimize c @ x subject to row constraints and variable bounds.

    Rows default to "<=". Variables default to [0, +inf); a None bound
    endpoint leaves that side unbounded. The result uses the original
    variables and counts pivots in both phases, including basis cleanup.

    Pass on_step to receive the initial basis and each subsequent pivot.
    Entering and leaving indices refer to zero-based standard-form columns.
    Snapshots always report the original objective and variables. Phase-one
    snapshots may be infeasible for the original constraints.

    Raises ValueError for invalid, infeasible, or unbounded problems and
    RuntimeError when the pivot budget is exhausted.
    """
    costs = as_float_vector(c, "c")
    rhs = as_float_vector(b, "b")
    constraints = np.asarray(A, dtype=float)
    if constraints.shape == (0,):
        constraints = constraints.reshape(0, costs.size)
    constraints = as_float_matrix(constraints, "A")
    validate_dimensions(costs, constraints, rhs)
    sense = normalize_sense(sense)
    if not all(np.isfinite(array).all() for array in (costs, constraints, rhs)):
        raise ValueError("c, A, and b must contain only finite values.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer.")

    relations = list(constraint_senses) if constraint_senses is not None else ["<="] * rhs.size
    if len(relations) != rhs.size or any(relation not in ("<=", ">=", "=") for relation in relations):
        raise ValueError("constraint_senses must contain one '<=', '>=', or '=' per row.")
    if bounds is None:
        bounds = [(0.0, None)] * costs.size
    elif len(bounds) != costs.size:
        raise ValueError("bounds must contain one (lower, upper) pair per variable.")

    # Write x as a shift plus signed nonnegative variables. Free variables
    # use a difference of two columns; fixed variables need no column.
    shift = np.zeros(costs.size)
    columns: list[int] = []
    signs: list[float] = []
    upper_limits: list[tuple[int, float]] = []
    for variable, (lower, upper) in enumerate(bounds):
        lower = -np.inf if lower is None else float(lower)
        upper = np.inf if upper is None else float(upper)
        if np.isnan(lower) or np.isnan(upper) or lower == np.inf or upper == -np.inf or lower > upper:
            raise ValueError(f"Invalid bounds for variable {variable}.")
        if np.isfinite(lower):
            shift[variable] = lower
            if upper == lower:
                continue
            if np.isfinite(upper):
                upper_limits.append((len(columns), upper - lower))
        elif np.isfinite(upper):
            shift[variable] = upper
        else:
            columns.append(variable)
            signs.append(1.0)
        columns.append(variable)
        signs.append(1.0 if np.isfinite(lower) else -1.0)

    rhs -= constraints @ shift
    constraints = constraints[:, columns] * signs
    if upper_limits:
        bound_rows = np.zeros((len(upper_limits), len(columns)))
        for row, (column, _) in enumerate(upper_limits):
            bound_rows[row, column] = 1.0
        constraints = np.vstack([constraints, bound_rows])
        rhs = np.append(rhs, [limit for _, limit in upper_limits])
        relations.extend(["<="] * len(upper_limits))

    transformed_costs = costs[columns] * signs
    if sense == "max":
        transformed_costs = -transformed_costs

    def original_solution(values: np.ndarray) -> np.ndarray:
        solution = shift.copy()
        np.add.at(solution, columns, np.asarray(signs) * values)
        return solution

    trace: _Trace | None = None
    if on_step is not None:
        callback = on_step

        def record_step(tableau, basis, iteration, phase, entering, leaving):
            values = extract_basic_solution(tableau, basis, zero_small=False)
            solution = original_solution(values[:len(columns)])
            callback(PivotStep(
                iteration, phase, entering, leaving,
                tuple(solution.tolist()), float(costs @ solution),
            ))

        trace = record_step

    values, iterations = _solve_standard_form(
        constraints, rhs, relations, transformed_costs, max_iterations, trace,
    )

    solution = original_solution(values)
    return SolverResult(solution, float(costs @ solution), iterations)


def _solve_standard_form(
    A: np.ndarray, b: np.ndarray, relations: list[str], costs: np.ndarray, max_iterations: int,
    trace: _Trace | None,
) -> tuple[np.ndarray, int]:
    # Make every RHS nonnegative before adding slack, surplus and artificial columns.
    for row in np.flatnonzero(b < 0):
        A[row] *= -1
        b[row] *= -1
        relations[row] = {"<=": ">=", ">=": "<=", "=": "="}[relations[row]]

    inequality_rows = [row for row, relation in enumerate(relations) if relation != "="]
    artificial_rows = [row for row, relation in enumerate(relations) if relation != "<="]
    first_artificial = costs.size + len(inequality_rows)
    variable_count = first_artificial + len(artificial_rows)
    standard = np.zeros((b.size, variable_count))
    standard[:, :costs.size] = A
    basis = np.empty(b.size, dtype=int)
    for column, row in enumerate(inequality_rows, start=costs.size):
        standard[row, column] = 1.0 if relations[row] == "<=" else -1.0
        basis[row] = column
    for column, row in enumerate(artificial_rows, start=first_artificial):
        standard[row, column] = 1.0
        basis[row] = column

    phase_one_costs = np.zeros(variable_count)
    phase_one_costs[first_artificial:] = 1.0
    tableau = build_tableau(phase_one_costs, standard, b)
    tableau[0] -= phase_one_costs[basis] @ tableau[1:]
    if trace is not None:
        trace(tableau, basis, 0, 1 if artificial_rows else 2, None, None)
    tableau, iterations = _minimize(tableau, basis, 0, max_iterations, phase=1, trace=trace)

    artificial_basis = basis >= first_artificial
    if np.sum(tableau[1:, -1][artificial_basis]) > EPSILON:
        raise ValueError("The linear program is infeasible.")

    # Pivot zero-valued artificials out. If no real column can enter,
    # the row reduces to 0 = 0 once the artificial columns are removed.
    keep_rows = np.ones(basis.size, dtype=bool)
    for row in np.flatnonzero(artificial_basis):
        candidates = np.abs(tableau[row + 1, :first_artificial]) > EPSILON
        candidates[basis[basis < first_artificial]] = False
        entering = np.flatnonzero(candidates)
        if entering.size == 0:
            keep_rows[row] = False
            continue
        tableau[row + 1, -1] = 0.0
        tableau, iterations = _take_step(
            tableau, basis, row + 1, int(entering[0]), iterations, max_iterations,
            phase=1, trace=trace,
        )

    tableau = tableau[np.r_[True, keep_rows]]
    tableau = np.column_stack([tableau[:, :first_artificial], tableau[:, -1]])
    basis = basis[keep_rows]

    phase_two_costs = np.zeros(first_artificial)
    phase_two_costs[:costs.size] = costs
    tableau[0, :-1] = phase_two_costs
    tableau[0, -1] = 0.0
    tableau[0] -= phase_two_costs[basis] @ tableau[1:]
    tableau, iterations = _minimize(
        tableau, basis, iterations, max_iterations, phase=2, trace=trace,
    )
    solution = extract_basic_solution(tableau, basis, zero_small=False)
    return solution[:costs.size], iterations


def _minimize(
    tableau: np.ndarray, basis: np.ndarray, iterations: int, max_iterations: int,
    *, phase: int, trace: _Trace | None,
) -> tuple[np.ndarray, int]:
    while True:
        improving = tableau[0, :-1] < -EPSILON
        improving[basis] = False
        entering = np.flatnonzero(improving)
        if entering.size == 0:
            return tableau, iterations

        # Bland's rule uses the smallest variable index to break pivot ties.
        column = int(entering[0])
        row = _get_leaving_row(tableau, column, basis=basis)
        tableau, iterations = _take_step(
            tableau, basis, row, column, iterations, max_iterations, phase=phase, trace=trace,
        )


def _take_step(
    tableau: np.ndarray, basis: np.ndarray, row: int, column: int, iterations: int, max_iterations: int,
    *, phase: int, trace: _Trace | None,
) -> tuple[np.ndarray, int]:
    if iterations >= max_iterations:
        raise RuntimeError("The simplex iteration limit was reached.")
    tableau = pivot(tableau, row, column, zero_small=False)
    leaving = int(basis[row - 1])
    basis[row - 1] = column
    iterations += 1
    if trace is not None:
        trace(tableau, basis, iterations, phase, column, leaving)
    return tableau, iterations
