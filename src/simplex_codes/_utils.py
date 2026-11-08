"""Internal helpers shared by the solver modules."""

from __future__ import annotations

from typing import Iterable

import numpy as np

EPSILON = 1e-9


def as_float_vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like input.")
    return array.copy()


def as_float_matrix(values: Iterable[Iterable[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array-like input.")
    return array.copy()


def validate_dimensions(c: np.ndarray, A: np.ndarray, b: np.ndarray) -> None:
    if A.shape[1] != c.shape[0]:
        raise ValueError("A column count must match the size of c.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("A row count must match the size of b.")


def normalize_sense(sense: str) -> str:
    normalized = sense.lower()
    if normalized not in {"max", "min"}:
        raise ValueError("sense must be either 'max' or 'min'.")
    return normalized


def build_tableau(
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    objective_offset: float = 0.0,
) -> np.ndarray:
    objective_row = np.append(c, float(objective_offset))
    constraints = np.column_stack([A, b])
    return np.vstack([objective_row, constraints]).astype(float)


def pivot(
    tableau: np.ndarray, row: int, column: int, *,
    zero_small: bool = True, pivot_tolerance: float = EPSILON,
) -> np.ndarray:
    updated = tableau.copy()
    pivot_value = updated[row, column]
    if abs(pivot_value) <= pivot_tolerance:
        raise ValueError("Cannot pivot on a zero value.")

    updated[row] = updated[row] / pivot_value
    for current_row in range(updated.shape[0]):
        if current_row == row:
            continue
        updated[current_row] = updated[current_row] - updated[row] * updated[current_row, column]

    if zero_small:
        updated[np.isclose(updated, 0.0, atol=EPSILON)] = 0.0
    return updated


def find_unit_basis(A: np.ndarray, columns: Iterable[int]) -> np.ndarray:
    """Find one unit column per row, in order of column preference."""
    basis = np.full(A.shape[0], -1, dtype=int)
    for column_index in columns:
        column = A[:, column_index]
        nonzero_rows = np.flatnonzero(np.abs(column) > EPSILON)
        if nonzero_rows.size == 1:
            row = nonzero_rows[0]
            if basis[row] == -1 and np.isclose(column[row], 1.0, atol=EPSILON, rtol=0):
                basis[row] = column_index
    if np.any(basis == -1):
        raise ValueError("The initial tableau must contain a unit basis.")
    return basis


def extract_basic_solution(
    tableau: np.ndarray, basis: np.ndarray, *, zero_small: bool = True,
) -> np.ndarray:
    solution = np.zeros(tableau.shape[1] - 1, dtype=float)
    solution[basis] = tableau[1:, -1]
    if zero_small:
        solution[np.isclose(solution, 0.0, atol=EPSILON)] = 0.0
    return solution


def compute_objective_value(
    c: np.ndarray,
    solution: np.ndarray,
    *,
    sense: str | None = None,
    offset: float = 0.0,
) -> float:
    value = float(np.dot(c, solution) + offset)
    if sense == "max":
        return -value
    return value

