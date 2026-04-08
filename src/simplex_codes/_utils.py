"""Internal helpers shared by the solver modules."""

from __future__ import annotations

from typing import Iterable

import numpy as np

EPSILON = 1e-9
BIG_M = 10_000.0


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


def pivot(tableau: np.ndarray, row: int, column: int) -> np.ndarray:
    updated = tableau.copy()
    pivot_value = updated[row, column]
    if np.isclose(pivot_value, 0.0, atol=EPSILON):
        raise ValueError("Cannot pivot on a zero value.")

    updated[row] = updated[row] / pivot_value
    for current_row in range(updated.shape[0]):
        if current_row == row:
            continue
        updated[current_row] = updated[current_row] - updated[row] * updated[current_row, column]

    updated[np.isclose(updated, 0.0, atol=EPSILON)] = 0.0
    return updated


def extract_basic_solution(tableau: np.ndarray, variable_count: int) -> np.ndarray:
    solution = np.zeros(variable_count, dtype=float)

    for column_index in range(variable_count):
        column = tableau[:, column_index]
        if not np.isclose(column[0], 0.0, atol=EPSILON):
            continue

        constraint_column = column[1:]
        pivot_rows = np.where(np.isclose(constraint_column, 1.0, atol=EPSILON))[0]
        if pivot_rows.size != 1:
            continue

        pivot_row = pivot_rows[0]
        if not np.all(np.isclose(np.delete(constraint_column, pivot_row), 0.0, atol=EPSILON)):
            continue

        solution[column_index] = tableau[pivot_row + 1, -1]

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

