"""Tests for solving LPs in their original variable space."""

import numpy as np
import pytest

from simplex_codes import solve


def test_solve_maximizes_with_natural_costs_and_default_constraints():
    result = solve([8, 5], [[1, 1], [9, 5]], [6, 45], sense="max")

    np.testing.assert_allclose(result.solution, [3.75, 2.25])
    assert result.objective_value == pytest.approx(41.25)


def test_solve_mixed_constraints_and_negative_rhs():
    result = solve(
        [1, 2],
        [[-1, 0], [0, 1], [1, 1]],
        [-2, 1, 5],
        ["<=", ">=", "="],
    )

    np.testing.assert_allclose(result.solution, [4, 1])
    assert result.objective_value == pytest.approx(6)


def test_solve_reconstructs_bounded_free_and_fixed_variables():
    result = solve(
        [1, 1, 1, -1],
        [[0, 0, 1, 1]],
        [0],
        ["="],
        sense="max",
        bounds=[(-3, -2), (None, -4), (None, None), (2, 2)],
    )

    np.testing.assert_allclose(result.solution, [-2, -4, -2, 2])
    assert result.objective_value == pytest.approx(-10)


def test_solve_minimizes_at_shifted_lower_bound_without_constraints():
    result = solve([3], [], [], bounds=[(-2, None)], max_iterations=0)

    np.testing.assert_allclose(result.solution, [-2])
    assert result.objective_value == pytest.approx(-6)
    assert result.iterations == 0


def test_solve_handles_all_fixed_variables():
    result = solve([3], [[1]], [2], ["="], bounds=[(2, 2)])

    np.testing.assert_allclose(result.solution, [2])
    assert result.objective_value == pytest.approx(6)


def test_solve_preserves_small_positive_variable_values():
    result = solve([1], [[1]], [1e-10], sense="max")

    np.testing.assert_allclose(result.solution, [1e-10], atol=0)


def test_solve_drops_redundant_rows_and_pivots_out_zero_artificials():
    result = solve(
        [0, 1],
        [[-1, 0], [-2, 0], [0, 1]],
        [0, 0, 3],
        ["=", "=", "<="],
        sense="max",
    )

    np.testing.assert_allclose(result.solution, [0, 3])
    assert result.iterations == 2


def test_solve_detects_infeasibility_with_bounds():
    with pytest.raises(ValueError, match="infeasible"):
        solve([1], [[1]], [2], [">="], bounds=[(0, 1)])


def test_solve_detects_unboundedness_after_phase_one():
    with pytest.raises(ValueError, match="unbounded"):
        solve([1], [[1]], [1], [">="], sense="max")


def test_solve_shares_iteration_budget_between_phases():
    with pytest.raises(RuntimeError, match="iteration limit"):
        solve([-1], [[1]], [1], [">="], bounds=[(0, 2)], max_iterations=1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"constraint_senses": [">"]}, "constraint_senses"),
        ({"bounds": [(2, 1)]}, "bounds"),
        ({"bounds": []}, "bounds"),
        ({"max_iterations": -1}, "max_iterations"),
        ({"b": [np.nan]}, "finite"),
    ],
)
def test_solve_validates_input(kwargs, message):
    problem = {"c": [1], "A": [[1]], "b": [1]}
    with pytest.raises(ValueError, match=message):
        solve(**(problem | kwargs))
