"""Tests for solving LPs in their original variable space."""

import numpy as np
import pytest

from simplex_codes import PivotStep, solve


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


@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10])
@pytest.mark.parametrize(("relation", "sense"), [("<=", "max"), ("=", "min"), (">=", "min")])
def test_solve_is_invariant_to_positive_constraint_scaling(scale, relation, sense):
    result = solve([1], [[scale]], [scale], [relation], sense=sense)

    np.testing.assert_allclose(result.solution, [1])
    assert result.objective_value == pytest.approx(1)


def test_solve_preserves_original_objective_when_costs_are_small():
    result = solve([1e-12], [[1]], [2], sense="max")

    np.testing.assert_allclose(result.solution, [2])
    assert result.objective_value == pytest.approx(2e-12, abs=0)


def test_solve_does_not_ignore_a_small_pivot_in_a_mixed_scale_row():
    result = solve([0, 1], [[1, 1e-10]], [1], sense="max")

    np.testing.assert_allclose(result.solution, [0, 1e10])


def test_solve_preserves_a_small_objective_coefficient_among_larger_costs():
    result = solve([1, -1e-16], [[0, 1]], [2])

    np.testing.assert_allclose(result.solution, [0, 2])
    assert result.objective_value == pytest.approx(-2e-16, abs=0)


def test_solve_does_not_report_unbounded_when_a_pivot_is_too_small():
    with pytest.raises(RuntimeError, match="Numerical failure.*pivot"):
        solve([0, 1], [[1, 1e-16]], [1], sense="max")


def test_solve_checks_original_constraints_after_phase_one():
    with pytest.raises(RuntimeError, match="original constraint"):
        solve([0], [[1], [1]], [0, 1e-10], ["<=", ">="])


def test_solve_detects_a_tiny_inconsistent_row_after_eliminating_a_fixed_variable():
    with pytest.raises(ValueError, match="infeasible"):
        solve([1], [[1e-10]], [2e-10], ["="], bounds=[(1, 1)])


@pytest.mark.parametrize(("lower", "upper"), [(10, 11), (0, 1e-12)])
def test_solve_verifies_bounds_in_original_coordinates(monkeypatch, lower, upper):
    def candidate_outside_transformed_bounds(*args):
        return np.array([2 * (upper - lower)]), 0

    monkeypatch.setattr("simplex_codes._two_phase._solve_standard_form", candidate_outside_transformed_bounds)
    with pytest.raises(RuntimeError, match="original bounds"):
        solve([1], [], [], bounds=[(lower, upper)])


def test_solve_rejects_an_objective_that_overflows_in_original_units():
    with pytest.raises(RuntimeError, match="Numerical failure"):
        solve([1e308], [[1]], [2], sense="max")


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


def test_solve_reports_original_coordinates_and_basis_changes_in_both_phases():
    steps: list[PivotStep] = []
    result = solve(
        [3], [[1]], [2], [">="],
        sense="max", bounds=[(1, 4)], on_step=steps.append,
    )

    assert steps == [
        PivotStep(0, 1, None, None, (1.0,), 3.0),
        PivotStep(1, 1, 0, 3, (2.0,), 6.0),
        PivotStep(2, 2, 1, 2, (4.0,), 12.0),
    ]
    np.testing.assert_allclose(result.solution, steps[-1].solution)
    assert result.iterations == steps[-1].iteration


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
