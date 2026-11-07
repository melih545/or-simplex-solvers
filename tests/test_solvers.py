"""Regression tests for the packaged solvers."""

import numpy as np
import pytest

from simplex_codes import big_m_method, dual_simplex, revised_simplex, simplex


def test_simplex_regression(simplex_problem):
    result = simplex(**simplex_problem)
    np.testing.assert_allclose(result.solution, [3.75, 2.25, 0.0, 0.0])
    assert result.objective_value == pytest.approx(41.25)
    assert result.iterations == 2


def test_dual_simplex_regression(dual_simplex_problem):
    result = dual_simplex(**dual_simplex_problem)
    np.testing.assert_allclose(result.solution, [0.0, 2.0, 0.0, 1.0])
    assert result.objective_value == pytest.approx(-6.0)
    assert result.iterations == 1


def test_revised_simplex_regression(revised_simplex_problem):
    result = revised_simplex(**revised_simplex_problem)
    np.testing.assert_allclose(result.solution, [1.0, 1.0, 0.0, 0.0])
    assert result.objective_value == pytest.approx(7.0)
    assert result.iterations == 2


def test_big_m_regression(big_m_problem):
    result = big_m_method(**big_m_problem)
    np.testing.assert_allclose(result.solution, [5.0, 5.0, 0.25, 0.0, 0.0, 0.0])
    assert result.objective_value == pytest.approx(-25.0)
    assert result.iterations == 2


def test_simplex_duplicate_unit_columns_use_one_basic_variable():
    A = np.array([[1, 1]])
    result = simplex(c=[0, 0], A=A, b=[1])

    np.testing.assert_allclose(A @ result.solution, [1])
    np.testing.assert_allclose(np.sort(result.solution), [0, 1])


def test_dual_simplex_tracks_entering_variable_with_duplicate_columns():
    A = np.array([[-1, -1, 1]])
    result = dual_simplex(c=[-1, -1, 0], A=A, b=[-1])

    np.testing.assert_allclose(A @ result.solution, [-1])
    assert result.objective_value == pytest.approx(-1)
    assert result.iterations == 1


def test_big_m_rejects_positive_artificial_variable():
    # x + s = 0 and x + a = 1 force the artificial variable a to be 1.
    with pytest.raises(ValueError, match="infeasible"):
        big_m_method(
            c=[0, 0, 0],
            A=[[1, 1, 0], [1, 0, 1]],
            b=[0, 1],
            sense="min",
            artificial_count=1,
        )


@pytest.mark.parametrize(("sense", "objective"), [("min", 20_000), ("max", -20_000)])
def test_big_m_prioritizes_feasibility_over_large_objective_coefficients(sense, objective):
    result = big_m_method(c=[20_000, 0], A=[[1, 1]], b=[1], sense=sense, artificial_count=1)

    np.testing.assert_allclose(result.solution, [1, 0])
    assert result.objective_value == pytest.approx(objective)


def test_revised_simplex_rejects_infeasible_starting_basis():
    with pytest.raises(ValueError, match="initial basis.*primal feasible"):
        revised_simplex(c=[0], A=[[1]], b=[-1], basic_indices=[0], nonbasic_indices=[])
