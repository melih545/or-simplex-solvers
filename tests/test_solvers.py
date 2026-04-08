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
