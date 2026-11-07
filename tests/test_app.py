"""Tests for the models and results exchanged by the local workbench."""

from base64 import b64decode

import pytest

from simplex_codes._app import _solve_model


def test_workbench_reports_solution_trace_and_optional_chart():
    pytest.importorskip("matplotlib")
    result = _solve_model({"c": [8, 5], "A": [[1, 1], [9, 5]], "b": [6, 45], "sense": "max"})

    assert result["solution"] == pytest.approx([3.75, 2.25])
    assert result["objective_value"] == pytest.approx(41.25)
    assert [step["iteration"] for step in result["steps"]] == [0, 1, 2]
    assert b64decode(result["chart"]).startswith(b"\x89PNG\r\n\x1a\n")


def test_workbench_supports_free_and_fixed_variables_without_a_chart():
    result = _solve_model({
        "c": [1, 2, 3],
        "A": [[1, 0, 0], [0, 1, 0]],
        "b": [-2, 1],
        "constraint_senses": ["=", ">="],
        "bounds": [[None, None], [0, None], [2, 2]],
    })

    assert result["solution"] == pytest.approx([-2, 1, 2])
    assert result["objective_value"] == pytest.approx(6)
    assert "chart" not in result


def test_workbench_preserves_solver_errors():
    with pytest.raises(ValueError, match="infeasible"):
        _solve_model({"c": [1], "A": [[1]], "b": [-1]})
