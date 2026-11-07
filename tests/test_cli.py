"""Tests for the JSON command-line interface."""

import json
import subprocess
import sys

import pytest


MODEL = {"c": [8, 5], "A": [[1, 1], [9, 5]], "b": [6, 45], "sense": "max"}


@pytest.mark.parametrize("from_file", [False, True])
def test_cli_solves_a_json_model(from_file, tmp_path):
    command = [sys.executable, "-m", "simplex_codes"]
    source = json.dumps(MODEL)
    if from_file:
        model_file = tmp_path / "model.json"
        model_file.write_text(source, encoding="utf-8")
        command.append(str(model_file))
        source = ""

    completed = subprocess.run(command, input=source, capture_output=True, text=True, check=True)

    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "solution": [3.75, 2.25], "objective_value": 41.25, "iterations": 2,
    }


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("{", "Expecting property name"),
        ('{"c": [1]}', "Missing model fields"),
        (json.dumps(MODEL | {"on_step": None}), "Unknown model fields"),
        (json.dumps({"c": [1], "A": [[1]], "b": [1], "constraint_senses": [">="],
                     "bounds": [[0, 0]]}), "infeasible"),
    ],
)
def test_cli_reports_input_and_solver_errors_on_stderr(source, message):
    completed = subprocess.run(
        [sys.executable, "-m", "simplex_codes", "-"],
        input=source, capture_output=True, text=True, check=False,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert message in completed.stderr
