"""Smoke tests for example scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_simplex_example_runs():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "simplex_example.py")],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "objective_value: 41.25" in completed.stdout
