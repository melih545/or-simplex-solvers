"""Smoke tests for example scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_simplex_example_runs():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "simplex_example.py")],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "objective_value: 41.25" in completed.stdout
