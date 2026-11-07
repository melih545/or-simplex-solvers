"""Command-line interface for solving JSON linear programs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._two_phase import solve


_MODEL_FIELDS = {"c", "A", "b", "constraint_senses", "sense", "bounds", "max_iterations"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solve a linear program from JSON.")
    parser.add_argument("model", nargs="?", default="-", help="JSON file or - for standard input")
    args = parser.parse_args(argv)

    try:
        source = sys.stdin.read() if args.model == "-" else Path(args.model).read_text(encoding="utf-8")
        model = json.loads(source)
        if not isinstance(model, dict):
            raise ValueError("The model must be a JSON object.")
        missing = {"c", "A", "b"} - model.keys()
        if missing:
            raise ValueError(f"Missing model fields: {', '.join(sorted(missing))}.")
        unknown = model.keys() - _MODEL_FIELDS
        if unknown:
            raise ValueError(f"Unknown model fields: {', '.join(sorted(unknown))}.")
        if "sense" in model and not isinstance(model["sense"], str):
            raise ValueError("sense must be either 'max' or 'min'.")

        result = solve(**model)
        output = json.dumps({
            "solution": result.solution.tolist(),
            "objective_value": result.objective_value,
            "iterations": result.iterations,
        }, allow_nan=False)
    except (OSError, TypeError, ValueError, RuntimeError, OverflowError) as exc:
        print(f"simplex-solve: {exc}", file=sys.stderr)
        return 1

    print(output)
    return 0
