"""Command-line interface for solving JSON linear programs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ._service import SolveOutcome, execute_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solve a linear program from JSON.")
    parser.add_argument("model", nargs="?", default="-", help="JSON file or - for standard input")
    parser.add_argument("--structured", action="store_true", help="emit a status and JSON errors")
    args = parser.parse_args(argv)

    try:
        source = sys.stdin.read() if args.model == "-" else Path(args.model).read_text(encoding="utf-8")
        outcome = execute_model(json.loads(source))
    except (OSError, ValueError) as exc:
        outcome = SolveOutcome("invalid_model", error=str(exc))
    if args.structured:
        print(json.dumps({key: value for key, value in asdict(outcome).items() if value is not None}))
        return int(outcome.status != "optimal")
    if outcome.status != "optimal":
        print(f"simplex-solve: {outcome.error}", file=sys.stderr)
        return 1

    print(json.dumps({
        "solution": outcome.solution,
        "objective_value": outcome.objective_value,
        "iterations": outcome.iterations,
    }, allow_nan=False))
    return 0
