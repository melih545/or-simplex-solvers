"""Shared model execution and outcome contract for external interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._errors import InfeasibleError, IterationLimitError, NumericalError, UnboundedError
from ._two_phase import PivotStep, solve


Status = Literal["optimal", "infeasible", "unbounded", "iteration_limit", "numerical_error", "invalid_model"]
_MODEL_FIELDS = {"c", "A", "b", "constraint_senses", "sense", "bounds", "max_iterations"}


@dataclass(frozen=True, slots=True)
class SolveOutcome:
    status: Status
    solution: list[float] | None = None
    objective_value: float | None = None
    iterations: int | None = None
    steps: list[PivotStep] | None = None
    error: str | None = None


def execute_model(model: object, *, trace: bool = False) -> SolveOutcome:
    """Solve a JSON-shaped model and classify expected failures by type."""
    steps: list[PivotStep] | None = [] if trace else None
    try:
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
        result = solve(**model, on_step=steps.append if steps is not None else None)
    except InfeasibleError as exc:
        return SolveOutcome("infeasible", error=str(exc))
    except UnboundedError as exc:
        return SolveOutcome("unbounded", error=str(exc))
    except IterationLimitError as exc:
        return SolveOutcome("iteration_limit", error=str(exc))
    except (NumericalError, OverflowError) as exc:
        return SolveOutcome("numerical_error", error=str(exc))
    except (TypeError, ValueError) as exc:
        return SolveOutcome("invalid_model", error=str(exc))
    return SolveOutcome(
        "optimal", result.solution.tolist(), result.objective_value, result.iterations, steps,
    )
