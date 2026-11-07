"""Plot the simplex pivots for a two-product production plan."""

import os
import tempfile

cache_dir = os.path.join(tempfile.gettempdir(), "simplex-codes-matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", cache_dir)

from simplex_codes import PivotStep, solve
from simplex_codes.visualization import plot_feasible_region, plot_trajectory


def main() -> None:
    A = [[1, 1], [9, 5]]
    b = [6, 45]
    steps: list[PivotStep] = []
    result = solve([8, 5], A, b, sense="max", on_step=steps.append)

    axis = plot_feasible_region(A, b, limits=(7, 7))
    plot_trajectory((step.solution for step in steps), ax=axis, annotate_steps=True)
    axis.set(xlabel="Tables", ylabel="Chairs", title="Production planning with simplex")
    axis.legend()
    axis.figure.tight_layout()
    axis.figure.savefig("simplex-path.png", dpi=150)
    print("solution:", result.solution.tolist())
    print("objective_value:", result.objective_value)
    print("iterations:", result.iterations)
    labels = ["tables", "chairs", "storage slack", "labor slack"]
    for step in steps[1:]:
        print(
            f"pivot {step.iteration}: {labels[step.entering]} enters, "
            f"{labels[step.leaving]} leaves, profit = {step.objective_value:g}"
        )
    print("saved:", "simplex-path.png")


if __name__ == "__main__":
    main()
