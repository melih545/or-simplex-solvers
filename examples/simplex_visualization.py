"""Simplex visualization example."""

import os
import tempfile

cache_dir = os.path.join(tempfile.gettempdir(), "simplex-codes-matplotlib")
os.makedirs(cache_dir, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", cache_dir)

from simplex_codes.visualization import solve_and_plot_simplex


def main() -> None:
    result, axis = solve_and_plot_simplex(
        c=[-8, -5, 0, 0],
        A=[
            [1, 1, 1, 0],
            [9, 5, 0, 1],
        ],
        b=[6, 45],
        sense="max",
    )
    axis.figure.tight_layout()
    axis.figure.savefig("simplex-path.png", dpi=150)
    print("solution:", result.solution.tolist())
    print("objective_value:", result.objective_value)
    print("iterations:", result.iterations)
    print("saved:", "simplex-path.png")


if __name__ == "__main__":
    main()
