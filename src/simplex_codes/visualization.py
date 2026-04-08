"""Optional visualization helpers for simplex trajectories."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ._result import SolverResult
from .simplex import _solve_simplex


def plot_trajectory(points: Iterable[Iterable[float]], *, ax=None):
    """Plot a 2D simplex path using matplotlib when available."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Visualization support requires matplotlib. Install it with `python -m pip install -e \".[viz]\"`."
        ) from exc

    trajectory = np.asarray(list(points), dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("plot_trajectory expects an iterable of 2D points.")

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(trajectory[:, 0], trajectory[:, 1], marker="o", linewidth=1.5, color="#0f766e")
    for start, end in zip(trajectory[:-1], trajectory[1:]):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#0f766e", "lw": 1.5},
        )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Simplex Trajectory")
    ax.grid(True, alpha=0.3)
    return ax


def solve_and_plot_simplex(
    c: list[float] | np.ndarray,
    A: list[list[float]] | np.ndarray,
    b: list[float] | np.ndarray,
    sense: str = "max",
) -> tuple[SolverResult, object]:
    """Solve a 2D simplex problem and plot the path."""

    result, trajectory = _solve_simplex(c, A, b, sense=sense, record_trajectory=True)
    axis = plot_trajectory(trajectory)
    return result, axis

