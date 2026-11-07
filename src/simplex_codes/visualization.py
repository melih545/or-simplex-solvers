"""Optional visualization helpers for simplex trajectories."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ._result import SolverResult
from .simplex import _solve_simplex


def _get_axis(ax):
    if ax is not None:
        return ax

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Visualization support requires matplotlib. Install it with `python -m pip install -e \".[viz]\"`."
        ) from exc

    _, ax = plt.subplots()
    return ax


def plot_trajectory(points: Iterable[Iterable[float]], *, ax=None, annotate_steps: bool = False):
    """Plot a 2D simplex path with optional zero-based step labels."""

    ax = _get_axis(ax)
    trajectory = np.asarray(list(points), dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("plot_trajectory expects an iterable of 2D points.")

    ax.plot(trajectory[:, 0], trajectory[:, 1], marker="o", linewidth=1.5, color="#0f766e")
    for start, end in zip(trajectory[:-1], trajectory[1:]):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#0f766e", "lw": 1.5},
        )

    if annotate_steps:
        labels: dict[tuple[float, float], list[str]] = {}
        for step, (x, y) in enumerate(trajectory):
            labels.setdefault((x, y), []).append(str(step))
        for point, steps in labels.items():
            ax.annotate(", ".join(steps), point, xytext=(6, 6), textcoords="offset points")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Simplex Trajectory")
    ax.grid(True, alpha=0.3)
    return ax


def plot_feasible_region(
    A, b, *, limits: tuple[float, float], origin: tuple[float, float] = (0, 0),
    draw_boundaries: bool = True, ax=None,
):
    """Draw Ax <= b in the rectangle between origin and limits."""
    constraints = np.asarray(A, dtype=float)
    rhs = np.asarray(b, dtype=float)
    viewport = np.asarray(limits, dtype=float)
    start = np.asarray(origin, dtype=float)
    if rhs.ndim != 1 or constraints.shape != (rhs.size, 2):
        raise ValueError("A must have two columns and one row per element of b.")
    if (
        viewport.shape != (2,) or start.shape != (2,)
        or not np.isfinite([viewport, start]).all() or np.any(viewport <= start)
    ):
        raise ValueError("origin and limits must be finite coordinates with origin below limits.")
    if not np.isfinite(constraints).all() or not np.isfinite(rhs).all():
        raise ValueError("A and b must contain only finite values.")

    ax = _get_axis(ax)
    width, height = viewport
    left, bottom = start
    polygon = np.array([[left, bottom], [width, bottom], [width, height], [left, height]])
    for normal, bound in zip(constraints, rhs):
        if polygon.size:
            polygon = _clip_polygon(polygon, normal, bound)

    if polygon.size:
        ax.fill(*polygon.T, color="#99f6e4", alpha=0.45, label="Feasible region")

    boundaries = zip(constraints, rhs) if draw_boundaries else ()
    for index, ((x_coeff, y_coeff), bound) in enumerate(boundaries, start=1):
        if x_coeff == y_coeff == 0:
            continue
        if abs(x_coeff) >= abs(y_coeff):
            point = (bound / x_coeff, 0)
            direction = (-y_coeff / x_coeff, 1)
        else:
            point = (0, bound / y_coeff)
            direction = (1, -x_coeff / y_coeff)
        ax.axline(
            point,
            np.add(point, direction),
            color=f"C{(index - 1) % 10}",
            linestyle="--",
            alpha=0.65,
            label=f"Constraint {index}",
        )

    ax.set_xlim(left, width)
    ax.set_ylim(bottom, height)
    return ax


def _clip_polygon(polygon: np.ndarray, normal: np.ndarray, bound: float) -> np.ndarray:
    """Clip a convex polygon against one linear inequality."""
    residuals = polygon @ normal - bound
    clipped = []
    for index, end in enumerate(polygon):
        start = polygon[index - 1]
        start_residual, end_residual = residuals[index - 1], residuals[index]
        if (start_residual < 0 < end_residual) or (end_residual < 0 < start_residual):
            fraction = start_residual / (start_residual - end_residual)
            clipped.append(start + fraction * (end - start))
        if end_residual <= 0:
            clipped.append(end)
    return np.asarray(clipped)


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

