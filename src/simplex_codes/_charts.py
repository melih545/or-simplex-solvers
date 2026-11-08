"""Optional rendering for the workbench's two-dimensional models."""

import base64
from io import BytesIO
from itertools import combinations

import numpy as np

from .visualization import plot_feasible_region, plot_trajectory


def render_chart(A, b, relations, bounds, steps) -> str:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    points = np.asarray([step.solution for step in steps])
    region_A = []
    region_b = []
    for row, rhs, relation in zip(A, b, relations):
        if relation in ("<=", "="):
            region_A.append(row)
            region_b.append(rhs)
        if relation in (">=", "="):
            region_A.append([-value for value in row])
            region_b.append(-rhs)
    for index, (lo, hi) in enumerate(bounds):
        normal = [int(index == column) for column in range(2)]
        if lo is not None and np.isfinite(lo):
            region_A.append([-value for value in normal])
            region_b.append(-lo)
        if hi is not None and np.isfinite(hi):
            region_A.append(normal)
            region_b.append(hi)

    matrix = np.asarray(region_A, dtype=float).reshape(-1, 2)
    rhs = np.asarray(region_b, dtype=float)
    # Include visible corners as well as pivots in the chart extent.
    vertices = list(points)
    for i, j in combinations(range(len(rhs)), 2):
        pair = matrix[[i, j]]
        try:
            vertex = np.linalg.solve(pair, rhs[[i, j]])
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(vertex).all() and np.all(matrix @ vertex <= rhs + 1e-8):
            vertices.append(vertex)
    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    padding = np.maximum((upper - lower) * 0.13, np.maximum(abs(lower), abs(upper)) * 0.05)
    padding = np.where(padding > 0, padding, 1)

    fig = Figure(figsize=(6.2, 3.6), layout="constrained", dpi=135)
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    plot_feasible_region(
        matrix, rhs,
        origin=tuple(lower - padding), limits=tuple(upper + padding), ax=ax,
        draw_boundaries=False,
    )
    plot_trajectory(points, ax=ax, annotate_steps=True)
    ax.set(title="", xlabel="x1", ylabel="x2")
    buffer = BytesIO()
    canvas.print_png(buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
