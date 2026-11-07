"""Loopback-only HTTP interface for the simplex workbench."""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from io import BytesIO
from itertools import combinations
from threading import Lock

import numpy as np

from ._two_phase import solve
from .visualization import plot_feasible_region, plot_trajectory


_ASSETS = {
    "/": ("index.html", "text/html"),
    "/app.css": ("app.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
_CHART_LOCK = Lock()


class _Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path not in _ASSETS:
            self._send(b"Not found", "text/plain", 404)
            return
        filename, content_type = _ASSETS[self.path]
        self._send(files("simplex_codes").joinpath("_web", filename).read_bytes(), content_type)

    def do_POST(self) -> None:
        if self.path != "/solve":
            self._send(b"Not found", "text/plain", 404)
            return

        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a JSON model.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65_536:
                raise ValueError("The model must be between 1 and 65,536 bytes.")
            model = json.loads(self.rfile.read(length))
            response = json.dumps(_solve_model(model), allow_nan=False)
        except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as exc:
            self._send(json.dumps({"error": str(exc)}).encode(), "application/json", 400)
            return
        self._send(response.encode(), "application/json")


def _solve_model(model: dict) -> dict:
    if not isinstance(model, dict):
        raise ValueError("Expected a model object.")
    c, A, b = (model[key] for key in ("c", "A", "b"))
    sense = model.get("sense", "min")
    if sense not in ("min", "max"):
        raise ValueError("Select minimization or maximization.")
    if not 1 <= len(c) <= 12 or len(A) > 30:
        raise ValueError("Enter 1–12 variables and up to 30 constraints.")

    relations = model.get("constraint_senses", ["<="] * len(b))
    bounds = model.get("bounds", [(0, None)] * len(c))
    steps = []
    result = solve(c, A, b, relations, sense=sense, bounds=bounds, on_step=steps.append)
    response = {
        "solution": result.solution.tolist(),
        "objective_value": result.objective_value,
        "iterations": result.iterations,
        "steps": [asdict(step) for step in steps],
    }
    if len(c) == 2:
        try:
            with _CHART_LOCK:
                response["chart"] = _chart(A, b, relations, bounds, steps)
        except ImportError:
            response["chart_message"] = "Install the viz extra to show the 2D chart."
    return response


def _chart(A, b, relations, bounds, steps) -> str:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local simplex workbench.")
    parser.add_argument("--port", type=int, default=8765, help="local HTTP port")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65_535:
        parser.error("--port must be between 0 and 65535.")
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.port), _Handler) as server:
            print(f"Simplex Workbench · http://127.0.0.1:{server.server_port}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except OSError as exc:
        parser.exit(1, f"Could not start the workbench: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
