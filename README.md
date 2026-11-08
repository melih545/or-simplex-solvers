# OR Simplex Solvers

`simplex_codes` is a small Python package that collects a few simplex-based linear programming routines in a cleaner, reusable repository layout.

The project is intentionally lightweight and GitHub-focused: it is installable, tested, and documented, but still close to the original academic implementations.

## Supported Solvers

- Two-phase simplex for LPs in their original variables
- Primal simplex
- Dual simplex
- Revised simplex
- Big-M method

## Installation

Core package:

```bash
python -m pip install -e .
```

Optional visualization helpers:

```bash
python -m pip install -e ".[viz]"
```

## Quick Start

Pass the objective coefficients as written in the problem. This example
maximizes `8x + 5y` with two upper constraints and nonnegative variables.

```python
from simplex_codes import solve

result = solve(
    c=[8, 5],
    A=[[1, 1], [9, 5]],
    b=[6, 45],
    sense="max",
)

print(result.solution)         # [3.75 2.25]
print(result.objective_value)  # 41.25
```

`solve` defaults to minimization and treats every row as `<=`. Pass
`constraint_senses` with one `<=`, `>=`, or `=` per row to mix constraint
types. Variables default to `(0, None)`. To set bounds explicitly, pass one
`(lower, upper)` pair per variable. `None` means there is no bound on that
side, so `(None, None)` is a free variable and `(2, 2)` fixes a variable to 2.
Use `A=[]` and `b=[]` for a problem with only variable bounds.

The solver returns values for the original variables. It raises `ValueError`
for infeasible or unbounded problems. It raises `RuntimeError` when the pivot
budget (`max_iterations`, default 10,000) is exhausted or numerical accuracy
prevents a verified result. The solver scales rows and objective coefficients
internally and checks the returned solution against the original constraints
and variable bounds.

### Local workbench

Run the workbench and open the printed loopback URL in your browser.
The example model is already filled in. Edit the objective, variable bounds,
and constraints, then select **Modeli çöz** to see the solution and both
phases of the pivot trace. Two-variable models also show the feasible region
and the pivot path when the visualization extra is installed.

```bash
python3 -m pip install -e ".[api,viz]"
python3 -m simplex_codes._app
```

Open `http://127.0.0.1:8765` while the server is running. The installed
`simplex-app` command also starts it. Use `--port` to change the port
and press Ctrl+C to stop the server.

The workbench also serves OpenAPI documentation at `/docs`. Send the same
JSON model used by the CLI to `POST /api/v1/solve`. Use `?trace=true` to include
pivots and `?chart=true` to request a chart for a two-variable model. Valid
models return HTTP 200 with an explicit solve status, including mathematical
and numerical failures. Invalid models return HTTP 422.

### Capacity scenarios

In the workbench, use **Kapasite karşılaştırması** to load the sample or
upload a UTF-8 CSV. The first two columns are `product` and `profit`.
Each remaining column names a resource and gives the quantity used by one
unit of each product. Enter the baseline and alternative capacity for each
resource. The comparison shows profit, production quantities, resource usage
and binding constraints, and can be downloaded as JSON.

### Docker and monitoring

```bash
docker compose up --build --wait
python3 scripts/smoke_stack.py
docker compose down
```

The workbench listens on `http://127.0.0.1:8765`, Prometheus on
`http://127.0.0.1:9090`, and the provisioned Grafana dashboard on
`http://127.0.0.1:3000/d/simplex-overview`. The Compose stack is a local
demo. Grafana grants anonymous viewer access on its loopback port.
Override occupied host ports with `SIMPLEX_PORT`, `PROMETHEUS_PORT`, and
`GRAFANA_PORT` in both the Compose and smoke-test environments.

The API exposes `/healthz` and `/metrics`. Each solver attempt produces a
JSON log with the outcome, duration, and iteration count. The dashboard
tracks solve rate by status, p95 solve time, and the share of numerical
failures. Malformed requests rejected before invoking the solver are outside
these metrics.

### Production planning and pivot visualization

A workshop earns 8 units of profit per table and 5 per chair. Each product
uses one storage slot, and there are 6 slots available. A table requires
9 labor hours, a chair requires 5, and the workshop has 45 hours available.
With table and chair quantities `x` and `y`, the model maximizes `8x + 5y`
subject to `x + y <= 6`, `9x + 5y <= 45`, and `x, y >= 0`.

The continuous optimum is 3.75 tables and 2.25 chairs for a profit of 41.25.
The demo prints the entering and leaving variables at each pivot and saves
`simplex-path.png` with the feasible region, constraint lines, and numbered
pivots. Run it from the repository root after installing the visualization extra.

```bash
python examples/simplex_visualization.py
```

To collect the same trace from your own code, pass `on_step=steps.append` to
`solve`, where `steps` is an empty list. It receives an immutable
`PivotStep` at the initial basis and after each pivot. The solution and
objective use the original problem's coordinates and coefficients. Entering
and leaving indices are zero-based standard-form columns, including slack,
surplus, and artificial variables. Phase-one points may violate the original
constraints while the algorithm searches for a feasible basis.

### JSON command-line interface

After installation, use [the example model](examples/production_plan.json)
or save a model such as the following in `model.json`.

```json
{
  "c": [8, 5],
  "A": [[1, 1], [9, 5]],
  "b": [6, 45],
  "sense": "max",
  "constraint_senses": ["<=", "<="],
  "bounds": [[0, null], [0, null]]
}
```

```bash
simplex-solve model.json
python -m simplex_codes < model.json
```

The CLI requires `c`, `A`, and `b`. It uses the same defaults as `solve`
and accepts an optional `max_iterations`. A successful run writes a JSON
object with `solution`, `objective_value`, and `iterations` to standard output.
Input and solver errors go to standard error with exit status 1.
Pass `--structured` for the same JSON status contract as the HTTP API.

### Tableau API

Use the original methods to work directly with a tableau.

```python
from simplex_codes import simplex

result = simplex(
    c=[-8, -5, 0, 0],
    A=[
        [1, 1, 1, 0],
        [9, 5, 0, 1],
    ],
    b=[6, 45],
    sense="max",
)

print(result.solution)
print(result.objective_value)
print(result.iterations)
```

## Input Format

The low-level methods expect coefficient arrays in tableau-friendly form.

- `c` must include every variable coefficient already aligned with the constraint matrix columns.
- Slack, surplus, or artificial variable columns should already be present in `A` and `c`.
- For `big_m_method`, artificial variables are assumed to be the last `artificial_count` columns in `A`.

This keeps the implementations close to the original project and makes the examples easy to compare.

## Examples

Runnable examples live in [`examples/`](examples/):

- [`simplex_example.py`](examples/simplex_example.py)
- [`dual_simplex_example.py`](examples/dual_simplex_example.py)
- [`revised_simplex_example.py`](examples/revised_simplex_example.py)
- [`big_m_example.py`](examples/big_m_example.py)
- [`simplex_visualization.py`](examples/simplex_visualization.py)
- [`solver_comparison_example.py`](examples/solver_comparison_example.py)

The comparison example solves the same production model with primal simplex,
revised simplex, and SciPy/HiGHS. It checks that all three agree on the
quantities and profit. Install the optional comparison dependency to run it.

```bash
python -m pip install -e ".[compare]"
python examples/solver_comparison_example.py
```

The HiGHS differential tests run in CI. For a reproducible benchmark with
runtime, iteration count, objective value, and maximum primal violation, run
`python3 benchmarks/benchmark_solvers.py --seed 247`. It writes CSV to standard
output. Iteration counts depend on each solver's algorithm and presolve.

## Project Structure

```text
.
├── .github/workflows/
├── examples/
├── src/simplex_codes/
└── tests/
```

## Limitations

- The low-level methods assume inputs are already in a canonical matrix form.
- The implementations are educational and do not yet cover every degeneracy or infeasibility edge case.
- The visualization helper is only designed for 2D simplex paths.

## Development

Install the development dependencies, run the test suite, and build the wheel.

```bash
python -m pip install -e ".[api,compare,dev,viz]"
python -m pytest
python -m build --wheel
```

CI builds and installs the wheel before running tests and examples on Python
3.10 through 3.14.

The browser workflow is tested on Python 3.12 in CI. To run it locally with
the workbench dependencies installed, install the browser extra and Chromium.

```bash
python3 -m pip install -e ".[browser]"
python3 -m playwright install chromium
python3 -m pytest browser_tests
```

For an offline editable installation, install the required dependencies in
advance and run the following command.

```bash
python -m pip install --no-build-isolation -e ".[api,compare,dev,viz]"
```
