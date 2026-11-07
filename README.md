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
for infeasible or unbounded problems and `RuntimeError` when the pivot budget
(`max_iterations`, default 10,000) is exhausted.

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

After installation, save a model such as the following in `model.json`.

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
python -m pip install -e ".[dev,viz]"
python -m pytest
python -m build --wheel
```

CI builds and installs the wheel before running tests and examples on Python
3.10 through 3.14.

For an offline editable installation, install the required dependencies in
advance and run the following command.

```bash
python -m pip install --no-build-isolation -e ".[dev,viz]"
```
