# Simplex Codes

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

Run the test suite:

```bash
python -m pytest
```

If you are working in an offline environment with the required build tools already installed, use:

```bash
python -m pip install --no-build-isolation -e .
```
