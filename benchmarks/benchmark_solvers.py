"""Print reproducible CSV benchmark measurements for bounded LPs."""

import argparse
import csv
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import linprog

from simplex_codes import solve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=247)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = csv.writer(sys.stdout, lineterminator="\n")
    rows.writerow(["solver", "seed", "variables", "constraints", "seconds", "iterations",
                   "objective", "max_primal_violation"])
    for variables, constraints in [(5, 10), (15, 30), (30, 60)]:
        A = rng.integers(1, 10, size=(constraints, variables))
        b = rng.integers(10, 100, size=constraints)
        c = rng.integers(1, 10, size=variables)
        start = perf_counter()
        simplex = solve(c, A, b, sense="max")
        simplex_time = perf_counter() - start
        start = perf_counter()
        highs = linprog(-c, A_ub=A, b_ub=b, method="highs")
        highs_time = perf_counter() - start
        if not highs.success:
            raise RuntimeError(highs.message)
        np.testing.assert_allclose(simplex.objective_value, -highs.fun, rtol=1e-7)
        for name, values, iterations, duration in [
            ("two-phase", simplex.solution, simplex.iterations, simplex_time),
            ("highs", highs.x, highs.nit, highs_time),
        ]:
            violation = max(0, np.max(A @ values - b), np.max(-values))
            rows.writerow([name, args.seed, variables, constraints, f"{duration:.9g}",
                           iterations, f"{c @ values:.12g}", f"{violation:.9g}"])


if __name__ == "__main__":
    main()
