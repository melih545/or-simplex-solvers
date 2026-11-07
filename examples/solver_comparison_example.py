"""Compare primal simplex, revised simplex, and HiGHS on one LP."""

import numpy as np
from scipy.optimize import linprog

from simplex_codes import revised_simplex, simplex


def main() -> None:
    c = np.array([8, 5])
    A = np.array([[1, 1], [9, 5]])
    b = np.array([6, 45])

    # The tableau methods take explicit slack columns. Primal simplex
    # expects negated maximization costs, while revised simplex uses c as is.
    tableau_A = np.column_stack([A, np.eye(len(b))])
    tableau_c = np.r_[c, np.zeros(len(b))]
    primal = simplex(-tableau_c, tableau_A, b, sense="max")
    revised = revised_simplex(
        tableau_c, tableau_A, b, basic_indices=[2, 3], nonbasic_indices=[0, 1],
    )

    # linprog minimizes, so negate the costs and convert its objective back.
    reference = linprog(-c, A_ub=A, b_ub=b, method="highs")
    if not reference.success:
        raise RuntimeError(f"HiGHS failed: {reference.message}")

    for result in (primal, revised):
        np.testing.assert_allclose(result.solution[:2], reference.x)
        np.testing.assert_allclose(result.objective_value, -reference.fun)

    print(f"{'Solver':<18} {'Tables':>8} {'Chairs':>8} {'Profit':>8}")
    for name, values, objective in (
        ("Primal simplex", primal.solution[:2], primal.objective_value),
        ("Revised simplex", revised.solution[:2], revised.objective_value),
        ("SciPy/HiGHS", reference.x, -reference.fun),
    ):
        print(f"{name:<18} {values[0]:>8.2f} {values[1]:>8.2f} {objective:>8.2f}")


if __name__ == "__main__":
    main()
