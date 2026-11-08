"""Differential tests against an independent LP solver."""

import numpy as np
import pytest

from simplex_codes import solve

linprog = pytest.importorskip("scipy.optimize").linprog


def assert_matches_highs(c, A, b, relations, bounds, sense="min", scales=None):
    A, b, c = (np.asarray(values, dtype=float) for values in (A, b, c))
    relations = np.asarray(relations)
    equality = relations == "="
    direction = np.where(relations == ">=", -1, 1)
    sign = -1 if sense == "max" else 1
    reference = linprog(
        sign * c, A_ub=(A * direction[:, None])[~equality], b_ub=(b * direction)[~equality],
        A_eq=A[equality], b_eq=b[equality], bounds=bounds, method="highs",
        options={"presolve": False},
    )
    assert reference.status in (0, 2, 3), reference.message
    scales = np.ones(b.size) if scales is None else np.asarray(scales)
    args = (c, A * scales[:, None], b * scales, relations)
    if reference.status:
        expected = "infeasible" if reference.status == 2 else "unbounded"
        with pytest.raises(ValueError, match=expected):
            solve(*args, bounds=bounds, sense=sense)
        return

    result = solve(*args, bounds=bounds, sense=sense)
    assert result.objective_value == pytest.approx(sign * reference.fun, abs=1e-7)
    residual = A @ result.solution - b
    np.testing.assert_allclose(residual[equality], 0, atol=1e-7)
    assert np.all((residual * direction)[~equality] <= 1e-7)
    for value, (lower, upper) in zip(result.solution, bounds):
        assert lower is None or value >= lower - 1e-7
        assert upper is None or value <= upper + 1e-7


@pytest.mark.parametrize(
    ("c", "A", "b", "relations", "bounds", "sense", "scales"),
    [
        ([1, 1], [[1, 1], [2, 2]], [2, 4], ["<=", "<="], [(0, None)] * 2, "max", [1e-10, 1e10]),
        ([0, 1], [[1, 0], [2, 0], [0, 1]], [0, 0, 2], ["=", "=", "<="],
         [(None, None), (0, None)], "max", [1e-12, 1e8, 1]),
        ([1, 2], [[1, 1], [1, 0]], [-2, -5], ["=", ">="],
         [(None, None), (0, 2)], "min", [1e8, 1e-10]),
        ([1], [[1], [1]], [2, 1], [">=", "<="], [(0, None)], "min", [1e-10, 1e9]),
        ([1], [[1]], [1], [">="], [(0, None)], "max", [1e-10]),
    ],
    ids=["alternate-optima", "degenerate-basis", "free-variable", "infeasible", "unbounded"],
)
def test_highs_reference_cases(c, A, b, relations, bounds, sense, scales):
    assert_matches_highs(c, A, b, relations, bounds, sense, scales)


@pytest.mark.parametrize("seed", range(20))
def test_highs_reference_generated_capacity_models(seed):
    rng = np.random.default_rng(seed)
    A = rng.integers(1, 10, size=(5, 4))
    b = rng.integers(5, 35, size=5)
    c = rng.integers(-5, 10, size=4)
    scales = 10.0 ** rng.integers(-12, 13, size=5)
    assert_matches_highs(c, A, b, ["<="] * 5, [(0, None)] * 4, "max", scales)
