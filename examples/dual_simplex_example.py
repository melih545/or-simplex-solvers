"""Basic dual simplex example."""

from simplex_codes import dual_simplex


def main() -> None:
    result = dual_simplex(
        c=[-2, -3, 0, 0],
        A=[
            [-1, -2, 1, 0],
            [-1, -2, 0, 1],
        ],
        b=[-4, -3],
    )
    print("solution:", result.solution.tolist())
    print("objective_value:", result.objective_value)
    print("iterations:", result.iterations)


if __name__ == "__main__":
    main()

