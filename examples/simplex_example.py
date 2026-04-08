"""Basic primal simplex example."""

from simplex_codes import simplex


def main() -> None:
    result = simplex(
        c=[-8, -5, 0, 0],
        A=[
            [1, 1, 1, 0],
            [9, 5, 0, 1],
        ],
        b=[6, 45],
        sense="max",
    )
    print("solution:", result.solution.tolist())
    print("objective_value:", result.objective_value)
    print("iterations:", result.iterations)


if __name__ == "__main__":
    main()

