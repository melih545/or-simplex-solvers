"""Basic revised simplex example."""

from simplex_codes import revised_simplex


def main() -> None:
    result = revised_simplex(
        c=[4, 3, 0, 0],
        A=[
            [1, 1, 1, 0],
            [2, 1, 0, 1],
        ],
        b=[2, 3],
        basic_indices=[2, 3],
        nonbasic_indices=[0, 1],
    )
    print("solution:", result.solution.tolist())
    print("objective_value:", result.objective_value)
    print("iterations:", result.iterations)


if __name__ == "__main__":
    main()

