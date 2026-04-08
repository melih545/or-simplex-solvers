"""Basic Big-M example."""

from simplex_codes import big_m_method


def main() -> None:
    result = big_m_method(
        c=[-2, -3, 0, 0, 0, 0],
        A=[
            [0.5, 0.25, 1, 0, 0, 0],
            [1, 3, 0, -1, 1, 0],
            [1, 1, 0, 0, 0, 1],
        ],
        b=[4, 20, 10],
        sense="min",
        artificial_count=2,
    )
    print("solution:", result.solution.tolist())
    print("objective_value:", result.objective_value)
    print("iterations:", result.iterations)


if __name__ == "__main__":
    main()

