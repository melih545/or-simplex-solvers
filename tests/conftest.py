"""Shared fixtures for solver tests."""

import pytest


@pytest.fixture
def simplex_problem():
    return {
        "c": [-8, -5, 0, 0],
        "A": [
            [1, 1, 1, 0],
            [9, 5, 0, 1],
        ],
        "b": [6, 45],
        "sense": "max",
    }


@pytest.fixture
def dual_simplex_problem():
    return {
        "c": [-2, -3, 0, 0],
        "A": [
            [-1, -2, 1, 0],
            [-1, -2, 0, 1],
        ],
        "b": [-4, -3],
    }


@pytest.fixture
def revised_simplex_problem():
    return {
        "c": [4, 3, 0, 0],
        "A": [
            [1, 1, 1, 0],
            [2, 1, 0, 1],
        ],
        "b": [2, 3],
        "basic_indices": [2, 3],
        "nonbasic_indices": [0, 1],
    }


@pytest.fixture
def big_m_problem():
    return {
        "c": [-2, -3, 0, 0, 0, 0],
        "A": [
            [0.5, 0.25, 1, 0, 0, 0],
            [1, 3, 0, -1, 1, 0],
            [1, 1, 0, 0, 0, 1],
        ],
        "b": [4, 20, 10],
        "sense": "min",
        "artificial_count": 2,
    }

