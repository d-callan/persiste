import numpy as np
import pytest

from persiste.core.baseline import Baseline
from persiste.core.data import ObservedTransitions


class GraphStub:
    """Minimal transition-graph stub for empirical baseline tests."""

    def __init__(self, allowed):
        self.allowed = set(allowed)

    def allows(self, i: int, j: int) -> bool:
        return (i, j) in self.allowed


def test_from_matrix_returns_numpy_rates():
    matrix = np.array(
        [
            [0.0, 0.5, 0.2],
            [0.1, 0.0, 0.3],
            [0.4, 0.6, 0.0],
        ]
    )

    baseline = Baseline.from_matrix(matrix)

    assert pytest.approx(baseline.get_rate(0, 1)) == 0.5
    assert pytest.approx(baseline.get_rate(2, 0)) == 0.4

    rebuilt = baseline.to_matrix(3)
    np.testing.assert_allclose(rebuilt, matrix)


def test_uniform_baseline_populates_off_diagonal_rates():
    baseline = Baseline.uniform(rate=0.75)

    matrix = baseline.to_matrix(3)

    # Diagonal entries stay zero, off-diagonals share the uniform rate
    assert np.allclose(np.diag(matrix), 0.0)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert matrix[i, j] == pytest.approx(0.75)


def test_empirical_baseline_uses_counts_and_respects_graph():
    data = ObservedTransitions.from_dict(
        counts={
            (0, 1): 8,
            (1, 2): 4,
            (2, 0): 2,
        },
        exposure=4.0,
    )
    # Only allow two transitions; others should be ignored
    graph = GraphStub(allowed={(0, 1), (1, 2)})

    baseline = Baseline.empirical(data, graph=graph)

    assert baseline.get_rate(0, 1) == pytest.approx(2.0)
    assert baseline.get_rate(1, 2) == pytest.approx(1.0)
    # Disallowed transitions default to zero
    assert baseline.get_rate(2, 0) == 0.0


def test_to_matrix_calls_rate_fn_for_each_pair():
    calls = []

    def rate_fn(i: int, j: int) -> float:
        calls.append((i, j))
        return i + j

    baseline = Baseline(rate_fn=rate_fn)

    matrix = baseline.to_matrix(3)

    # Ensure rate_fn was consulted for every off-diagonal pair
    expected_calls = [(i, j) for i in range(3) for j in range(3) if i != j]
    assert sorted(calls) == expected_calls

    assert matrix[0, 2] == pytest.approx(2)
    assert matrix[2, 1] == pytest.approx(3)
    assert matrix[1, 1] == 0.0
