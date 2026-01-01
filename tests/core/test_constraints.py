from unittest.mock import MagicMock

import numpy as np
import pytest

from persiste.core.baseline import Baseline
from persiste.core.constraints import ConstraintModel
from persiste.core.states import StateSpace
from persiste.core.transitions import TransitionGraph


@pytest.fixture()
def state_graph():
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    return states, graph, baseline


def test_effective_rate_per_transition_and_facilitation_policy(state_graph):
    states, graph, baseline = state_graph
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition",
        allow_facilitation=False,
    )
    model.set_parameters(theta={(0, 1): 0.25, (1, 2): 5.0})

    suppressed = model.effective_rate(0, 1)
    assert suppressed == pytest.approx(0.5)

    # θ>1 should be clipped to 1 when facilitation disabled
    clipped = model.effective_rate(1, 2)
    assert clipped == pytest.approx(2.0)


def test_per_state_theta_and_num_parameters(state_graph):
    states, graph, baseline = state_graph
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_state",
    )
    model.set_parameters(theta={0: 0.1, 1: 1.0})

    assert model.get_theta(0, 2) == 0.1
    assert model.num_free_parameters() == 2


def test_hierarchical_pack_and_unpack():
    states = StateSpace.from_list(['A', 'B', 'C', 'D'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=1.5)

    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="hierarchical",
    )
    model.set_parameters(
        groups={(0, 1): "core", (1, 0): "core", (2, 3): "accessory"},
        theta={'core': 0.8, 'accessory': 0.3, (0, 2): 1.2},
    )

    packed = model.pack()
    unpacked = model.unpack(packed)
    assert unpacked["theta"] == model.parameters["theta"]
    assert unpacked["groups"] == model.parameters["groups"]


def test_sparse_structure_defaults_to_small_eps(state_graph):
    states, graph, baseline = state_graph
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="sparse",
    )
    model.set_parameters(theta={(0, 1): 1.0})

    assert model.effective_rate(0, 1) == pytest.approx(2.0)
    # unspecified transitions fall back to epsilon (≈0)
    assert model.effective_rate(1, 0) < 1e-5


def test_get_constrained_baseline_calls_effective_rate(state_graph):
    states, graph, baseline = state_graph
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition",
    )
    model.set_parameters(theta={(0, 1): 0.5})

    spy = MagicMock(side_effect=model.effective_rate)
    model.effective_rate = spy  # type: ignore[assignment]

    constrained = model.get_constrained_baseline()
    rate = constrained.get_rate(0, 1)

    assert rate == pytest.approx(1.0)
    assert spy.called
