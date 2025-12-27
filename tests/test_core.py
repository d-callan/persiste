"""Tests for core abstractions."""

import pytest
from persiste.core import (
    StateSpace,
    TransitionGraph,
    PoissonObservationModel,
    Baseline,
    ObservedTransitions,
    ConstraintModel,
)


def test_statespace_from_list():
    """Test state space creation from list."""
    states = StateSpace.from_list(['α', 'β', 'γ'])
    assert len(states) == 3
    assert states.dimension == 3
    assert states[0] == 'α'


def test_statespace_from_types_simple():
    """Test basic state space creation from types."""
    states = StateSpace.from_types(['α', 'β', 'γ'])
    assert len(states) == 3
    assert states.dimension == 3
    assert states[0] == 'α'


def test_statespace_from_types_with_enumerator():
    """Test state space with lazy enumerator."""
    def simple_enumerator(types):
        for t in types:
            yield t
        yield f"{types[0]}+{types[1]}"
    
    states = StateSpace.from_types(['α', 'β'], enumerator=simple_enumerator)
    state_list = list(states.iter_states())
    assert len(state_list) == 3
    assert state_list == ['α', 'β', 'α+β']


def test_statespace_from_generator():
    """Test state space with generator."""
    def gen():
        yield 'A'
        yield 'B'
    
    states = StateSpace.from_generator(gen, dimension=2)
    assert len(states) == 2
    assert list(states.iter_states()) == ['A', 'B']


def test_statespace_iter_states():
    """Test state iteration."""
    states = StateSpace.from_list(['x', 'y', 'z'])
    assert list(states.iter_states()) == ['x', 'y', 'z']


def test_statespace_no_enumeration():
    """Test that state space without enumeration raises error."""
    states = StateSpace()
    with pytest.raises(NotImplementedError):
        list(states.iter_states())


def test_statespace_len_unknown_dimension():
    """Test that len() raises for unknown dimension."""
    states = StateSpace(generator=lambda: iter(['a', 'b']))
    with pytest.raises(TypeError):
        len(states)


def test_statespace_indexing_generator():
    """Test that indexing raises for generator-based space."""
    states = StateSpace(generator=lambda: iter(['a', 'b']))
    with pytest.raises(TypeError):
        _ = states[0]


def test_baseline_uniform():
    """Test uniform baseline creation."""
    baseline = Baseline.uniform(rate=2.0)
    assert baseline.get_rate(0, 1) == 2.0
    assert baseline.get_rate(1, 0) == 2.0


def test_constraint_model_creation():
    """Test constraint model instantiation."""
    states = StateSpace.from_types(['α', 'β'])
    baseline = Baseline.uniform()
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        constraint_structure='per_transition'
    )
    
    assert len(model.states) == 2
    assert model.constraint_structure == 'per_transition'


def test_constraint_model_invalid_structure():
    """Test that invalid constraint structure raises error."""
    states = StateSpace.from_types(['α', 'β'])
    baseline = Baseline.uniform()
    
    with pytest.raises(ValueError):
        ConstraintModel(
            states=states,
            baseline=baseline,
            constraint_structure='invalid'
        )


def test_transition_graph_explicit():
    """Test transition graph with explicit edges."""
    states = StateSpace.from_list(['A', 'B', 'C'])
    edges = [(0, 1), (1, 2), (2, 0)]
    graph = TransitionGraph.from_edges(states, edges)
    
    assert graph.allows(0, 1)
    assert graph.allows(1, 2)
    assert not graph.allows(0, 2)
    assert list(graph.iter_edges()) == edges


def test_transition_graph_implicit():
    """Test transition graph with adjacency function (implicit)."""
    states = StateSpace.from_list(['A', 'B', 'C'])
    # Only allow transitions to next state (circular)
    adjacency_fn = lambda i, j: (j == (i + 1) % 3)
    graph = TransitionGraph.from_adjacency(states, adjacency_fn)
    
    assert graph.allows(0, 1)
    assert graph.allows(1, 2)
    assert graph.allows(2, 0)
    assert not graph.allows(0, 2)
    
    edges = list(graph.iter_edges())
    assert len(edges) == 3
    assert (0, 1) in edges
    assert (1, 2) in edges
    assert (2, 0) in edges


def test_transition_graph_complete():
    """Test complete transition graph."""
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    
    # All transitions allowed except self-loops
    assert graph.allows(0, 1)
    assert graph.allows(0, 2)
    assert graph.allows(1, 0)
    assert not graph.allows(0, 0)
    
    edges = list(graph.iter_edges())
    assert len(edges) == 6  # 3 * 2 (no self-loops)


def test_transition_graph_no_structure():
    """Test that graph without structure raises error."""
    states = StateSpace.from_list(['A', 'B'])
    with pytest.raises(ValueError):
        TransitionGraph(states=states)


def test_observed_transitions_creation():
    """Test ObservedTransitions data class."""
    counts = {(0, 1): 5, (1, 2): 3, (2, 0): 2}
    data = ObservedTransitions(counts=counts, exposure=1.0)
    
    assert data.total_transitions() == 10
    assert data.counts[(0, 1)] == 5
    assert data.exposure == 1.0


def test_observed_transitions_validation():
    """Test that negative counts raise error."""
    counts = {(0, 1): -1}
    with pytest.raises(ValueError):
        ObservedTransitions(counts=counts)


def test_poisson_observation_model():
    """Test Poisson observation model with new data flow."""
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    model = PoissonObservationModel(graph, base_rate=2.0)
    
    # Create observed data
    data = ObservedTransitions(
        counts={(0, 1): 5, (1, 2): 3},
        exposure=1.0
    )
    
    # Compute likelihood: data → baseline → observation model
    ll = model.log_likelihood(data, baseline, graph)
    assert isinstance(ll, float)
    assert ll < 0  # Log-likelihood should be negative


def test_poisson_model_respects_graph_structure():
    """Test that Poisson observation model respects graph structure."""
    states = StateSpace.from_list(['A', 'B', 'C'])
    edges = [(0, 1), (1, 2)]  # Only two transitions allowed
    graph = TransitionGraph.from_edges(states, edges)
    baseline = Baseline.uniform(rate=1.0)
    model = PoissonObservationModel(graph, base_rate=1.0)
    
    # Allowed transitions
    assert model.rate(0, 1) == 1.0
    assert model.rate(1, 2) == 1.0
    
    # Disallowed transitions
    assert model.rate(2, 0) == 0.0
    assert model.rate(0, 2) == 0.0


def test_poisson_likelihood_with_exposure():
    """Test that exposure scaling works correctly."""
    states = StateSpace.from_list(['A', 'B'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=1.0)
    model = PoissonObservationModel(graph)
    
    # Same counts, different exposure
    data1 = ObservedTransitions(counts={(0, 1): 10}, exposure=1.0)
    data2 = ObservedTransitions(counts={(0, 1): 10}, exposure=2.0)
    
    ll1 = model.log_likelihood(data1, baseline, graph)
    ll2 = model.log_likelihood(data2, baseline, graph)
    
    # Different exposures should give different likelihoods
    assert ll1 != ll2


def test_baseline_empirical_warning():
    """Test that empirical baseline works but is philosophically marked."""
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    
    data = ObservedTransitions(
        counts={(0, 1): 10, (1, 2): 5},
        exposure=2.0
    )
    
    # Empirical baseline: λ_ij = count / exposure
    baseline = Baseline.empirical(data, graph)
    
    # Should give empirical rates
    assert baseline.get_rate(0, 1) == 10 / 2.0
    assert baseline.get_rate(1, 2) == 5 / 2.0
    assert baseline.get_rate(2, 0) == 0.0  # Not observed
