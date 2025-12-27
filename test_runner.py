#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple test runner to verify core functionality without pytest."""

import sys
sys.path.insert(0, 'src')

from persiste.core import (
    StateSpace,
    TransitionGraph,
    PoissonObservationModel,
    Baseline,
    ObservedTransitions,
    ConstraintModel,
)

def test_statespace():
    """Test StateSpace basic functionality."""
    print("Testing StateSpace...")
    states = StateSpace.from_list(['α', 'β', 'γ'])
    assert len(states) == 3
    assert states.dimension == 3
    assert states[0] == 'α'
    print("  ✓ StateSpace.from_list works")
    
    states2 = StateSpace.from_types(['A', 'B'])
    assert len(states2) == 2
    print("  ✓ StateSpace.from_types works")

def test_transition_graph():
    """Test TransitionGraph structure-only design."""
    print("Testing TransitionGraph...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    
    # Explicit edges
    edges = [(0, 1), (1, 2), (2, 0)]
    graph = TransitionGraph.from_edges(states, edges)
    assert graph.allows(0, 1)
    assert not graph.allows(0, 2)
    print("  ✓ Explicit edges work")
    
    # Implicit adjacency
    graph2 = TransitionGraph.from_adjacency(
        states,
        lambda i, j: (j == (i + 1) % 3)
    )
    assert graph2.allows(0, 1)
    assert graph2.allows(2, 0)
    assert not graph2.allows(0, 2)
    print("  ✓ Implicit adjacency works")
    
    # Complete graph
    graph3 = TransitionGraph.complete(states)
    assert graph3.allows(0, 1)
    assert graph3.allows(0, 2)
    assert not graph3.allows(0, 0)
    print("  ✓ Complete graph works")

def test_observed_transitions():
    """Test ObservedTransitions data class."""
    print("Testing ObservedTransitions...")
    counts = {(0, 1): 5, (1, 2): 3, (2, 0): 2}
    data = ObservedTransitions(counts=counts, exposure=1.0)
    
    assert data.total_transitions() == 10
    assert data.counts[(0, 1)] == 5
    assert data.exposure == 1.0
    print("  ✓ ObservedTransitions creation works")

def test_baseline():
    """Test Baseline rates-only design."""
    print("Testing Baseline...")
    baseline = Baseline.uniform(rate=2.0)
    assert baseline.get_rate(0, 1) == 2.0
    assert baseline.get_rate(1, 0) == 2.0
    print("  ✓ Baseline.uniform works")

def test_observation_model():
    """Test ObservationModel with clean data flow."""
    print("Testing ObservationModel (data → baseline → observation)...")
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
    print(f"  ✓ Poisson log-likelihood computed: {ll:.4f}")

def test_baseline_empirical():
    """Test empirical baseline (with philosophical warning)."""
    print("Testing Baseline.empirical (⚠️  collapses layers)...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    
    data = ObservedTransitions(
        counts={(0, 1): 10, (1, 2): 5},
        exposure=2.0
    )
    
    # Empirical baseline: λ_ij = count / exposure
    baseline = Baseline.empirical(data, graph)
    
    assert baseline.get_rate(0, 1) == 10 / 2.0
    assert baseline.get_rate(1, 2) == 5 / 2.0
    assert baseline.get_rate(2, 0) == 0.0
    print("  ✓ Empirical baseline works (but philosophically problematic)")

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PERSISTE Core Functionality Tests")
    print("="*60 + "\n")
    
    tests = [
        test_statespace,
        test_transition_graph,
        test_observed_transitions,
        test_baseline,
        test_observation_model,
        test_baseline_empirical,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
