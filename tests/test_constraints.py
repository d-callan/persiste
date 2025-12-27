#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for constraint models and effective rates.

Test organization:
- Unit tests: verify constraint semantics (per-transition, per-state, etc.)
- Integration tests: verify likelihood comparisons and theory validation
- Philosophy tests: verify constraint-first commitments (suppression, persistence)
"""

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


def test_per_transition_constraint():
    """Test per-transition constraint: λ*_ij = θ_ij × λ_ij."""
    print("Testing per-transition constraint...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    # Create constraint model
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition"
    )
    
    # Set constraint parameters
    model.set_parameters(theta={
        (0, 1): 0.5,  # Suppressed (θ < 1)
        (1, 2): 1.0,  # Neutral (θ = 1)
        (2, 0): 1.5,  # Facilitated (θ > 1)
    })
    
    # Check effective rates
    assert model.effective_rate(0, 1) == 0.5 * 2.0  # Suppressed
    assert model.effective_rate(1, 2) == 1.0 * 2.0  # Neutral
    assert model.effective_rate(2, 0) == 1.5 * 2.0  # Facilitated
    assert model.effective_rate(0, 2) == 1.0 * 2.0  # Default (no constraint)
    
    print("  ✓ Per-transition constraint works")


def test_per_state_constraint():
    """Test per-state constraint: λ*_ij = θ_i × λ_ij (canalization)."""
    print("Testing per-state constraint (canalization)...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_state"
    )
    
    # State 0 is highly conserved (resists all changes)
    model.set_parameters(theta={
        0: 0.1,  # State A resists change
        1: 1.0,  # State B neutral
        2: 0.5,  # State C partially conserved
    })
    
    # All transitions from state 0 are suppressed
    assert model.effective_rate(0, 1) == 0.1 * 2.0
    assert model.effective_rate(0, 2) == 0.1 * 2.0
    
    # Transitions from state 1 are neutral
    assert model.effective_rate(1, 0) == 1.0 * 2.0
    assert model.effective_rate(1, 2) == 1.0 * 2.0
    
    print("  ✓ Per-state constraint (canalization) works")


def test_sparse_constraint():
    """Test sparse constraint: θ_ij ∈ {0, 1} with most suppressed."""
    print("Testing sparse constraint (life refuses to vary)...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="sparse"
    )
    
    # Only allow a few specific transitions
    model.set_parameters(theta={
        (0, 1): 1.0,  # Allowed
        (1, 2): 1.0,  # Allowed
        # Everything else defaults to 0 (suppressed)
    })
    
    # Allowed transitions
    assert model.effective_rate(0, 1) == 1.0 * 2.0
    assert model.effective_rate(1, 2) == 1.0 * 2.0
    
    # Suppressed transitions (default to ε ~ 1e-6, effectively zero)
    assert model.effective_rate(2, 0) < 1e-5
    assert model.effective_rate(0, 2) < 1e-5
    assert model.effective_rate(1, 0) < 1e-5
    
    print("  ✓ Sparse constraint works")


def test_hierarchical_constraint():
    """Test hierarchical constraint: shared strength across groups."""
    print("Testing hierarchical constraint (shared strength)...")
    states = StateSpace.from_list(['A', 'B', 'C', 'D'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="hierarchical"
    )
    
    # Group transitions by functional class
    model.set_parameters(
        groups={
            (0, 1): "conservative",
            (1, 0): "conservative",
            (2, 3): "radical",
            (3, 2): "radical",
        },
        theta={
            "conservative": 0.8,  # Slightly suppressed
            "radical": 0.2,       # Strongly suppressed
            (0, 2): 1.0,          # Specific override
        }
    )
    
    # Conservative transitions
    assert model.effective_rate(0, 1) == 0.8 * 2.0
    assert model.effective_rate(1, 0) == 0.8 * 2.0
    
    # Radical transitions
    assert model.effective_rate(2, 3) == 0.2 * 2.0
    assert model.effective_rate(3, 2) == 0.2 * 2.0
    
    # Specific override
    assert model.effective_rate(0, 2) == 1.0 * 2.0
    
    print("  ✓ Hierarchical constraint works")


# ============================================================
# Integration / Theory Tests
# ============================================================

def test_constraint_vs_baseline_likelihood():
    """[INTEGRATION] Test likelihood comparison: baseline vs constrained."""
    print("[INTEGRATION] Testing constraint detection via likelihood comparison...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    # Observed data: (1,2) transition much more common than expected
    # Under uniform baseline, we'd expect ~2 of each, but we see 10 of (1,2)
    data = ObservedTransitions(
        counts={(0, 1): 2, (1, 2): 10, (2, 0): 2},
        exposure=1.0
    )
    
    # Baseline likelihood (no constraint)
    obs_model = PoissonObservationModel(graph)
    ll_baseline = obs_model.log_likelihood(data, baseline, graph)
    
    # Constrained model: facilitate (1,2) to match observed pattern
    constraint = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition"
    )
    constraint.set_parameters(theta={
        (0, 1): 1.0,
        (1, 2): 5.0,  # Facilitate this transition (θ > 1)
        (2, 0): 1.0,
    })
    
    # Constrained likelihood
    constrained_baseline = constraint.get_constrained_baseline()
    ll_constrained = obs_model.log_likelihood(data, constrained_baseline, graph)
    
    # Constraint should improve likelihood (better match to observed pattern)
    Δ = ll_constrained - ll_baseline
    print(f"  Δ = logL(constrained) - logL(baseline) = {Δ:.4f}")
    print(f"  ll_baseline = {ll_baseline:.4f}, ll_constrained = {ll_constrained:.4f}")
    assert Δ > 0, "Constrained model should fit better when it matches observed pattern"
    
    print("  ✓ Constraint improves likelihood")


# ============================================================
# Philosophy Tests (Constraint-First Commitments)
# ============================================================

def test_neutral_constraint_equals_baseline():
    """[INVARIANT] Test that θ=1 everywhere → effective_rate == baseline.
    
    This is the sanity axiom: neutral constraint should not modify rates.
    """
    print("Testing neutral constraint invariant (θ=1 → no modification)...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.5)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition"
    )
    
    # Set all θ = 1 (neutral)
    model.set_parameters(theta={
        (0, 1): 1.0,
        (0, 2): 1.0,
        (1, 0): 1.0,
        (1, 2): 1.0,
        (2, 0): 1.0,
        (2, 1): 1.0,
    })
    
    # Verify: effective_rate == baseline for all transitions
    for i in range(3):
        for j in range(3):
            if i != j:
                λ_baseline = baseline.get_rate(i, j)
                λ_effective = model.effective_rate(i, j)
                assert λ_effective == λ_baseline, \
                    f"Neutral constraint failed: λ*_{i}{j} = {λ_effective} != λ_{i}{j} = {λ_baseline}"
    
    print("  ✓ Neutral constraint (θ=1) preserves baseline rates")
    print("  ✓ Sanity axiom verified")


def test_constraint_respects_graph_topology():
    """Test that constraint cannot create transitions forbidden by graph."""
    print("Testing constraint respects graph topology...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    
    # Restricted graph: only (0,1) and (1,2) allowed
    edges = [(0, 1), (1, 2)]
    graph = TransitionGraph.from_edges(states, edges)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition"
    )
    
    # Try to set constraint on forbidden transition
    model.set_parameters(theta={
        (0, 1): 0.5,  # Allowed
        (1, 2): 0.8,  # Allowed
        (2, 0): 10.0, # Forbidden by graph (should be ignored)
        (0, 2): 10.0, # Forbidden by graph (should be ignored)
    })
    
    # Allowed transitions should have constrained rates
    assert model.effective_rate(0, 1) == 0.5 * 2.0
    assert model.effective_rate(1, 2) == 0.8 * 2.0
    
    # Forbidden transitions should return 0.0, regardless of θ
    assert model.effective_rate(2, 0) < 1e-6
    assert model.effective_rate(0, 2) < 1e-6
    assert model.effective_rate(1, 0) < 1e-6
    assert model.effective_rate(2, 1) < 1e-6
    
    print("  ✓ Constraint respects graph topology")
    print("  ✓ Forbidden transitions return 0.0 regardless of θ")


def test_constraint_only_suppresses():
    """[PHILOSOPHY] Test model where θ ≤ 1 always (constraint-first).
    
    This reinforces the constraint-first philosophy:
    - Constraint = systematic suppression
    - Not facilitation or positive selection
    - Aligns with early-life and pre-Darwinian regimes
    """
    print("[PHILOSOPHY] Testing constraint-only suppression (θ ≤ 1)...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition"
    )
    
    # All constraint parameters ≤ 1 (suppression or neutral, no facilitation)
    model.set_parameters(theta={
        (0, 1): 0.5,  # Suppressed
        (1, 2): 0.1,  # Strongly suppressed
        (2, 0): 1.0,  # Neutral (no constraint)
    })
    
    # Verify all effective rates ≤ baseline rates
    for i in range(3):
        for j in range(3):
            if i != j:
                λ_ij = baseline.get_rate(i, j)
                λ_star = model.effective_rate(i, j)
                assert λ_star <= λ_ij, f"Constraint should only suppress: λ*_{i}{j} = {λ_star} > λ_{i}{j} = {λ_ij}"
    
    print("  ✓ All constraints suppress or are neutral (θ ≤ 1)")
    print("  ✓ Constraint-first philosophy validated")


def test_constraint_persists_across_baselines():
    """[PHILOSOPHY] Test that constraint persists under baseline rescaling.
    
    This tests the central idea:
    - Constraint survives regime change
    - Selection = persistent deviation from production
    - Relative suppression is regime-invariant
    
    If baseline rates scale by factor k, constrained rates should scale by same k.
    """
    print("[PHILOSOPHY] Testing constraint persistence across baseline rescaling...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    
    # Two baselines: one 10x faster than the other
    baseline_slow = Baseline.uniform(rate=1.0)
    baseline_fast = Baseline.uniform(rate=10.0)
    
    # Same constraint structure
    graph = TransitionGraph.complete(states)
    constraint_slow = ConstraintModel(
        states=states,
        baseline=baseline_slow,
        graph=graph,
        constraint_structure="per_transition"
    )
    constraint_slow.set_parameters(theta={
        (0, 1): 0.5,
        (1, 2): 0.2,
        (2, 0): 0.8,
    })
    
    constraint_fast = ConstraintModel(
        states=states,
        baseline=baseline_fast,
        graph=graph,
        constraint_structure="per_transition"
    )
    constraint_fast.set_parameters(theta={
        (0, 1): 0.5,
        (1, 2): 0.2,
        (2, 0): 0.8,
    })
    
    # Verify: constrained rates scale proportionally
    for i in range(3):
        for j in range(3):
            if i != j:
                λ_slow = constraint_slow.effective_rate(i, j)
                λ_fast = constraint_fast.effective_rate(i, j)
                
                # Fast should be exactly 10x slow
                ratio = λ_fast / λ_slow if λ_slow > 0 else float('inf')
                assert abs(ratio - 10.0) < 1e-6, f"Constraint should persist: ratio = {ratio}, expected 10.0"
    
    print("  ✓ Constraint persists under 10x baseline rescaling")
    print("  ✓ Relative suppression is regime-invariant")
    print("  ✓ Selection survives regime change")


def main():
    """Run all constraint tests."""
    print("\n" + "="*60)
    print("PERSISTE Constraint Model Tests")
    print("="*60 + "\n")
    
    # Unit tests
    unit_tests = [
        test_per_transition_constraint,
        test_per_state_constraint,
        test_sparse_constraint,
        test_hierarchical_constraint,
    ]
    
    # Integration tests
    integration_tests = [
        test_constraint_vs_baseline_likelihood,
    ]
    
    # Philosophy tests
    philosophy_tests = [
        test_neutral_constraint_equals_baseline,
        test_constraint_respects_graph_topology,
        test_constraint_only_suppresses,
        test_constraint_persists_across_baselines,
    ]
    
    tests = unit_tests + integration_tests + philosophy_tests
    
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
