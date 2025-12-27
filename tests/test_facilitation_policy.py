#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for facilitation policy (θ > 1 control)."""

import sys
sys.path.insert(0, 'src')

from persiste.core import (
    StateSpace,
    TransitionGraph,
    Baseline,
    ConstraintModel,
)


def test_facilitation_allowed_by_default():
    """Test that facilitation (θ > 1) is allowed by default."""
    print("Testing facilitation allowed by default...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    # Default: allow_facilitation=True
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition"
    )
    
    model.set_parameters(theta={
        (0, 1): 2.0,  # Facilitation
        (1, 2): 0.5,  # Suppression
    })
    
    # Facilitation should work
    assert model.effective_rate(0, 1) == 2.0 * 2.0  # θ = 2.0 applied
    assert model.effective_rate(1, 2) == 0.5 * 2.0  # θ = 0.5 applied
    
    print("  ✓ Facilitation (θ > 1) allowed by default")


def test_facilitation_disabled_pure_constraint():
    """Test pure constraint model (allow_facilitation=False)."""
    print("Testing pure constraint model (no facilitation)...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    # Pure constraint model: no facilitation
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition",
        allow_facilitation=False  # ← Pure constraint
    )
    
    model.set_parameters(theta={
        (0, 1): 2.0,  # Attempted facilitation (should be clipped to 1.0)
        (1, 2): 0.5,  # Suppression (should work normally)
        (2, 0): 1.0,  # Neutral (should work normally)
    })
    
    # Facilitation clipped to 1.0 (neutral)
    assert model.effective_rate(0, 1) == 1.0 * 2.0  # θ clipped to 1.0
    
    # Suppression and neutral work normally
    assert model.effective_rate(1, 2) == 0.5 * 2.0  # θ = 0.5 applied
    assert model.effective_rate(2, 0) == 1.0 * 2.0  # θ = 1.0 applied
    
    print("  ✓ Facilitation (θ > 1) clipped to 1.0 when disabled")
    print("  ✓ Suppression (θ < 1) works normally")


def test_regime_distinction():
    """Test regime distinction: pure constraint vs selection-enabled."""
    print("Testing regime distinction...")
    states = StateSpace.from_list(['A', 'B'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=1.0)
    
    # Regime 1: Pure constraint (early life, assembly theory)
    constraint_only = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        allow_facilitation=False
    )
    constraint_only.set_parameters(theta={(0, 1): 5.0})
    
    # Regime 2: Selection-enabled (phylogenetics, HyPhy)
    selection_enabled = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        allow_facilitation=True
    )
    selection_enabled.set_parameters(theta={(0, 1): 5.0})
    
    # Same θ, different regimes, different effective rates
    rate_constraint_only = constraint_only.effective_rate(0, 1)
    rate_selection = selection_enabled.effective_rate(0, 1)
    
    assert rate_constraint_only == 1.0  # Clipped to neutral
    assert rate_selection == 5.0  # Facilitation allowed
    
    print("  ✓ Pure constraint regime: θ clipped to [0, 1]")
    print("  ✓ Selection-enabled regime: θ unrestricted")
    print("  ✓ HyPhy lives entirely in selection-enabled regime")


def test_per_state_facilitation_policy():
    """Test facilitation policy with per-state constraint structure."""
    print("Testing facilitation policy with per-state constraints...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_state",
        allow_facilitation=False
    )
    
    model.set_parameters(theta={
        0: 3.0,  # Attempted facilitation (should be clipped)
        1: 0.5,  # Suppression (should work)
        2: 1.0,  # Neutral (should work)
    })
    
    # State 0: facilitation clipped
    assert model.effective_rate(0, 1) == 1.0 * 2.0
    assert model.effective_rate(0, 2) == 1.0 * 2.0
    
    # State 1: suppression works
    assert model.effective_rate(1, 0) == 0.5 * 2.0
    assert model.effective_rate(1, 2) == 0.5 * 2.0
    
    print("  ✓ Per-state facilitation policy enforced")


def main():
    """Run all facilitation policy tests."""
    print("\n" + "="*60)
    print("PERSISTE Facilitation Policy Tests")
    print("="*60 + "\n")
    
    tests = [
        test_facilitation_allowed_by_default,
        test_facilitation_disabled_pure_constraint,
        test_regime_distinction,
        test_per_state_facilitation_policy,
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
