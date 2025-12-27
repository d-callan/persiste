#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for MLE inference and LRT.

Demonstrates minimal viable inference:
- Parameter vectorization (pack/unpack)
- MLE optimization
- Likelihood ratio test
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

try:
    from scipy import optimize, stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available, skipping MLE tests")
    sys.exit(0)

from persiste.core import (
    StateSpace,
    TransitionGraph,
    PoissonObservationModel,
    Baseline,
    ObservedTransitions,
    ConstraintModel,
    ConstraintInference,
)


def test_parameter_vectorization():
    """Test pack/unpack for parameter vectorization."""
    print("Testing parameter vectorization (pack/unpack)...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition"
    )
    
    # Set some parameters
    model.set_parameters(theta={
        (0, 1): 0.5,
        (0, 2): 0.8,
        (1, 0): 1.2,
        (1, 2): 0.3,
        (2, 0): 1.5,
        (2, 1): 0.9,
    })
    
    # Pack to vector
    vec = model.pack()
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 6  # 6 transitions
    
    # Unpack back to dict
    params = model.unpack(vec)
    assert params["theta"][(0, 1)] == 0.5
    assert params["theta"][(2, 0)] == 1.5
    
    print(f"  ✓ Packed {len(vec)} parameters to vector")
    print(f"  ✓ Unpacked vector back to dict")
    print("  ✓ Bridge between math and optimization works")


def test_initial_parameters():
    """Test initial parameter generation for optimization."""
    print("Testing initial parameters...")
    states = StateSpace.from_list(['A', 'B'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
    )
    
    # Need to set structure first (which transitions exist)
    model.set_parameters(theta={(0, 1): 1.0, (1, 0): 1.0})
    
    # Get initial parameters
    theta0 = model.initial_parameters()
    assert isinstance(theta0, np.ndarray)
    assert len(theta0) == 2
    assert np.all(theta0 == 1.0)  # Neutral starting point
    
    print(f"  ✓ Initial parameters: {theta0}")
    print("  ✓ Neutral starting point (θ = 1 everywhere)")


def test_mle_simple():
    """Test minimal viable MLE on simple data."""
    print("Testing minimal viable MLE...")
    states = StateSpace.from_list(['A', 'B'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    # Create model
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
    )
    
    # Initialize with transitions we want to fit
    model.set_parameters(theta={(0, 1): 1.0, (1, 0): 1.0})
    
    # Observed data: (0,1) much more common than (1,0)
    data = ObservedTransitions(
        counts={(0, 1): 20, (1, 0): 2},
        exposure=1.0
    )
    
    # Fit via MLE
    obs_model = PoissonObservationModel(graph)
    engine = ConstraintInference(model, obs_model)
    
    result = engine.fit(data, method="MLE")
    
    # Check result structure
    assert result.method == "MLE"
    assert result.log_likelihood < 0
    assert result.aic is not None
    assert result.bic is not None
    
    # Fitted parameters should reflect data
    # (0,1) should have θ > 1 (facilitated)
    # (1,0) should have θ < 1 (suppressed)
    θ_01 = result.parameters["theta"][(0, 1)]
    θ_10 = result.parameters["theta"][(1, 0)]
    
    print(f"  ✓ MLE converged: {result.metadata['success']}")
    print(f"  ✓ Log-likelihood: {result.log_likelihood:.4f}")
    print(f"  ✓ AIC: {result.aic:.4f}, BIC: {result.bic:.4f}")
    print(f"  ✓ Fitted θ(0→1) = {θ_01:.4f} (facilitated)")
    print(f"  ✓ Fitted θ(1→0) = {θ_10:.4f} (suppressed)")
    
    assert θ_01 > θ_10, "MLE should detect asymmetry in data"


def test_lrt_trivial():
    """Test that LRT becomes trivial with ConstraintResult."""
    print("Testing likelihood ratio test...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    # Null model: simpler (only 2 transitions)
    null_model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
    )
    null_model.set_parameters(theta={(0, 1): 1.0, (1, 2): 1.0})
    
    # Alternative model: more complex (all 6 transitions)
    alt_model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
    )
    alt_model.set_parameters(theta={
        (0, 1): 1.0, (0, 2): 1.0,
        (1, 0): 1.0, (1, 2): 1.0,
        (2, 0): 1.0, (2, 1): 1.0,
    })
    
    # Data with clear pattern
    data = ObservedTransitions(
        counts={(0, 1): 20, (1, 2): 15, (2, 0): 2, (0, 2): 1, (1, 0): 1, (2, 1): 1},
        exposure=1.0
    )
    
    obs_model = PoissonObservationModel(graph)
    
    # Fit null (simpler model)
    null_engine = ConstraintInference(null_model, obs_model)
    null_result = null_engine.fit(data, method="MLE")
    
    # Fit alternative (more complex model)
    alt_engine = ConstraintInference(alt_model, obs_model)
    alt_result = alt_engine.fit(data, method="MLE")
    
    # Likelihood ratio test
    test_result = alt_engine.test(data, null_result, alt_result, method="LRT")
    
    print(f"  ✓ LRT statistic: {test_result.statistic:.4f}")
    print(f"  ✓ Degrees of freedom: {test_result.metadata['df']}")
    print(f"  ✓ p-value: {test_result.pvalue:.4e}")
    print(f"  ✓ k_null={test_result.metadata['k_null']}, k_alt={test_result.metadata['k_alt']}")
    
    # Alternative should fit better (more parameters)
    assert test_result.statistic > 0, "Alternative should fit better"
    assert test_result.metadata['df'] == 4, "df should be 6 - 2 = 4"
    
    print("  ✓ LRT correctly compares models")
    print("  ✓ No model logic in test - just ConstraintResult comparison")


def test_facilitation_policy_in_mle():
    """Test that facilitation policy affects MLE bounds."""
    print("Testing facilitation policy in MLE...")
    states = StateSpace.from_list(['A', 'B'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=1.0)
    
    # Pure constraint model (no facilitation)
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        allow_facilitation=False,  # θ ∈ [0, 1]
    )
    model.set_parameters(theta={(0, 1): 1.0, (1, 0): 1.0})
    
    # Data that would suggest facilitation
    data = ObservedTransitions(
        counts={(0, 1): 20, (1, 0): 2},
        exposure=1.0
    )
    
    obs_model = PoissonObservationModel(graph)
    engine = ConstraintInference(model, obs_model)
    
    result = engine.fit(data, method="MLE")
    
    # All θ should be ≤ 1 (clipped by bounds)
    for θ in result.parameters["theta"].values():
        assert θ <= 1.0, f"Pure constraint model should have θ ≤ 1, got {θ}"
    
    print("  ✓ Pure constraint model: all θ ≤ 1")
    print("  ✓ Facilitation policy enforced in optimization bounds")


def main():
    """Run all MLE inference tests."""
    print("\n" + "="*60)
    print("PERSISTE MLE Inference Tests")
    print("="*60 + "\n")
    
    tests = [
        test_parameter_vectorization,
        test_initial_parameters,
        test_mle_simple,
        test_lrt_trivial,
        test_facilitation_policy_in_mle,
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
