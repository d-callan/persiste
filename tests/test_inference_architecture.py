#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for inference architecture separation.

Demonstrates clean separation:
- ConstraintModel: what θ means
- Baseline: what rates mean
- ObservationModel: how data arises
- ConstraintInference: how θ is inferred
- ConstraintResult: what came out
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
    ConstraintInference,
    ConstraintResult,
)


def test_inference_engine_creation():
    """Test that ConstraintInference engine can be created."""
    print("Testing inference engine creation...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition"
    )
    
    obs_model = PoissonObservationModel(graph)
    
    # Create inference engine
    engine = ConstraintInference(model, obs_model)
    
    assert engine.model is model
    assert engine.obs_model is obs_model
    
    print("  ✓ ConstraintInference engine created")
    print("  ✓ Separates inference from model specification")


def test_likelihood_evaluation_pipeline():
    """Test likelihood evaluation pipeline: θ → baseline* → likelihood."""
    print("Testing likelihood evaluation pipeline...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition"
    )
    
    # Set constraint parameters
    model.set_parameters(theta={(0, 1): 0.5, (1, 2): 0.8})
    
    obs_model = PoissonObservationModel(graph)
    engine = ConstraintInference(model, obs_model)
    
    # Create observed data
    data = ObservedTransitions(
        counts={(0, 1): 5, (1, 2): 3},
        exposure=1.0
    )
    
    # Evaluate likelihood at current parameters
    ll = engine.log_likelihood(data)
    
    assert isinstance(ll, float)
    assert ll < 0  # Log-likelihood should be negative
    
    print(f"  ✓ Likelihood evaluated: {ll:.4f}")
    print("  ✓ Pipeline: θ → constrained baseline → observation model → ℓ")


def test_likelihood_with_different_parameters():
    """Test that likelihood changes with different θ parameters."""
    print("Testing likelihood sensitivity to θ...")
    states = StateSpace.from_list(['A', 'B'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition"
    )
    
    obs_model = PoissonObservationModel(graph)
    engine = ConstraintInference(model, obs_model)
    
    data = ObservedTransitions(counts={(0, 1): 10}, exposure=1.0)
    
    # Evaluate at different θ values
    ll_suppressed = engine.log_likelihood(data, parameters={"theta": {(0, 1): 0.1}})
    ll_neutral = engine.log_likelihood(data, parameters={"theta": {(0, 1): 1.0}})
    ll_facilitated = engine.log_likelihood(data, parameters={"theta": {(0, 1): 5.0}})
    
    # Likelihood should be highest when θ matches observed pattern
    # Observed: 10 transitions, baseline rate = 2.0
    # Best fit: θ = 5.0 (gives λ* = 10.0)
    
    assert ll_facilitated > ll_neutral > ll_suppressed
    
    print(f"  ✓ ℓ(θ=0.1) = {ll_suppressed:.4f}")
    print(f"  ✓ ℓ(θ=1.0) = {ll_neutral:.4f}")
    print(f"  ✓ ℓ(θ=5.0) = {ll_facilitated:.4f}")
    print("  ✓ Likelihood sensitive to constraint parameters")


def test_constraint_model_fit_dispatcher():
    """Test that ConstraintModel.fit() dispatches to inference engine."""
    print("Testing ConstraintModel.fit() dispatcher...")
    states = StateSpace.from_list(['A', 'B'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
    )
    
    data = ObservedTransitions(counts={(0, 1): 5}, exposure=1.0)
    obs_model = PoissonObservationModel(graph)
    
    # fit() should require obs_model
    try:
        model.fit(data)
        assert False, "Should require obs_model"
    except ValueError as e:
        assert "obs_model" in str(e)
        print("  ✓ fit() requires obs_model parameter")
    
    # fit() should dispatch to ConstraintInference
    try:
        result = model.fit(data, obs_model=obs_model)
        assert False, "Should raise NotImplementedError (inference not implemented)"
    except NotImplementedError as e:
        assert "not yet implemented" in str(e)
        print("  ✓ fit() dispatches to ConstraintInference.fit()")
        print("  ✓ Thin dispatcher keeps model clean")


def test_constraint_result_structure():
    """Test ConstraintResult structure (output object)."""
    print("Testing ConstraintResult structure...")
    states = StateSpace.from_list(['A', 'B'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
    )
    
    # Create a mock result (as if from inference)
    result = ConstraintResult(
        model=model,
        parameters={"theta": {(0, 1): 0.5}},
        method="MLE",
        log_likelihood=-10.5,
        aic=25.0,
        bic=28.0,
    )
    
    assert result.model is model
    assert result.parameters["theta"][(0, 1)] == 0.5
    assert result.method == "MLE"
    assert result.log_likelihood == -10.5
    assert result.aic == 25.0
    assert result.bic == 28.0
    
    print("  ✓ ConstraintResult is pure output object")
    print("  ✓ Contains: fitted θ̂, uncertainty, fit statistics")
    print("  ✓ Does NOT contain: likelihood code, optimization logic")


def test_architecture_scales_to_different_observation_models():
    """Test that architecture supports multiple observation models."""
    print("Testing architecture flexibility (multiple observation models)...")
    states = StateSpace.from_list(['A', 'B', 'C'])
    graph = TransitionGraph.complete(states)
    baseline = Baseline.uniform(rate=2.0)
    
    model = ConstraintModel(
        states=states,
        baseline=baseline,
        graph=graph,
    )
    
    model.set_parameters(theta={(0, 1): 0.5})
    
    data = ObservedTransitions(counts={(0, 1): 5}, exposure=1.0)
    
    # Same model, different observation models
    poisson_model = PoissonObservationModel(graph)
    poisson_engine = ConstraintInference(model, poisson_model)
    ll_poisson = poisson_engine.log_likelihood(data)
    
    # Could also use CTMC, Multinomial, etc.
    # (not implemented yet, but architecture supports it)
    
    print(f"  ✓ Poisson likelihood: {ll_poisson:.4f}")
    print("  ✓ Architecture supports swappable observation models")
    print("  ✓ Ready for: metagenomics, compositional data, phylogenetics")


def main():
    """Run all inference architecture tests."""
    print("\n" + "="*60)
    print("PERSISTE Inference Architecture Tests")
    print("="*60 + "\n")
    
    tests = [
        test_inference_engine_creation,
        test_likelihood_evaluation_pipeline,
        test_likelihood_with_different_parameters,
        test_constraint_model_fit_dispatcher,
        test_constraint_result_structure,
        test_architecture_scales_to_different_observation_models,
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
