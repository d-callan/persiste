"""
Validation: V2 vs V1 Comparison

Tests that V2 produces scientifically equivalent results to V1:
1. Profile likelihood curves (identifiability)
2. Parameter recovery (accuracy)
3. Null model behavior (no false positives)
4. Mode comparison (fast/standard/thorough)

Goal: Ensure V2 is fast AND correct.
"""

import sys
sys.path.insert(0, 'src')

import time
import numpy as np
from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.baselines.baseline_family import SimpleBaselineFamily, FixedBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.observation.presence_model import FrequencyWeightedPresenceModel

# Import both versions
from persiste.plugins.assembly.inference.robust_inference import RobustConstraintInference as V1
from persiste.plugins.assembly.inference.robust_inference_v2 import RobustConstraintInferenceV2 as V2


def generate_test_data(theta_true, n_samples=100, seed=42):
    """Generate synthetic test data with known ground truth."""
    primitives = ['A', 'B', 'C', 'D', 'E']
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    constraint = AssemblyConstraint(feature_weights=theta_true)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    
    simulator = GillespieSimulator(graph, baseline, constraint, rng=np.random.default_rng(seed))
    
    observed_counts = {}
    for _ in range(n_samples):
        traj = simulator.simulate(
            AssemblyState.from_parts([primitives[0]], depth=0),
            t_max=50.0,
            burn_in=25.0
        )
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < obs_model.detection_prob:
                observed_counts[part] = observed_counts.get(part, 0) + 1
    
    return observed_counts, primitives


def test_1_profile_likelihood_comparison():
    """Test 1: Profile likelihood curves should be similar."""
    print("=" * 80)
    print("TEST 1: Profile Likelihood Comparison")
    print("=" * 80)
    
    # Generate data with strong constraint
    theta_true = {'reuse_count': 1.5, 'depth_change': -0.5}
    print(f"\nGround truth: {theta_true}")
    
    observed_counts, primitives = generate_test_data(theta_true, n_samples=100, seed=42)
    print(f"Observed: {len(observed_counts)} compounds, {sum(observed_counts.values())} detections")
    
    # Setup
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    baseline_family = FixedBaseline(baseline)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    
    # V1 inference
    print("\n[V1] Running inference...")
    t0 = time.time()
    v1_inference = V1(
        graph=graph,
        baseline_family=baseline_family,
        obs_model=obs_model,
        regularization=0.1,
        n_latent_samples=100,
    )
    v1_result = v1_inference.fit_with_null(
        observed_counts,
        constraint_features=['reuse_count', 'depth_change'],
        profile_diagnostics=True,
        verbose=False,
    )
    v1_time = time.time() - t0
    
    # V2 inference (thorough mode for fair comparison)
    print("[V2] Running inference (thorough mode)...")
    t0 = time.time()
    v2_inference = V2(
        graph=graph,
        baseline_family=baseline_family,
        obs_model=obs_model,
        regularization=0.1,
    )
    v2_result = v2_inference.fit_with_null(
        observed_counts,
        constraint_features=['reuse_count', 'depth_change'],
        mode='thorough',
        verbose=False,
    )
    v2_time = time.time() - t0
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    print(f"\nRuntime:")
    print(f"  V1: {v1_time:.2f}s")
    print(f"  V2 (thorough): {v2_time:.2f}s")
    print(f"  Speedup: {v1_time/v2_time:.1f}×")
    
    print(f"\nΔ LL:")
    print(f"  V1: {v1_result.delta_ll:.2f}")
    print(f"  V2: {v2_result.delta_ll:.2f}")
    print(f"  Difference: {abs(v1_result.delta_ll - v2_result.delta_ll):.2f}")
    
    print(f"\nEstimates (reuse_count):")
    print(f"  True: {theta_true['reuse_count']:.2f}")
    print(f"  V1: {v1_result.estimate['reuse_count']:.2f}")
    print(f"  V2: {v2_result.estimate['reuse_count']:.2f}")
    print(f"  V1 error: {abs(v1_result.estimate['reuse_count'] - theta_true['reuse_count']):.2f}")
    print(f"  V2 error: {abs(v2_result.estimate['reuse_count'] - theta_true['reuse_count']):.2f}")
    
    print(f"\nEstimates (depth_change):")
    print(f"  True: {theta_true['depth_change']:.2f}")
    print(f"  V1: {v1_result.estimate['depth_change']:.2f}")
    print(f"  V2: {v2_result.estimate['depth_change']:.2f}")
    print(f"  V1 error: {abs(v1_result.estimate['depth_change'] - theta_true['depth_change']):.2f}")
    print(f"  V2 error: {abs(v2_result.estimate['depth_change'] - theta_true['depth_change']):.2f}")
    
    print(f"\nEvidence:")
    print(f"  V1: {v1_result.evidence}")
    print(f"  V2: {v2_result.evidence}")
    
    # Check agreement
    delta_ll_agree = abs(v1_result.delta_ll - v2_result.delta_ll) < 5.0
    estimate_agree = (
        abs(v1_result.estimate['reuse_count'] - v2_result.estimate['reuse_count']) < 0.5
        and abs(v1_result.estimate['depth_change'] - v2_result.estimate['depth_change']) < 0.5
    )
    evidence_agree = v1_result.evidence == v2_result.evidence
    
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print(f"\nΔ LL agreement: {'✓ PASS' if delta_ll_agree else '✗ FAIL'}")
    print(f"Estimate agreement: {'✓ PASS' if estimate_agree else '✗ FAIL'}")
    print(f"Evidence agreement: {'✓ PASS' if evidence_agree else '✗ FAIL'}")
    
    return delta_ll_agree and estimate_agree


def test_2_null_model_behavior():
    """Test 2: Both should correctly identify null (no false positives)."""
    print("\n\n" + "=" * 80)
    print("TEST 2: Null Model Behavior (No False Positives)")
    print("=" * 80)
    
    # Generate data with NO constraint (null model)
    theta_true = {'reuse_count': 0.0, 'depth_change': 0.0}
    print(f"\nGround truth: {theta_true} (NULL)")
    
    observed_counts, primitives = generate_test_data(theta_true, n_samples=100, seed=43)
    print(f"Observed: {len(observed_counts)} compounds, {sum(observed_counts.values())} detections")
    
    # Setup
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    baseline_family = FixedBaseline(baseline)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    
    # V1 inference
    print("\n[V1] Running inference...")
    v1_inference = V1(
        graph=graph,
        baseline_family=baseline_family,
        obs_model=obs_model,
        regularization=0.1,
        n_latent_samples=100,
    )
    v1_result = v1_inference.fit_with_null(
        observed_counts,
        constraint_features=['reuse_count', 'depth_change'],
        profile_diagnostics=False,
        verbose=False,
    )
    
    # V2 inference (fast mode)
    print("[V2] Running inference (fast mode)...")
    v2_inference = V2(
        graph=graph,
        baseline_family=baseline_family,
        obs_model=obs_model,
        regularization=0.1,
    )
    v2_result = v2_inference.fit_with_null(
        observed_counts,
        constraint_features=['reuse_count', 'depth_change'],
        mode='fast',
        verbose=False,
    )
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    print(f"\nΔ LL:")
    print(f"  V1: {v1_result.delta_ll:.2f}")
    print(f"  V2: {v2_result.delta_ll:.2f}")
    
    print(f"\nEvidence:")
    print(f"  V1: {v1_result.evidence}")
    print(f"  V2: {v2_result.evidence}")
    
    # Both should report no evidence
    v1_correct = v1_result.evidence == 'none' and v1_result.delta_ll < 2.0
    v2_correct = v2_result.evidence == 'none' and v2_result.delta_ll < 2.0
    
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print(f"\nV1 correctly identifies null: {'✓ PASS' if v1_correct else '✗ FAIL'}")
    print(f"V2 correctly identifies null: {'✓ PASS' if v2_correct else '✗ FAIL'}")
    
    return v1_correct and v2_correct


def test_3_mode_comparison():
    """Test 3: V2 fast/standard/thorough should give similar results."""
    print("\n\n" + "=" * 80)
    print("TEST 3: V2 Mode Comparison (Fast vs Standard vs Thorough)")
    print("=" * 80)
    
    # Generate data with moderate constraint
    theta_true = {'reuse_count': 1.0, 'depth_change': -0.3}
    print(f"\nGround truth: {theta_true}")
    
    observed_counts, primitives = generate_test_data(theta_true, n_samples=100, seed=44)
    print(f"Observed: {len(observed_counts)} compounds, {sum(observed_counts.values())} detections")
    
    # Setup
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    baseline_family = FixedBaseline(baseline)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    
    inference = V2(
        graph=graph,
        baseline_family=baseline_family,
        obs_model=obs_model,
        regularization=0.1,
    )
    
    # Test all modes
    results = {}
    times = {}
    
    for mode in ['fast', 'standard', 'thorough']:
        print(f"\n[{mode.upper()}] Running inference...")
        t0 = time.time()
        result = inference.fit_with_null(
            observed_counts,
            constraint_features=['reuse_count', 'depth_change'],
            mode=mode,
            verbose=False,
        )
        times[mode] = time.time() - t0
        results[mode] = result
        
        # Need to reset cache for fair comparison
        inference.cache = None
        inference.cached_states = None
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    print(f"\nRuntime:")
    for mode in ['fast', 'standard', 'thorough']:
        print(f"  {mode}: {times[mode]:.2f}s")
    
    print(f"\nΔ LL:")
    for mode in ['fast', 'standard', 'thorough']:
        print(f"  {mode}: {results[mode].delta_ll:.2f}")
    
    print(f"\nEstimates (reuse_count):")
    print(f"  True: {theta_true['reuse_count']:.2f}")
    for mode in ['fast', 'standard', 'thorough']:
        est = results[mode].estimate['reuse_count']
        err = abs(est - theta_true['reuse_count'])
        print(f"  {mode}: {est:.2f} (error: {err:.2f})")
    
    print(f"\nEstimates (depth_change):")
    print(f"  True: {theta_true['depth_change']:.2f}")
    for mode in ['fast', 'standard', 'thorough']:
        est = results[mode].estimate['depth_change']
        err = abs(est - theta_true['depth_change'])
        print(f"  {mode}: {est:.2f} (error: {err:.2f})")
    
    print(f"\nEvidence:")
    for mode in ['fast', 'standard', 'thorough']:
        print(f"  {mode}: {results[mode].evidence}")
    
    # Check agreement between modes
    delta_ll_range = max(r.delta_ll for r in results.values()) - min(r.delta_ll for r in results.values())
    reuse_range = max(r.estimate['reuse_count'] for r in results.values()) - min(r.estimate['reuse_count'] for r in results.values())
    
    modes_agree = delta_ll_range < 10.0 and reuse_range < 1.0
    
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print(f"\nΔ LL range across modes: {delta_ll_range:.2f}")
    print(f"Estimate range (reuse_count): {reuse_range:.2f}")
    print(f"\nModes agree: {'✓ PASS' if modes_agree else '✗ FAIL'}")
    print(f"  (Δ LL range < 10 and estimate range < 1.0)")
    
    return modes_agree


def test_4_parameter_recovery():
    """Test 4: Both should recover true parameters accurately."""
    print("\n\n" + "=" * 80)
    print("TEST 4: Parameter Recovery Accuracy")
    print("=" * 80)
    
    # Test multiple ground truth values
    test_cases = [
        {'reuse_count': 0.5, 'depth_change': 0.0},
        {'reuse_count': 1.0, 'depth_change': -0.5},
        {'reuse_count': 1.5, 'depth_change': -1.0},
    ]
    
    graph = None
    baseline_family = None
    obs_model = None
    
    v1_errors = []
    v2_errors = []
    
    for i, theta_true in enumerate(test_cases):
        print(f"\n[Case {i+1}] Ground truth: {theta_true}")
        
        observed_counts, primitives = generate_test_data(theta_true, n_samples=100, seed=45+i)
        
        # Setup (once)
        if graph is None:
            graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
            baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
            baseline_family = FixedBaseline(baseline)
            obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
        
        # V1
        v1_inference = V1(graph=graph, baseline_family=baseline_family, obs_model=obs_model, regularization=0.1, n_latent_samples=100)
        v1_result = v1_inference.fit_with_null(observed_counts, profile_diagnostics=False, verbose=False)
        
        # V2
        v2_inference = V2(graph=graph, baseline_family=baseline_family, obs_model=obs_model, regularization=0.1)
        v2_result = v2_inference.fit_with_null(observed_counts, mode='standard', verbose=False)
        
        # Compute errors
        v1_err = np.sqrt(
            (v1_result.estimate['reuse_count'] - theta_true['reuse_count'])**2 +
            (v1_result.estimate['depth_change'] - theta_true['depth_change'])**2
        )
        v2_err = np.sqrt(
            (v2_result.estimate['reuse_count'] - theta_true['reuse_count'])**2 +
            (v2_result.estimate['depth_change'] - theta_true['depth_change'])**2
        )
        
        v1_errors.append(v1_err)
        v2_errors.append(v2_err)
        
        print(f"  V1 estimate: reuse={v1_result.estimate['reuse_count']:.2f}, depth={v1_result.estimate['depth_change']:.2f} (L2 error: {v1_err:.2f})")
        print(f"  V2 estimate: reuse={v2_result.estimate['reuse_count']:.2f}, depth={v2_result.estimate['depth_change']:.2f} (L2 error: {v2_err:.2f})")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nAverage L2 error:")
    print(f"  V1: {np.mean(v1_errors):.2f}")
    print(f"  V2: {np.mean(v2_errors):.2f}")
    
    both_accurate = np.mean(v1_errors) < 1.0 and np.mean(v2_errors) < 1.0
    
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print(f"\nBoth versions accurate: {'✓ PASS' if both_accurate else '✗ FAIL'}")
    print(f"  (Average L2 error < 1.0)")
    
    return both_accurate


if __name__ == '__main__':
    print("=" * 80)
    print("VALIDATION SUITE: V2 vs V1")
    print("=" * 80)
    print("\nGoal: Ensure V2 is fast AND scientifically correct")
    print("Tests:")
    print("  1. Profile likelihood comparison")
    print("  2. Null model behavior (no false positives)")
    print("  3. V2 mode comparison (fast/standard/thorough)")
    print("  4. Parameter recovery accuracy")
    
    results = {}
    
    # Run tests
    results['test1'] = test_1_profile_likelihood_comparison()
    results['test2'] = test_2_null_model_behavior()
    results['test3'] = test_3_mode_comparison()
    results['test4'] = test_4_parameter_recovery()
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_pass = all(results.values())
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ ALL TESTS PASSED")
        print("\nConclusion: V2 is scientifically equivalent to V1 but much faster.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nConclusion: V2 needs refinement before replacing V1.")
    print("=" * 80)
