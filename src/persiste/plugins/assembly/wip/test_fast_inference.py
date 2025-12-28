"""
Test that fast inference mode completes quickly (<30s).
"""

import sys
sys.path.insert(0, 'src')

import time
import numpy as np
from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.baselines.baseline_family import SimpleBaselineFamily
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.observation.presence_model import FrequencyWeightedPresenceModel
from persiste.plugins.assembly.inference.robust_inference_v2 import RobustConstraintInferenceV2


def generate_test_data():
    """Generate synthetic test data."""
    primitives = ['A', 'B', 'C', 'D', 'E']
    theta_true = {'reuse_count': 1.5, 'depth_change': -0.5}
    baseline_true = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    constraint = AssemblyConstraint(feature_weights=theta_true)
    simulator = GillespieSimulator(graph, baseline_true, constraint, rng=np.random.default_rng(42))
    
    observed_counts = {}
    for _ in range(100):
        traj = simulator.simulate(
            AssemblyState.from_parts([primitives[0]], depth=0),
            t_max=50.0,
            burn_in=25.0
        )
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < 0.9:
                observed_counts[part] = observed_counts.get(part, 0) + 1
    
    return observed_counts


def test_fast_mode():
    """Test fast mode completes in <30s."""
    print("=" * 80)
    print("TEST: Fast Inference Mode")
    print("=" * 80)
    
    # Generate data
    print("\n[1] Generating test data...")
    t0 = time.time()
    observed_counts = generate_test_data()
    data_time = time.time() - t0
    print(f"  Time: {data_time:.2f}s")
    print(f"  Observed: {len(observed_counts)} compounds, {sum(observed_counts.values())} detections")
    
    # Setup inference
    print("\n[2] Setting up inference...")
    primitives = ['A', 'B', 'C', 'D', 'E']
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline_family = SimpleBaselineFamily(parameter='join_exponent', initial_value=-0.5, prior_std=0.2)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    
    inference = RobustConstraintInferenceV2(
        graph=graph,
        baseline_family=baseline_family,
        obs_model=obs_model,
        regularization=0.1,
    )
    
    # Test FAST mode
    print("\n[3] Running FAST mode...")
    t0 = time.time()
    result = inference.fit_with_null(
        observed_counts,
        constraint_features=['reuse_count', 'depth_change'],
        mode='fast',
        verbose=True,
    )
    fast_time = time.time() - t0
    
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"\nTotal time: {fast_time:.2f}s")
    print(f"Evidence: {result.evidence}")
    print(f"Δ LL: {result.delta_ll:.2f}")
    print(f"Estimates: {result.estimate}")
    
    # Check timing
    print("\n" + "=" * 80)
    print("PERFORMANCE CHECK")
    print("=" * 80)
    if fast_time < 30.0:
        print(f"\n✓ PASS: Fast mode completed in {fast_time:.2f}s (<30s target)")
    else:
        print(f"\n✗ FAIL: Fast mode took {fast_time:.2f}s (>30s target)")
    
    return fast_time < 30.0


def test_standard_mode():
    """Test standard mode with conditional diagnostics."""
    print("\n\n" + "=" * 80)
    print("TEST: Standard Inference Mode (Conditional Diagnostics)")
    print("=" * 80)
    
    # Generate data
    print("\n[1] Generating test data...")
    observed_counts = generate_test_data()
    
    # Setup inference
    primitives = ['A', 'B', 'C', 'D', 'E']
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline_family = SimpleBaselineFamily(parameter='join_exponent', initial_value=-0.5, prior_std=0.2)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    
    inference = RobustConstraintInferenceV2(
        graph=graph,
        baseline_family=baseline_family,
        obs_model=obs_model,
        regularization=0.1,
    )
    
    # Test STANDARD mode
    print("\n[2] Running STANDARD mode...")
    t0 = time.time()
    result = inference.fit_with_null(
        observed_counts,
        constraint_features=['reuse_count', 'depth_change'],
        mode='standard',
        verbose=True,
    )
    standard_time = time.time() - t0
    
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"\nTotal time: {standard_time:.2f}s")
    print(f"Evidence: {result.evidence}")
    print(f"Diagnostics run: {len(result.profile_diagnostics) > 0}")
    
    # Check timing
    print("\n" + "=" * 80)
    print("PERFORMANCE CHECK")
    print("=" * 80)
    if standard_time < 120.0:
        print(f"\n✓ PASS: Standard mode completed in {standard_time:.2f}s (<2min target)")
    else:
        print(f"\n✗ FAIL: Standard mode took {standard_time:.2f}s (>2min target)")
    
    return standard_time < 120.0


if __name__ == '__main__':
    # Test fast mode
    fast_pass = test_fast_mode()
    
    # Test standard mode
    standard_pass = test_standard_mode()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nFast mode:     {'✓ PASS' if fast_pass else '✗ FAIL'}")
    print(f"Standard mode: {'✓ PASS' if standard_pass else '✗ FAIL'}")
    
    if fast_pass and standard_pass:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
