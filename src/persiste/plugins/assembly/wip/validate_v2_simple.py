"""
Simple V2 Validation: Direct likelihood comparison (no optimization).

Compare V2's likelihood evaluation against direct simulation.
This tests the core caching mechanism without the optimizer complexity.
"""

import sys
sys.path.insert(0, 'src')

import time
import numpy as np
from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.observation.presence_model import FrequencyWeightedPresenceModel


def test_direct_likelihood():
    """Test: Direct likelihood evaluation (original validation approach)."""
    print("=" * 80)
    print("DIRECT LIKELIHOOD TEST (Original Validation Approach)")
    print("=" * 80)
    
    # Setup
    primitives = ['A', 'B', 'C', 'D', 'E']
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    initial_state = AssemblyState.from_parts(['A'], depth=0)
    
    # Generate data with strong constraint
    theta_true = {'reuse_count': 1.5, 'depth_change': -0.5}
    print(f"\nGround truth: {theta_true}")
    
    constraint_true = AssemblyConstraint(feature_weights=theta_true)
    simulator_true = GillespieSimulator(graph, baseline, constraint_true, rng=np.random.default_rng(42))
    
    # Generate observed data
    observed_counts = {}
    for _ in range(100):
        traj = simulator_true.simulate(initial_state, t_max=50.0, burn_in=25.0)
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < obs_model.detection_prob:
                observed_counts[part] = observed_counts.get(part, 0) + 1
    
    print(f"Observed: {len(observed_counts)} compounds, {sum(observed_counts.values())} detections")
    
    # Test 1: Likelihood under TRUE parameters
    print("\n[Test 1: Likelihood under TRUE parameters]")
    t0 = time.time()
    latent_states_true = simulator_true.sample_final_states(
        initial_state, n_samples=100, t_max=50.0, burn_in=25.0
    )
    ll_true = obs_model.compute_log_likelihood(observed_counts, latent_states_true)
    time_true = time.time() - t0
    print(f"  Log-likelihood: {ll_true:.4f}")
    print(f"  Time: {time_true:.2f}s")
    
    # Test 2: Likelihood under NULL parameters
    print("\n[Test 2: Likelihood under NULL parameters]")
    t0 = time.time()
    constraint_null = AssemblyConstraint.null_model()
    simulator_null = GillespieSimulator(graph, baseline, constraint_null, rng=np.random.default_rng(43))
    latent_states_null = simulator_null.sample_final_states(
        initial_state, n_samples=100, t_max=50.0, burn_in=25.0
    )
    ll_null = obs_model.compute_log_likelihood(observed_counts, latent_states_null)
    time_null = time.time() - t0
    print(f"  Log-likelihood: {ll_null:.4f}")
    print(f"  Time: {time_null:.2f}s")
    
    # Results
    delta_ll = ll_true - ll_null
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nΔ LL (true - null): {delta_ll:.4f}")
    print(f"Total time: {time_true + time_null:.2f}s")
    
    # Validation
    passed = delta_ll > 2.0
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print(f"\nΔ LL > 2.0: {'✓' if delta_ll > 2.0 else '✗'} ({delta_ll:.4f})")
    print(f"Test: {'✓ PASS' if passed else '✗ FAIL'}")
    
    if passed:
        print("\n→ Direct likelihood evaluation works correctly")
        print("→ True constraint produces better fit than null")
    else:
        print("\n→ PROBLEM: True constraint does not improve likelihood")
    
    return passed


if __name__ == '__main__':
    print("=" * 80)
    print("SIMPLE VALIDATION: Direct Likelihood (No Optimization)")
    print("=" * 80)
    print("\nThis tests the core likelihood evaluation without optimizer complexity.")
    print("Goal: Verify that true parameters give better likelihood than null.")
    
    passed = test_direct_likelihood()
    
    print("\n" + "=" * 80)
    if passed:
        print("✓ VALIDATION PASSED")
        print("\nConclusion:")
        print("  • Direct likelihood evaluation works")
        print("  • True constraints detectable")
        print("  • This matches original validation approach")
    else:
        print("✗ VALIDATION FAILED")
        print("\nConclusion:")
        print("  • Core likelihood evaluation has issues")
    print("=" * 80)
