"""
Validation with ALL Fixes Applied

Fixes:
1. RNG bug - use different seeds per simulation
2. Frequency counts (not just presence/absence)
3. Larger system (5 primitives, depth 5)
4. Stronger constraints (larger effect sizes)
5. More samples

This should reveal identifiability.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.observation.presence_model import FrequencyWeightedPresenceModel


def profile_likelihood_fixed(feature_name, theta_base, verbose=True):
    """
    Profile likelihood with ALL fixes applied.
    
    Fixes:
    - Different RNG seed per simulation (not always 42!)
    - Frequency counts (Poisson model)
    - Larger system (5 primitives, depth 5)
    - More samples (100 instead of 30)
    """
    if verbose:
        print("=" * 80)
        print(f"Profile Likelihood: {feature_name} (FIXED)")
        print("=" * 80)
    
    # Setup - LARGER SYSTEM
    primitives = ['A', 'B', 'C', 'D', 'E']  # 5 instead of 3
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)  # depth 5 instead of 3
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    initial_state = AssemblyState.from_parts(['A'], depth=0)
    
    # Generate data with base parameters
    constraint_base = AssemblyConstraint(feature_weights=theta_base)
    
    # FIX: Use None for RNG seed (different each time)
    simulator = GillespieSimulator(graph, baseline, constraint_base, rng=np.random.default_rng(None))
    
    # Simulate observations - MORE SAMPLES
    n_samples = 100  # increased from 30
    observed_counts = {}
    
    for _ in range(n_samples):
        traj = simulator.simulate(initial_state, t_max=50.0, burn_in=25.0)
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < obs_model.detection_prob:
                observed_counts[part] = observed_counts.get(part, 0) + 1
    
    if verbose:
        print(f"\nBase θ: {theta_base}")
        print(f"Observed counts: {observed_counts}")
        print(f"Total detections: {sum(observed_counts.values())}")
    
    # Grid search over feature_name
    grid_values = np.linspace(-2.0, 3.0, 11)  # Extended range
    log_liks = []
    
    if verbose:
        print(f"\nProfiling {feature_name}:")
        print(f"{'Value':>8s} {'Log L':>10s}")
        print("-" * 20)
    
    for value in grid_values:
        # Create theta with this value for feature_name
        theta_test = theta_base.copy()
        theta_test[feature_name] = value
        
        # Simulate with this theta - FIX: Use None for RNG (different each time)
        constraint_test = AssemblyConstraint(feature_weights=theta_test)
        sim_test = GillespieSimulator(graph, baseline, constraint_test, rng=np.random.default_rng(None))
        
        # MORE SAMPLES for latent state estimation
        latent_states = sim_test.sample_final_states(
            initial_state,
            n_samples=100,  # increased from 50
            t_max=50.0,
            burn_in=25.0,
        )
        
        # Compute likelihood
        ll = obs_model.compute_log_likelihood(observed_counts, latent_states)
        log_liks.append(ll)
        
        if verbose:
            marker = " ← BASE" if abs(value - theta_base[feature_name]) < 0.01 else ""
            print(f"{value:>8.2f} {ll:>10.4f}{marker}")
    
    # Analyze profile
    log_liks = np.array(log_liks)
    max_ll = np.max(log_liks)
    peak_idx = np.argmax(log_liks)
    peak_value = grid_values[peak_idx]
    
    # Compute curvature (second derivative at peak)
    if 0 < peak_idx < len(grid_values) - 1:
        curvature = abs(log_liks[peak_idx-1] - 2*log_liks[peak_idx] + log_liks[peak_idx+1])
    else:
        curvature = 0.0
    
    # Identifiability metrics
    ll_range = np.max(log_liks) - np.min(log_liks)
    
    if verbose:
        print(f"\nAnalysis:")
        print(f"  Peak at: {peak_value:.2f}")
        print(f"  Max log L: {max_ll:.4f}")
        print(f"  Range: {ll_range:.4f}")
        print(f"  Curvature: {curvature:.4f}")
        
        if ll_range < 1.0:
            print(f"  → FLAT (not identifiable)")
        elif curvature > 0.5:
            print(f"  → SHARP PEAK (identifiable)")
        else:
            print(f"  → WEAK SIGNAL (marginal identifiability)")
    
    return {
        'feature': feature_name,
        'grid_values': grid_values,
        'log_liks': log_liks,
        'peak_value': peak_value,
        'max_ll': max_ll,
        'll_range': ll_range,
        'curvature': curvature,
        'identifiable': ll_range > 1.0 and curvature > 0.5,
    }


def test_parameter_recovery_fixed(theta_true, verbose=True):
    """
    Test parameter recovery with ALL fixes.
    
    Uses frequency counts (not just presence/absence).
    """
    if verbose:
        print("=" * 80)
        print("Parameter Recovery Test (FIXED)")
        print("=" * 80)
        print(f"\nθ_true = {theta_true}")
    
    # Setup - LARGER SYSTEM
    primitives = ['A', 'B', 'C', 'D', 'E']
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    initial_state = AssemblyState.from_parts(['A'], depth=0)
    
    # Generate data with true parameters
    constraint_true = AssemblyConstraint(feature_weights=theta_true)
    simulator = GillespieSimulator(graph, baseline, constraint_true, rng=np.random.default_rng(None))
    
    # Simulate observations - MORE SAMPLES
    n_samples = 100
    observed_counts = {}
    
    for _ in range(n_samples):
        traj = simulator.simulate(initial_state, t_max=50.0, burn_in=25.0)
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < obs_model.detection_prob:
                observed_counts[part] = observed_counts.get(part, 0) + 1
    
    if verbose:
        print(f"\nObserved counts: {observed_counts}")
        print(f"Total detections: {sum(observed_counts.values())}")
    
    # Get latent state distribution under true parameters
    latent_states_true = simulator.sample_final_states(
        initial_state,
        n_samples=100,
        t_max=50.0,
        burn_in=25.0,
    )
    
    ll_true = obs_model.compute_log_likelihood(observed_counts, latent_states_true)
    
    if verbose:
        print(f"\nLog-likelihood under true θ: {ll_true:.4f}")
    
    # Compare to null model
    constraint_null = AssemblyConstraint.null_model()
    simulator_null = GillespieSimulator(graph, baseline, constraint_null, rng=np.random.default_rng(None))
    latent_states_null = simulator_null.sample_final_states(
        initial_state,
        n_samples=100,
        t_max=50.0,
        burn_in=25.0,
    )
    ll_null = obs_model.compute_log_likelihood(observed_counts, latent_states_null)
    
    if verbose:
        print(f"Log-likelihood under null θ: {ll_null:.4f}")
        print(f"Δ log-likelihood: {ll_true - ll_null:.4f}")
    
    improvement = ll_true - ll_null
    
    if verbose:
        if improvement > 2.0:
            print(f"\n✓ TRUE MODEL MUCH BETTER (Δ = {improvement:.4f})")
        elif improvement > 0:
            print(f"\n✓ True model better (Δ = {improvement:.4f})")
        else:
            print(f"\n✗ Null model better (Δ = {improvement:.4f})")
    
    return {
        'test': 'parameter_recovery_fixed',
        'theta_true': theta_true,
        'll_true': ll_true,
        'll_null': ll_null,
        'improvement': improvement,
        'identifiable': improvement > 2.0,
    }


def main():
    print("=" * 80)
    print("Validation with ALL Fixes Applied")
    print("=" * 80)
    
    print("\nFixes:")
    print("  1. ✓ RNG bug fixed (different seeds per simulation)")
    print("  2. ✓ Frequency counts (Poisson model, not just presence)")
    print("  3. ✓ Larger system (5 primitives, depth 5)")
    print("  4. ✓ More samples (100 instead of 30)")
    print("  5. ✓ Stronger constraints (testing larger effect sizes)")
    
    # ========================================================================
    # Test 1: Moderate Constraints
    # ========================================================================
    print("\n")
    theta_moderate = {'reuse_count': 1.5, 'depth_change': -0.5}
    result1 = test_parameter_recovery_fixed(theta_moderate, verbose=True)
    
    # ========================================================================
    # Test 2: Strong Constraints
    # ========================================================================
    print("\n")
    theta_strong = {'reuse_count': 3.0, 'depth_change': -1.0}
    result2 = test_parameter_recovery_fixed(theta_strong, verbose=True)
    
    # ========================================================================
    # Profile Likelihood: reuse_count
    # ========================================================================
    print("\n")
    result3 = profile_likelihood_fixed('reuse_count', theta_strong, verbose=True)
    
    # ========================================================================
    # Profile Likelihood: depth_change
    # ========================================================================
    print("\n")
    result4 = profile_likelihood_fixed('depth_change', theta_strong, verbose=True)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    print("\nParameter Recovery:")
    print(f"  Moderate constraints: improvement={result1['improvement']:.4f}, identifiable={result1['identifiable']}")
    print(f"  Strong constraints:   improvement={result2['improvement']:.4f}, identifiable={result2['identifiable']}")
    
    print("\nProfile Likelihoods:")
    print(f"  reuse_count:  range={result3['ll_range']:.4f}, curvature={result3['curvature']:.4f}, identifiable={result3['identifiable']}")
    print(f"  depth_change: range={result4['ll_range']:.4f}, curvature={result4['curvature']:.4f}, identifiable={result4['identifiable']}")
    
    # ========================================================================
    # Diagnosis
    # ========================================================================
    print("\n" + "=" * 80)
    print("Diagnosis")
    print("=" * 80)
    
    any_identifiable = (result1['identifiable'] or result2['identifiable'] or 
                        result3['identifiable'] or result4['identifiable'])
    
    if any_identifiable:
        print("\n✓ SUCCESS! Parameters are identifiable with fixes!")
        print("\nKey improvements:")
        if result3['ll_range'] > 1.0:
            print(f"  - reuse_count shows {result3['ll_range']:.1f} log-lik range")
        if result4['ll_range'] > 1.0:
            print(f"  - depth_change shows {result4['ll_range']:.1f} log-lik range")
        if result2['improvement'] > 2.0:
            print(f"  - Strong constraints give {result2['improvement']:.1f} log-lik improvement")
        
        print("\nWhat worked:")
        print("  ✓ Frequency counts (not just presence)")
        print("  ✓ Larger system (5 primitives, depth 5)")
        print("  ✓ Fixed RNG bug")
        print("  ✓ More samples")
    else:
        print("\n⚠ Still challenging even with fixes")
        print("\nPossible remaining issues:")
        print("  - Constraints still too weak (try even larger values)")
        print("  - Need even more samples (200+)")
        print("  - Need longer simulation time")
        print("  - Observation model may need refinement")
        
        print("\nBut we've made progress:")
        print(f"  - Moderate: Δ log-lik = {result1['improvement']:.2f}")
        print(f"  - Strong:   Δ log-lik = {result2['improvement']:.2f}")
        if result2['improvement'] > result1['improvement']:
            print("  → Stronger constraints do help!")
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
