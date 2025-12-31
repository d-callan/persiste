"""
Validation with Improved Observations (Options A & B)

Tests:
1. Frequency-weighted presence (Option A - cheap, high value)
2. Time-sliced presence (Option B - very powerful)
3. Profile likelihood diagnostics (systematic identifiability analysis)
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
from persiste.plugins.assembly.observation.timeslice_model import TimeSlicedPresenceModel


def test_frequency_weighted_recovery(theta_true, verbose=True):
    """
    Test parameter recovery with frequency-weighted observations (Option A).
    
    This breaks symmetry between θ that affect rates vs reachability.
    """
    if verbose:
        print("=" * 80)
        print("Test: Frequency-Weighted Presence (Option A)")
        print("=" * 80)
        print(f"\nθ_true = {theta_true}")
    
    # Setup
    primitives = ['A', 'B', 'C']
    graph = AssemblyGraph(primitives, max_depth=3, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    initial_state = AssemblyState.from_parts(['A'], depth=0)
    
    # Generate data with true parameters
    constraint_true = AssemblyConstraint(feature_weights=theta_true)
    simulator = GillespieSimulator(graph, baseline, constraint_true, rng=np.random.default_rng(42))
    
    # Simulate multiple samples
    n_samples = 50
    observed_counts = {}
    
    for _ in range(n_samples):
        traj = simulator.simulate(initial_state, t_max=30.0, burn_in=15.0)
        final_state = traj.final_state()
        
        # Detect compounds with some probability
        for part in final_state.get_parts_list():
            if np.random.rand() < obs_model.detection_prob:
                observed_counts[part] = observed_counts.get(part, 0) + 1
    
    if verbose:
        print(f"\nObserved counts: {observed_counts}")
        print(f"Total detections: {sum(observed_counts.values())}")
    
    # Get latent state distribution
    latent_states = simulator.sample_final_states(
        initial_state,
        n_samples=100,
        t_max=30.0,
        burn_in=15.0,
    )
    
    # Compute likelihood under true parameters
    ll_true = obs_model.compute_log_likelihood(observed_counts, latent_states)
    
    if verbose:
        print(f"\nLog-likelihood under true θ: {ll_true:.4f}")
    
    # Compute likelihood under null model
    constraint_null = AssemblyConstraint.null_model()
    simulator_null = GillespieSimulator(graph, baseline, constraint_null, rng=np.random.default_rng(42))
    latent_states_null = simulator_null.sample_final_states(
        initial_state,
        n_samples=100,
        t_max=30.0,
        burn_in=15.0,
    )
    ll_null = obs_model.compute_log_likelihood(observed_counts, latent_states_null)
    
    if verbose:
        print(f"Log-likelihood under null θ: {ll_null:.4f}")
        print(f"Δ log-likelihood: {ll_true - ll_null:.4f}")
    
    # Check if true model is better
    improvement = ll_true - ll_null
    
    if verbose:
        if improvement > 0:
            print(f"\n✓ True model is better (Δ = {improvement:.4f})")
        else:
            print(f"\n✗ Null model is better (Δ = {improvement:.4f})")
    
    return {
        'test': 'frequency_weighted',
        'theta_true': theta_true,
        'll_true': ll_true,
        'll_null': ll_null,
        'improvement': improvement,
        'identifiable': improvement > 1.0,  # Generous threshold
    }


def test_timeslice_recovery(theta_true, verbose=True):
    """
    Test parameter recovery with time-sliced observations (Option B).
    
    This adds directional information and dynamics.
    """
    if verbose:
        print("=" * 80)
        print("Test: Time-Sliced Presence (Option B)")
        print("=" * 80)
        print(f"\nθ_true = {theta_true}")
    
    # Setup
    primitives = ['A', 'B', 'C']
    graph = AssemblyGraph(primitives, max_depth=3, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    obs_model = TimeSlicedPresenceModel(detection_prob=0.9, false_positive_prob=0.01)
    initial_state = AssemblyState.from_parts(['A'], depth=0)
    
    # Generate data with true parameters
    constraint_true = AssemblyConstraint(feature_weights=theta_true)
    simulator = GillespieSimulator(graph, baseline, constraint_true, rng=np.random.default_rng(42))
    
    # Simulate trajectories
    n_trajectories = 30
    trajectories = []
    
    for _ in range(n_trajectories):
        traj = simulator.simulate(initial_state, t_max=50.0)
        trajectories.append(traj)
    
    # Create time slices
    time_points = [10.0, 20.0, 30.0, 40.0]
    time_slices = {}
    
    for t in time_points:
        observed = set()
        for traj in trajectories:
            # Find state at time t
            for i, time in enumerate(traj.times):
                if time >= t:
                    state = traj.states[i]
                    # Detect compounds
                    for part in state.get_parts_list():
                        if np.random.rand() < obs_model.detection_prob:
                            observed.add(part)
                    break
        time_slices[t] = observed
    
    if verbose:
        print(f"\nTime slices:")
        for t, compounds in sorted(time_slices.items()):
            print(f"  t={t:5.1f}: {compounds}")
    
    # Compute likelihood under true parameters
    ll_true = obs_model.compute_log_likelihood(time_slices, trajectories)
    
    if verbose:
        print(f"\nLog-likelihood under true θ: {ll_true:.4f}")
    
    # Compute likelihood under null model
    constraint_null = AssemblyConstraint.null_model()
    simulator_null = GillespieSimulator(graph, baseline, constraint_null, rng=np.random.default_rng(42))
    trajectories_null = []
    for _ in range(n_trajectories):
        traj = simulator_null.simulate(initial_state, t_max=50.0)
        trajectories_null.append(traj)
    
    ll_null = obs_model.compute_log_likelihood(time_slices, trajectories_null)
    
    if verbose:
        print(f"Log-likelihood under null θ: {ll_null:.4f}")
        print(f"Δ log-likelihood: {ll_true - ll_null:.4f}")
    
    # Check if true model is better
    improvement = ll_true - ll_null
    
    if verbose:
        if improvement > 0:
            print(f"\n✓ True model is better (Δ = {improvement:.4f})")
        else:
            print(f"\n✗ Null model is better (Δ = {improvement:.4f})")
    
    return {
        'test': 'timeslice',
        'theta_true': theta_true,
        'll_true': ll_true,
        'll_null': ll_null,
        'improvement': improvement,
        'identifiable': improvement > 1.0,
    }


def profile_likelihood_analysis(feature_name, theta_base, verbose=True):
    """
    Profile likelihood for systematic identifiability diagnosis.
    
    For each θᵢ:
    - Fix θᵢ at grid values
    - Compute likelihood (with other params at base values)
    - Plot log L
    
    Outcomes:
    - Flat → not identifiable
    - Sharp peak → identifiable
    - Ridge → parameter tradeoff
    """
    if verbose:
        print("=" * 80)
        print(f"Profile Likelihood: {feature_name}")
        print("=" * 80)
    
    # Setup
    primitives = ['A', 'B', 'C']
    graph = AssemblyGraph(primitives, max_depth=3, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    initial_state = AssemblyState.from_parts(['A'], depth=0)
    
    # Generate data with base parameters
    constraint_base = AssemblyConstraint(feature_weights=theta_base)
    simulator = GillespieSimulator(graph, baseline, constraint_base, rng=np.random.default_rng(42))
    
    # Simulate observations
    n_samples = 30
    observed_counts = {}
    
    for _ in range(n_samples):
        traj = simulator.simulate(initial_state, t_max=30.0, burn_in=15.0)
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < obs_model.detection_prob:
                observed_counts[part] = observed_counts.get(part, 0) + 1
    
    if verbose:
        print(f"\nBase θ: {theta_base}")
        print(f"Observed counts: {observed_counts}")
    
    # Grid search over feature_name
    grid_values = np.linspace(-2.0, 2.0, 11)
    log_liks = []
    
    if verbose:
        print(f"\nProfiling {feature_name}:")
        print(f"{'Value':>8s} {'Log L':>10s}")
        print("-" * 20)
    
    for value in grid_values:
        # Create theta with this value for feature_name
        theta_test = theta_base.copy()
        theta_test[feature_name] = value
        
        # Simulate with this theta
        constraint_test = AssemblyConstraint(feature_weights=theta_test)
        sim_test = GillespieSimulator(graph, baseline, constraint_test, rng=np.random.default_rng(42))
        latent_states = sim_test.sample_final_states(
            initial_state,
            n_samples=50,
            t_max=30.0,
            burn_in=15.0,
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
        curvature = (log_liks[peak_idx-1] - 2*log_liks[peak_idx] + log_liks[peak_idx+1])
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
        
        if ll_range < 0.5:
            print(f"  → FLAT (not identifiable)")
        elif abs(curvature) > 0.1:
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
        'identifiable': ll_range > 0.5,
    }


def main():
    print("=" * 80)
    print("Validation with Improved Observations")
    print("=" * 80)
    
    print("\nOptions:")
    print("  A. Frequency-weighted presence (cheap, high value)")
    print("  B. Time-sliced presence (very powerful)")
    print("  C. Profile likelihood diagnostics")
    
    results = []
    
    # ========================================================================
    # Option A: Frequency-Weighted Presence
    # ========================================================================
    print("\n")
    theta_test = {'reuse_count': 1.0, 'depth_change': -0.3}
    result_a = test_frequency_weighted_recovery(theta_test, verbose=True)
    results.append(result_a)
    
    # ========================================================================
    # Option B: Time-Sliced Presence
    # ========================================================================
    print("\n")
    result_b = test_timeslice_recovery(theta_test, verbose=True)
    results.append(result_b)
    
    # ========================================================================
    # Profile Likelihood Analysis
    # ========================================================================
    print("\n")
    result_c1 = profile_likelihood_analysis('reuse_count', theta_test, verbose=True)
    
    print("\n")
    result_c2 = profile_likelihood_analysis('depth_change', theta_test, verbose=True)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    print("\nOption A (Frequency-Weighted):")
    print(f"  Improvement: {result_a['improvement']:.4f}")
    print(f"  Identifiable: {result_a['identifiable']}")
    
    print("\nOption B (Time-Sliced):")
    print(f"  Improvement: {result_b['improvement']:.4f}")
    print(f"  Identifiable: {result_b['identifiable']}")
    
    print("\nProfile Likelihoods:")
    print(f"  reuse_count:  range={result_c1['ll_range']:.4f}, identifiable={result_c1['identifiable']}")
    print(f"  depth_change: range={result_c2['ll_range']:.4f}, identifiable={result_c2['identifiable']}")
    
    print("\n" + "=" * 80)
    print("Diagnosis")
    print("=" * 80)
    
    if result_a['identifiable'] or result_b['identifiable']:
        print("\n✓ Richer observations help!")
        print("\nKey findings:")
        if result_a['identifiable']:
            print("  - Frequency weighting breaks symmetry")
        if result_b['identifiable']:
            print("  - Time slices add directional information")
    else:
        print("\n⚠ Still challenging")
        print("\nPossible issues:")
        print("  - Need more samples")
        print("  - Need stronger constraints")
        print("  - Need richer chemistry (more primitives)")
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
