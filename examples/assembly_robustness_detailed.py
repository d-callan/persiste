"""
Detailed Robustness Analysis

Investigate the false positive issue more carefully.
Is it a real problem or an artifact of:
- Limited grid resolution
- Stochastic noise
- Small sample size
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from typing import Dict, List

from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.observation.presence_model import FrequencyWeightedPresenceModel


def detailed_false_positive_test(verbose: bool = True) -> Dict:
    """
    Detailed investigation of false positive.
    
    Use finer grid, more samples, multiple replicates.
    """
    if verbose:
        print("=" * 80)
        print("Detailed False Positive Investigation")
        print("=" * 80)
        print("\nScenario: Null model + wrong baseline")
        print("Question: Is peak at -1.0 a real false positive or artifact?")
    
    primitives = ['A', 'B', 'C', 'D', 'E']
    
    # TRUE baseline (data generation)
    baseline_true = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    
    # WRONG baseline (inference) - 20% off
    baseline_wrong = AssemblyBaseline(kappa=1.2, join_exponent=-0.6, split_exponent=0.4)
    
    # Generate data with NULL model
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    constraint_null = AssemblyConstraint.null_model()
    simulator = GillespieSimulator(graph, baseline_true, constraint_null, rng=np.random.default_rng(42))
    
    observed_counts = {}
    n_samples = 100  # More samples
    for _ in range(n_samples):
        traj = simulator.simulate(AssemblyState.from_parts(['A'], depth=0), t_max=50.0, burn_in=25.0)
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < 0.9:
                observed_counts[part] = observed_counts.get(part, 0) + 1
    
    if verbose:
        print(f"\nObserved counts (n={n_samples}): {observed_counts}")
    
    # FINE GRID search
    grid_values = np.linspace(-2.0, 2.0, 21)  # Finer grid
    log_liks = []
    
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    initial_state = AssemblyState.from_parts([primitives[0]], depth=0)
    
    if verbose:
        print(f"\nProfile likelihood (fine grid, n=21 points):")
        print(f"{'Value':>8s} {'Log L':>10s}")
        print("-" * 20)
    
    for value in grid_values:
        theta_test = {'reuse_count': value}
        
        constraint_test = AssemblyConstraint(feature_weights=theta_test)
        sim_test = GillespieSimulator(graph, baseline_wrong, constraint_test, rng=np.random.default_rng(None))
        
        # More samples for latent states
        latent_states = sim_test.sample_final_states(
            initial_state,
            n_samples=100,
            t_max=50.0,
            burn_in=25.0,
        )
        
        ll = obs_model.compute_log_likelihood(observed_counts, latent_states)
        log_liks.append(ll)
        
        if verbose:
            marker = ""
            if abs(value) < 0.1:
                marker = " ← NULL"
            print(f"{value:>8.2f} {ll:>10.4f}{marker}")
    
    # Analyze
    log_liks = np.array(log_liks)
    peak_idx = np.argmax(log_liks)
    peak_value = grid_values[peak_idx]
    ll_at_zero = log_liks[np.argmin(np.abs(grid_values))]
    ll_at_peak = log_liks[peak_idx]
    
    # Is peak significantly better than zero?
    delta_ll = ll_at_peak - ll_at_zero
    
    if verbose:
        print("\n" + "-" * 40)
        print("Analysis:")
        print(f"  Peak at: {peak_value:.2f}")
        print(f"  LL at peak: {ll_at_peak:.4f}")
        print(f"  LL at zero: {ll_at_zero:.4f}")
        print(f"  Δ LL (peak - zero): {delta_ll:.4f}")
        
        if abs(delta_ll) < 2.0:
            print(f"\n  ✓ Peak not significantly better than zero (Δ < 2)")
            print(f"  → No meaningful false positive")
        else:
            print(f"\n  ✗ Peak significantly better than zero (Δ = {delta_ll:.1f})")
            print(f"  → Real false positive!")
    
    return {
        'peak_value': peak_value,
        'll_at_peak': ll_at_peak,
        'll_at_zero': ll_at_zero,
        'delta_ll': delta_ll,
        'false_positive': abs(delta_ll) > 2.0,
    }


def test_replicate_stability(verbose: bool = True) -> Dict:
    """
    Test stability across replicates.
    
    Does the false positive appear consistently or is it noise?
    """
    if verbose:
        print("\n" + "=" * 80)
        print("Replicate Stability Test")
        print("=" * 80)
        print("\nQuestion: Is false positive consistent across replicates?")
    
    primitives = ['A', 'B', 'C', 'D', 'E']
    baseline_true = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    baseline_wrong = AssemblyBaseline(kappa=1.2, join_exponent=-0.6, split_exponent=0.4)
    
    n_replicates = 5
    peaks = []
    delta_lls = []
    
    if verbose:
        print(f"\nRunning {n_replicates} independent replicates...")
        print(f"{'Rep':>4s} {'Peak':>8s} {'Δ LL':>8s}")
        print("-" * 25)
    
    for rep in range(n_replicates):
        # Generate data
        graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
        constraint_null = AssemblyConstraint.null_model()
        simulator = GillespieSimulator(graph, baseline_true, constraint_null, rng=np.random.default_rng(None))
        
        observed_counts = {}
        for _ in range(80):
            traj = simulator.simulate(AssemblyState.from_parts(['A'], depth=0), t_max=50.0, burn_in=25.0)
            final_state = traj.final_state()
            for part in final_state.get_parts_list():
                if np.random.rand() < 0.9:
                    observed_counts[part] = observed_counts.get(part, 0) + 1
        
        # Profile likelihood
        grid_values = np.linspace(-1.5, 1.5, 13)
        log_liks = []
        
        obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
        initial_state = AssemblyState.from_parts([primitives[0]], depth=0)
        
        for value in grid_values:
            theta_test = {'reuse_count': value}
            constraint_test = AssemblyConstraint(feature_weights=theta_test)
            sim_test = GillespieSimulator(graph, baseline_wrong, constraint_test, rng=np.random.default_rng(None))
            
            latent_states = sim_test.sample_final_states(initial_state, n_samples=50, t_max=50.0, burn_in=25.0)
            ll = obs_model.compute_log_likelihood(observed_counts, latent_states)
            log_liks.append(ll)
        
        log_liks = np.array(log_liks)
        peak_idx = np.argmax(log_liks)
        peak_value = grid_values[peak_idx]
        ll_at_zero = log_liks[np.argmin(np.abs(grid_values))]
        delta_ll = log_liks[peak_idx] - ll_at_zero
        
        peaks.append(peak_value)
        delta_lls.append(delta_ll)
        
        if verbose:
            print(f"{rep+1:>4d} {peak_value:>8.2f} {delta_ll:>8.2f}")
    
    # Analyze consistency
    peaks = np.array(peaks)
    delta_lls = np.array(delta_lls)
    
    if verbose:
        print("\n" + "-" * 40)
        print("Consistency Analysis:")
        print(f"  Mean peak: {np.mean(peaks):.2f} ± {np.std(peaks):.2f}")
        print(f"  Mean Δ LL: {np.mean(delta_lls):.2f} ± {np.std(delta_lls):.2f}")
        
        consistent_fp = np.sum(np.abs(delta_lls) > 2.0)
        print(f"\n  False positives: {consistent_fp}/{n_replicates} replicates")
        
        if consistent_fp == 0:
            print(f"  ✓ No consistent false positives")
        elif consistent_fp < n_replicates / 2:
            print(f"  ⚠ Occasional false positives (noise)")
        else:
            print(f"  ✗ Consistent false positives (real problem)")
    
    return {
        'peaks': peaks.tolist(),
        'delta_lls': delta_lls.tolist(),
        'mean_peak': float(np.mean(peaks)),
        'std_peak': float(np.std(peaks)),
        'mean_delta_ll': float(np.mean(delta_lls)),
        'n_false_positives': int(np.sum(np.abs(delta_lls) > 2.0)),
    }


def test_baseline_sensitivity(verbose: bool = True) -> Dict:
    """
    How sensitive is false positive to baseline misspecification?
    
    Test different levels of misspecification: 5%, 10%, 20%, 50%.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("Baseline Misspecification Sensitivity")
        print("=" * 80)
        print("\nQuestion: How much misspecification causes false positives?")
    
    primitives = ['A', 'B', 'C', 'D', 'E']
    baseline_true = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    
    # Generate data once
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    constraint_null = AssemblyConstraint.null_model()
    simulator = GillespieSimulator(graph, baseline_true, constraint_null, rng=np.random.default_rng(42))
    
    observed_counts = {}
    for _ in range(100):
        traj = simulator.simulate(AssemblyState.from_parts(['A'], depth=0), t_max=50.0, burn_in=25.0)
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < 0.9:
                observed_counts[part] = observed_counts.get(part, 0) + 1
    
    # Test different levels of misspecification
    misspec_levels = [0.0, 0.05, 0.10, 0.20, 0.50]
    results = []
    
    if verbose:
        print(f"\n{'Misspec':>8s} {'Peak':>8s} {'Δ LL':>8s} {'FP?':>6s}")
        print("-" * 35)
    
    for misspec in misspec_levels:
        # Create misspecified baseline
        baseline_wrong = AssemblyBaseline(
            kappa=baseline_true.kappa * (1 + misspec),
            join_exponent=baseline_true.join_exponent * (1 + misspec),
            split_exponent=baseline_true.split_exponent * (1 + misspec),
        )
        
        # Profile likelihood
        grid_values = np.linspace(-1.5, 1.5, 13)
        log_liks = []
        
        obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
        initial_state = AssemblyState.from_parts([primitives[0]], depth=0)
        
        for value in grid_values:
            theta_test = {'reuse_count': value}
            constraint_test = AssemblyConstraint(feature_weights=theta_test)
            sim_test = GillespieSimulator(graph, baseline_wrong, constraint_test, rng=np.random.default_rng(None))
            
            latent_states = sim_test.sample_final_states(initial_state, n_samples=50, t_max=50.0, burn_in=25.0)
            ll = obs_model.compute_log_likelihood(observed_counts, latent_states)
            log_liks.append(ll)
        
        log_liks = np.array(log_liks)
        peak_idx = np.argmax(log_liks)
        peak_value = grid_values[peak_idx]
        ll_at_zero = log_liks[np.argmin(np.abs(grid_values))]
        delta_ll = log_liks[peak_idx] - ll_at_zero
        
        fp = abs(delta_ll) > 2.0
        
        results.append({
            'misspec': misspec,
            'peak': peak_value,
            'delta_ll': delta_ll,
            'false_positive': fp,
        })
        
        if verbose:
            fp_marker = "✗" if fp else "✓"
            print(f"{misspec*100:>7.0f}% {peak_value:>8.2f} {delta_ll:>8.2f} {fp_marker:>6s}")
    
    if verbose:
        print("\n" + "-" * 40)
        print("Sensitivity:")
        fp_count = sum(r['false_positive'] for r in results)
        if fp_count == 0:
            print(f"  ✓ No false positives at any level")
        else:
            fp_threshold = min(r['misspec'] for r in results if r['false_positive'])
            print(f"  ⚠ False positives start at {fp_threshold*100:.0f}% misspecification")
    
    return results


def main():
    print("=" * 80)
    print("Detailed Robustness Analysis")
    print("=" * 80)
    print("\nInvestigating false positive issue in depth...")
    print()
    
    # Test 1: Fine grid
    result1 = detailed_false_positive_test(verbose=True)
    
    # Test 2: Replicates
    result2 = test_replicate_stability(verbose=True)
    
    # Test 3: Sensitivity
    result3 = test_baseline_sensitivity(verbose=True)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n1. Fine Grid Analysis:")
    if result1['false_positive']:
        print(f"   ✗ Real false positive (Δ LL = {result1['delta_ll']:.1f})")
    else:
        print(f"   ✓ No meaningful false positive (Δ LL = {result1['delta_ll']:.1f})")
    
    print("\n2. Replicate Stability:")
    print(f"   False positives: {result2['n_false_positives']}/5 replicates")
    print(f"   Mean peak: {result2['mean_peak']:.2f} ± {result2['std_peak']:.2f}")
    
    print("\n3. Baseline Sensitivity:")
    fp_results = [r for r in result3 if r['false_positive']]
    if fp_results:
        min_misspec = min(r['misspec'] for r in fp_results)
        print(f"   False positives start at {min_misspec*100:.0f}% misspecification")
    else:
        print(f"   No false positives up to 50% misspecification")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if result1['false_positive'] or result2['n_false_positives'] > 2:
        print("\n⚠ FALSE POSITIVES ARE A REAL ISSUE")
        print("\nCauses:")
        print("  • Baseline misspecification creates systematic bias")
        print("  • Model tries to compensate with spurious constraints")
        print("\nMitigation:")
        print("  • Use conservative threshold (Δ LL > 5 instead of > 2)")
        print("  • Validate baseline before inference")
        print("  • Use robust estimation methods")
    else:
        print("\n✓ FALSE POSITIVES ARE NOISE, NOT SYSTEMATIC")
        print("\nKey findings:")
        print("  • Occasional false positives due to stochastic variation")
        print("  • Not consistent across replicates")
        print("  • Δ LL typically small (<2)")
        print("\n→ Current methods are robust enough for real data")
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
