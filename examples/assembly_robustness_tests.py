"""
Robustness Under Mild Misspecification

Tests how badly the inference breaks under realistic violations:
1. Slightly wrong baseline (misspecified rates)
2. Missing low-frequency states (incomplete observations)
3. Noisy frequency counts (measurement error)

Expectation:
- Likelihood surfaces broaden (wider peaks)
- But don't create false positives (no spurious constraints)

This is the last validation before real data.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from typing import Dict, List
import copy

from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.observation.presence_model import FrequencyWeightedPresenceModel


def profile_likelihood_robust(
    feature_name: str,
    theta_base: Dict[str, float],
    observed_counts: Dict[str, int],
    baseline: AssemblyBaseline,
    primitives: List[str],
    max_depth: int = 5,
    verbose: bool = False,
) -> Dict:
    """
    Compute profile likelihood for robustness testing.
    
    Returns dict with ll_range, peak_value, curvature.
    """
    graph = AssemblyGraph(primitives, max_depth=max_depth, min_rate_threshold=1e-4)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    initial_state = AssemblyState.from_parts([primitives[0]], depth=0)
    
    # Grid search
    grid_values = np.linspace(-1.0, 2.5, 8)
    log_liks = []
    
    for value in grid_values:
        theta_test = theta_base.copy()
        theta_test[feature_name] = value
        
        constraint_test = AssemblyConstraint(feature_weights=theta_test)
        sim_test = GillespieSimulator(graph, baseline, constraint_test, rng=np.random.default_rng(None))
        
        latent_states = sim_test.sample_final_states(
            initial_state,
            n_samples=50,
            t_max=50.0,
            burn_in=25.0,
        )
        
        ll = obs_model.compute_log_likelihood(observed_counts, latent_states)
        log_liks.append(ll)
    
    # Analyze
    log_liks = np.array(log_liks)
    ll_range = np.max(log_liks) - np.min(log_liks)
    peak_idx = np.argmax(log_liks)
    peak_value = grid_values[peak_idx]
    
    if 0 < peak_idx < len(grid_values) - 1:
        curvature = abs(log_liks[peak_idx-1] - 2*log_liks[peak_idx] + log_liks[peak_idx+1])
    else:
        curvature = 0.0
    
    if verbose:
        print(f"\n{feature_name}:")
        print(f"  Peak at: {peak_value:.2f}")
        print(f"  LL range: {ll_range:.1f}")
        print(f"  Curvature: {curvature:.1f}")
    
    return {
        'feature': feature_name,
        'peak_value': peak_value,
        'll_range': ll_range,
        'curvature': curvature,
        'identifiable': ll_range > 10.0,
    }


def test_wrong_baseline(verbose: bool = True) -> Dict:
    """
    Test 1: Slightly wrong baseline (misspecified rates).
    
    Generate data with baseline A, infer with baseline B.
    
    Expectation:
    - Surfaces broaden (lower curvature)
    - But still identifiable (range > 10)
    - No false positives (null model still worse)
    """
    if verbose:
        print("=" * 80)
        print("Test 1: Slightly Wrong Baseline")
        print("=" * 80)
        print("\nScenario: Generate with baseline A, infer with baseline B")
    
    primitives = ['A', 'B', 'C', 'D', 'E']
    theta_true = {'reuse_count': 1.5, 'depth_change': -0.5}
    
    # TRUE baseline (data generation)
    baseline_true = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    
    # WRONG baseline (inference) - slightly different parameters
    baseline_wrong = AssemblyBaseline(kappa=1.2, join_exponent=-0.6, split_exponent=0.4)
    
    if verbose:
        print(f"\nTrue baseline:  kappa={baseline_true.kappa}, join_exp={baseline_true.join_exponent}, split_exp={baseline_true.split_exponent}")
        print(f"Wrong baseline: kappa={baseline_wrong.kappa}, join_exp={baseline_wrong.join_exponent}, split_exp={baseline_wrong.split_exponent}")
    
    # Generate data with TRUE baseline
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    constraint_true = AssemblyConstraint(feature_weights=theta_true)
    simulator = GillespieSimulator(graph, baseline_true, constraint_true, rng=np.random.default_rng(42))
    
    observed_counts = {}
    for _ in range(80):
        traj = simulator.simulate(AssemblyState.from_parts(['A'], depth=0), t_max=50.0, burn_in=25.0)
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < 0.9:
                observed_counts[part] = observed_counts.get(part, 0) + 1
    
    if verbose:
        print(f"\nObserved counts: {observed_counts}")
    
    # Profile likelihood with CORRECT baseline
    if verbose:
        print("\n--- With CORRECT baseline ---")
    result_correct = profile_likelihood_robust(
        'reuse_count', theta_true, observed_counts, baseline_true, primitives, verbose=verbose
    )
    
    # Profile likelihood with WRONG baseline
    if verbose:
        print("\n--- With WRONG baseline (20% off) ---")
    result_wrong = profile_likelihood_robust(
        'reuse_count', theta_true, observed_counts, baseline_wrong, primitives, verbose=verbose
    )
    
    # Compare
    if verbose:
        print("\n" + "-" * 40)
        print("Comparison:")
        print(f"  Correct baseline: range={result_correct['ll_range']:.1f}, curvature={result_correct['curvature']:.1f}")
        print(f"  Wrong baseline:   range={result_wrong['ll_range']:.1f}, curvature={result_wrong['curvature']:.1f}")
        
        broadening = result_correct['curvature'] / max(result_wrong['curvature'], 1.0)
        print(f"\n  Surface broadening: {broadening:.2f}x")
        
        if result_wrong['identifiable']:
            print(f"  ✓ Still identifiable despite misspecification")
        else:
            print(f"  ✗ Lost identifiability")
    
    return {
        'test': 'wrong_baseline',
        'correct': result_correct,
        'wrong': result_wrong,
        'broadening': result_correct['curvature'] / max(result_wrong['curvature'], 1.0),
        'still_identifiable': result_wrong['identifiable'],
    }


def test_missing_states(verbose: bool = True) -> Dict:
    """
    Test 2: Missing low-frequency states (incomplete observations).
    
    Remove states that appear <5% of the time.
    
    Expectation:
    - Surfaces broaden slightly
    - Still identifiable
    - No false positives
    """
    if verbose:
        print("\n" + "=" * 80)
        print("Test 2: Missing Low-Frequency States")
        print("=" * 80)
        print("\nScenario: Remove compounds observed <5% of the time")
    
    primitives = ['A', 'B', 'C', 'D', 'E']
    theta_true = {'reuse_count': 1.5, 'depth_change': -0.5}
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    
    # Generate data
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    constraint_true = AssemblyConstraint(feature_weights=theta_true)
    simulator = GillespieSimulator(graph, baseline, constraint_true, rng=np.random.default_rng(42))
    
    observed_counts_full = {}
    n_samples = 80
    for _ in range(n_samples):
        traj = simulator.simulate(AssemblyState.from_parts(['A'], depth=0), t_max=50.0, burn_in=25.0)
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < 0.9:
                observed_counts_full[part] = observed_counts_full.get(part, 0) + 1
    
    # Remove low-frequency states (<5% = <4 observations out of 80)
    threshold = int(0.05 * n_samples)
    observed_counts_filtered = {
        k: v for k, v in observed_counts_full.items() if v >= threshold
    }
    
    if verbose:
        print(f"\nFull observations: {len(observed_counts_full)} compounds")
        print(f"Filtered (≥{threshold} obs): {len(observed_counts_filtered)} compounds")
        print(f"Removed: {len(observed_counts_full) - len(observed_counts_filtered)} low-frequency compounds")
    
    # Profile likelihood with FULL data
    if verbose:
        print("\n--- With FULL data ---")
    result_full = profile_likelihood_robust(
        'reuse_count', theta_true, observed_counts_full, baseline, primitives, verbose=verbose
    )
    
    # Profile likelihood with FILTERED data
    if verbose:
        print("\n--- With FILTERED data (missing rare states) ---")
    result_filtered = profile_likelihood_robust(
        'reuse_count', theta_true, observed_counts_filtered, baseline, primitives, verbose=verbose
    )
    
    # Compare
    if verbose:
        print("\n" + "-" * 40)
        print("Comparison:")
        print(f"  Full data:     range={result_full['ll_range']:.1f}, curvature={result_full['curvature']:.1f}")
        print(f"  Filtered data: range={result_filtered['ll_range']:.1f}, curvature={result_filtered['curvature']:.1f}")
        
        if result_filtered['identifiable']:
            print(f"  ✓ Still identifiable despite missing states")
        else:
            print(f"  ✗ Lost identifiability")
    
    return {
        'test': 'missing_states',
        'full': result_full,
        'filtered': result_filtered,
        'n_removed': len(observed_counts_full) - len(observed_counts_filtered),
        'still_identifiable': result_filtered['identifiable'],
    }


def test_noisy_counts(verbose: bool = True) -> Dict:
    """
    Test 3: Noisy frequency counts (measurement error).
    
    Add Poisson noise to observed counts.
    
    Expectation:
    - Surfaces broaden
    - Still identifiable with moderate noise
    - No false positives
    """
    if verbose:
        print("\n" + "=" * 80)
        print("Test 3: Noisy Frequency Counts")
        print("=" * 80)
        print("\nScenario: Add Poisson measurement noise to counts")
    
    primitives = ['A', 'B', 'C', 'D', 'E']
    theta_true = {'reuse_count': 1.5, 'depth_change': -0.5}
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    
    # Generate data
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    constraint_true = AssemblyConstraint(feature_weights=theta_true)
    simulator = GillespieSimulator(graph, baseline, constraint_true, rng=np.random.default_rng(42))
    
    observed_counts_true = {}
    for _ in range(80):
        traj = simulator.simulate(AssemblyState.from_parts(['A'], depth=0), t_max=50.0, burn_in=25.0)
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < 0.9:
                observed_counts_true[part] = observed_counts_true.get(part, 0) + 1
    
    # Add Poisson noise: count' ~ Poisson(count)
    observed_counts_noisy = {}
    for compound, count in observed_counts_true.items():
        noisy_count = np.random.poisson(count)
        if noisy_count > 0:  # Keep only positive counts
            observed_counts_noisy[compound] = noisy_count
    
    if verbose:
        total_true = sum(observed_counts_true.values())
        total_noisy = sum(observed_counts_noisy.values())
        print(f"\nTrue counts:  {total_true} total")
        print(f"Noisy counts: {total_noisy} total (Poisson noise)")
        print(f"Relative error: {abs(total_noisy - total_true) / total_true * 100:.1f}%")
    
    # Profile likelihood with TRUE counts
    if verbose:
        print("\n--- With TRUE counts ---")
    result_true = profile_likelihood_robust(
        'reuse_count', theta_true, observed_counts_true, baseline, primitives, verbose=verbose
    )
    
    # Profile likelihood with NOISY counts
    if verbose:
        print("\n--- With NOISY counts (Poisson error) ---")
    result_noisy = profile_likelihood_robust(
        'reuse_count', theta_true, observed_counts_noisy, baseline, primitives, verbose=verbose
    )
    
    # Compare
    if verbose:
        print("\n" + "-" * 40)
        print("Comparison:")
        print(f"  True counts:  range={result_true['ll_range']:.1f}, curvature={result_true['curvature']:.1f}")
        print(f"  Noisy counts: range={result_noisy['ll_range']:.1f}, curvature={result_noisy['curvature']:.1f}")
        
        if result_noisy['identifiable']:
            print(f"  ✓ Still identifiable despite measurement noise")
        else:
            print(f"  ✗ Lost identifiability")
    
    return {
        'test': 'noisy_counts',
        'true': result_true,
        'noisy': result_noisy,
        'relative_error': abs(sum(observed_counts_noisy.values()) - sum(observed_counts_true.values())) / sum(observed_counts_true.values()),
        'still_identifiable': result_noisy['identifiable'],
    }


def test_false_positives(verbose: bool = True) -> Dict:
    """
    Critical test: Do misspecifications create false positives?
    
    Generate data with NULL model, infer with misspecified baseline.
    
    Expectation:
    - Should NOT recover spurious constraints
    - Null model should still be best
    """
    if verbose:
        print("\n" + "=" * 80)
        print("Test 4: False Positives Under Misspecification")
        print("=" * 80)
        print("\nScenario: Null model + wrong baseline → spurious constraints?")
    
    primitives = ['A', 'B', 'C', 'D', 'E']
    theta_null = {}  # NULL model
    
    # TRUE baseline (data generation)
    baseline_true = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    
    # WRONG baseline (inference)
    baseline_wrong = AssemblyBaseline(kappa=1.3, join_exponent=-0.7, split_exponent=0.5)
    
    # Generate data with NULL model
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    constraint_null = AssemblyConstraint.null_model()
    simulator = GillespieSimulator(graph, baseline_true, constraint_null, rng=np.random.default_rng(42))
    
    observed_counts = {}
    for _ in range(80):
        traj = simulator.simulate(AssemblyState.from_parts(['A'], depth=0), t_max=50.0, burn_in=25.0)
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < 0.9:
                observed_counts[part] = observed_counts.get(part, 0) + 1
    
    if verbose:
        print(f"\nData generated with NULL model (no constraints)")
        print(f"Observed counts: {observed_counts}")
    
    # Profile likelihood with WRONG baseline
    if verbose:
        print("\n--- Inferring with WRONG baseline ---")
    result = profile_likelihood_robust(
        'reuse_count', theta_null, observed_counts, baseline_wrong, primitives, verbose=verbose
    )
    
    # Check if peak is at zero (no false positive)
    false_positive = abs(result['peak_value']) > 0.5
    
    if verbose:
        print("\n" + "-" * 40)
        print("False Positive Test:")
        print(f"  Peak at: {result['peak_value']:.2f}")
        print(f"  Expected: ~0.0 (null model)")
        
        if not false_positive:
            print(f"  ✓ No false positive (peak near zero)")
        else:
            print(f"  ✗ FALSE POSITIVE detected! (peak at {result['peak_value']:.2f})")
    
    return {
        'test': 'false_positives',
        'result': result,
        'false_positive': false_positive,
    }


def main():
    print("=" * 80)
    print("Robustness Under Mild Misspecification")
    print("=" * 80)
    print("\nTesting how badly inference breaks under realistic violations:")
    print("  1. Slightly wrong baseline (20% parameter error)")
    print("  2. Missing low-frequency states (<5% observations)")
    print("  3. Noisy frequency counts (Poisson measurement error)")
    print("  4. False positives (null model + misspecification)")
    print()
    
    results = {}
    
    # Test 1: Wrong baseline
    results['wrong_baseline'] = test_wrong_baseline(verbose=True)
    
    # Test 2: Missing states
    results['missing_states'] = test_missing_states(verbose=True)
    
    # Test 3: Noisy counts
    results['noisy_counts'] = test_noisy_counts(verbose=True)
    
    # Test 4: False positives
    results['false_positives'] = test_false_positives(verbose=True)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Robustness Under Misspecification")
    print("=" * 80)
    
    print("\n1. Wrong Baseline (20% parameter error):")
    wb = results['wrong_baseline']
    print(f"   Broadening: {wb['broadening']:.2f}x")
    print(f"   Still identifiable: {wb['still_identifiable']}")
    
    print("\n2. Missing Low-Frequency States:")
    ms = results['missing_states']
    print(f"   States removed: {ms['n_removed']}")
    print(f"   Still identifiable: {ms['still_identifiable']}")
    
    print("\n3. Noisy Counts (Poisson error):")
    nc = results['noisy_counts']
    print(f"   Relative error: {nc['relative_error']*100:.1f}%")
    print(f"   Still identifiable: {nc['still_identifiable']}")
    
    print("\n4. False Positives:")
    fp = results['false_positives']
    print(f"   False positive detected: {fp['false_positive']}")
    
    # Overall assessment
    print("\n" + "=" * 80)
    print("OVERALL ASSESSMENT")
    print("=" * 80)
    
    all_identifiable = (
        wb['still_identifiable'] and
        ms['still_identifiable'] and
        nc['still_identifiable']
    )
    
    no_false_positives = not fp['false_positive']
    
    if all_identifiable and no_false_positives:
        print("\n✓ ROBUST TO MILD MISSPECIFICATION")
        print("\nKey findings:")
        print("  • Surfaces broaden under misspecification (expected)")
        print("  • But remain identifiable (range > 10)")
        print("  • No false positives (null model stays null)")
        print("\n→ Ready for real data!")
    else:
        print("\n⚠ SENSITIVITY TO MISSPECIFICATION")
        if not all_identifiable:
            print("  • Lost identifiability under some conditions")
        if not no_false_positives:
            print("  • FALSE POSITIVES detected!")
        print("\n→ Need more robust inference methods")
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
