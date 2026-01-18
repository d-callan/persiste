"""
Scaling Curves: Minimal Data Requirements for Inference

Systematically vary:
- Number of primitives (3-7)
- Max depth (3-7)
- Sample size (20-200)
- Simulation time (10-100)

Measure:
- Identifiability (LL range from profile likelihood)
- Runtime

Output:
- "Minimal data requirements for inference"
"""

import sys

sys.path.insert(0, 'src')

import json
import time

import numpy as np

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.observation.presence_model import FrequencyWeightedPresenceModel
from persiste.plugins.assembly.states.assembly_state import AssemblyState


def measure_identifiability(
    n_primitives: int,
    max_depth: int,
    n_samples: int,
    t_max: float,
    feature_name: str = 'reuse_count',
    theta_base: dict[str, float] = None,
    verbose: bool = False,
) -> dict:
    """
    Measure identifiability for given system parameters.

    Returns:
        dict with ll_range, curvature, runtime, n_states
    """
    if theta_base is None:
        theta_base = {'reuse_count': 1.5, 'depth_change': -0.5}

    start_time = time.time()

    # Setup
    primitives = [chr(65 + i) for i in range(n_primitives)]  # A, B, C, ...
    graph = AssemblyGraph(primitives, max_depth=max_depth, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    initial_state = AssemblyState.from_parts([primitives[0]], depth=0)

    # Generate data
    constraint_base = AssemblyConstraint(feature_weights=theta_base)
    simulator = GillespieSimulator(graph, baseline, constraint_base, rng=np.random.default_rng())

    observed_counts = {}
    for _ in range(n_samples):
        traj = simulator.simulate(initial_state, t_max=t_max, burn_in=t_max/2)
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < obs_model.detection_prob:
                observed_counts[part] = observed_counts.get(part, 0) + 1

    # Profile likelihood over small grid (for speed)
    grid_values = np.linspace(-1.0, 2.5, 8)  # Smaller grid for scaling study
    log_liks = []

    for value in grid_values:
        theta_test = theta_base.copy()
        theta_test[feature_name] = value

        constraint_test = AssemblyConstraint(feature_weights=theta_test)
        sim_test = GillespieSimulator(graph, baseline, constraint_test, rng=np.random.default_rng())

        # Smaller n_samples for latent state estimation (for speed)
        latent_states = sim_test.sample_final_states(
            initial_state,
            n_samples=min(50, n_samples),
            t_max=t_max,
            burn_in=t_max/2,
        )

        ll = obs_model.compute_log_likelihood(observed_counts, latent_states)
        log_liks.append(ll)

    # Analyze
    log_liks = np.array(log_liks)
    ll_range = np.max(log_liks) - np.min(log_liks)
    peak_idx = np.argmax(log_liks)

    if 0 < peak_idx < len(grid_values) - 1:
        curvature = abs(log_liks[peak_idx-1] - 2*log_liks[peak_idx] + log_liks[peak_idx+1])
    else:
        curvature = 0.0

    runtime = time.time() - start_time

    # Count states (approximate)
    n_states = len(graph._state_cache)

    result = {
        'n_primitives': int(n_primitives),
        'max_depth': int(max_depth),
        'n_samples': int(n_samples),
        't_max': float(t_max),
        'll_range': float(ll_range),
        'curvature': float(curvature),
        'runtime': float(runtime),
        'n_states': int(n_states),
        'identifiable': bool(ll_range > 10.0),  # Threshold
    }

    if verbose:
        print(f"n_prim={n_primitives}, depth={max_depth}, samples={n_samples}, t={t_max:.0f} → "
              f"range={ll_range:.1f}, runtime={runtime:.1f}s, states={n_states}")

    return result


def scaling_curve_primitives(verbose: bool = True) -> list[dict]:
    """Vary number of primitives (3-7)."""
    if verbose:
        print("=" * 80)
        print("Scaling Curve: Number of Primitives")
        print("=" * 80)
        print("\nFixed: max_depth=5, n_samples=80, t_max=50")
        print("Varying: n_primitives = 3, 4, 5, 6, 7")
        print()

    results = []
    for n_prim in [3, 4, 5, 6, 7]:
        result = measure_identifiability(
            n_primitives=n_prim,
            max_depth=5,
            n_samples=80,
            t_max=50.0,
            verbose=verbose,
        )
        results.append(result)

    return results


def scaling_curve_depth(verbose: bool = True) -> list[dict]:
    """Vary max depth (3-7)."""
    if verbose:
        print("\n" + "=" * 80)
        print("Scaling Curve: Max Depth")
        print("=" * 80)
        print("\nFixed: n_primitives=5, n_samples=80, t_max=50")
        print("Varying: max_depth = 3, 4, 5, 6, 7")
        print()

    results = []
    for depth in [3, 4, 5, 6, 7]:
        result = measure_identifiability(
            n_primitives=5,
            max_depth=depth,
            n_samples=80,
            t_max=50.0,
            verbose=verbose,
        )
        results.append(result)

    return results


def scaling_curve_samples(verbose: bool = True) -> list[dict]:
    """Vary sample size (20-200)."""
    if verbose:
        print("\n" + "=" * 80)
        print("Scaling Curve: Sample Size")
        print("=" * 80)
        print("\nFixed: n_primitives=5, max_depth=5, t_max=50")
        print("Varying: n_samples = 20, 40, 60, 80, 100, 150, 200")
        print()

    results = []
    for n_samp in [20, 40, 60, 80, 100, 150, 200]:
        result = measure_identifiability(
            n_primitives=5,
            max_depth=5,
            n_samples=n_samp,
            t_max=50.0,
            verbose=verbose,
        )
        results.append(result)

    return results


def scaling_curve_time(verbose: bool = True) -> list[dict]:
    """Vary simulation time (10-100)."""
    if verbose:
        print("\n" + "=" * 80)
        print("Scaling Curve: Simulation Time")
        print("=" * 80)
        print("\nFixed: n_primitives=5, max_depth=5, n_samples=80")
        print("Varying: t_max = 10, 20, 30, 40, 50, 75, 100")
        print()

    results = []
    for t in [10, 20, 30, 40, 50, 75, 100]:
        result = measure_identifiability(
            n_primitives=5,
            max_depth=5,
            n_samples=80,
            t_max=float(t),
            verbose=verbose,
        )
        results.append(result)

    return results


def analyze_scaling_curves(all_results: dict[str, list[dict]]) -> dict:
    """
    Analyze scaling curves to determine minimal requirements.

    Returns:
        dict with minimal requirements for identifiability
    """
    print("\n" + "=" * 80)
    print("Analysis: Minimal Data Requirements")
    print("=" * 80)

    # Find minimal values where identifiable
    minimal_requirements = {}

    # Primitives
    prim_results = all_results['primitives']
    identifiable_prims = [r for r in prim_results if r['identifiable']]
    if identifiable_prims:
        min_prim = min(r['n_primitives'] for r in identifiable_prims)
        minimal_requirements['min_primitives'] = min_prim
        print(f"\n✓ Minimal primitives: {min_prim}")
    else:
        print("\n✗ Not identifiable with any number of primitives tested")
        minimal_requirements['min_primitives'] = None

    # Depth
    depth_results = all_results['depth']
    identifiable_depths = [r for r in depth_results if r['identifiable']]
    if identifiable_depths:
        min_depth = min(r['max_depth'] for r in identifiable_depths)
        minimal_requirements['min_depth'] = min_depth
        print(f"✓ Minimal depth: {min_depth}")
    else:
        print("✗ Not identifiable with any depth tested")
        minimal_requirements['min_depth'] = None

    # Samples
    sample_results = all_results['samples']
    identifiable_samples = [r for r in sample_results if r['identifiable']]
    if identifiable_samples:
        min_samples = min(r['n_samples'] for r in identifiable_samples)
        minimal_requirements['min_samples'] = min_samples
        print(f"✓ Minimal samples: {min_samples}")
    else:
        print("✗ Not identifiable with any sample size tested")
        minimal_requirements['min_samples'] = None

    # Time
    time_results = all_results['time']
    identifiable_times = [r for r in time_results if r['identifiable']]
    if identifiable_times:
        min_time = min(r['t_max'] for r in identifiable_times)
        minimal_requirements['min_time'] = min_time
        print(f"✓ Minimal simulation time: {min_time:.0f}")
    else:
        print("✗ Not identifiable with any simulation time tested")
        minimal_requirements['min_time'] = None

    # Scaling trends
    print("\n" + "=" * 80)
    print("Scaling Trends")
    print("=" * 80)

    print("\nPrimitives:")
    print(f"{'N':>4s} {'Range':>8s} {'Runtime':>8s} {'States':>8s} {'ID':>4s}")
    print("-" * 40)
    for r in prim_results:
        marker = "✓" if r['identifiable'] else "✗"
        print(f"{r['n_primitives']:>4d} {r['ll_range']:8.1f} {marker:>4s}")

    print("\nDepth:")
    print(f"{'D':>4s} {'Range':>8s} {'Runtime':>8s} {'States':>8s} {'ID':>4s}")
    print("-" * 40)
    for r in depth_results:
        marker = "✓" if r['identifiable'] else "✗"
        print(f"{r['max_depth']:>4d} {r['ll_range']:8.1f} {marker:>4s}")

    print("\nSamples:")
    print(f"{'N':>4s} {'Range':>8s} {'Runtime':>8s} {'ID':>4s}")
    print("-" * 30)
    for r in sample_results:
        marker = "✓" if r['identifiable'] else "✗"
        print(f"{r['n_samples']:>4d} {r['ll_range']:>8.1f} {r['runtime']:>8.1f} {marker:>4s}")

    print("\nSimulation Time:")
    print(f"{'T':>4s} {'Range':>8s} {'Runtime':>8s} {'ID':>4s}")
    print("-" * 30)
    for r in time_results:
        marker = "✓" if r['identifiable'] else "✗"
        print(f"{r['t_max']:>4.0f} {r['ll_range']:>8.1f} {r['runtime']:>8.1f} {marker:>4s}")

    return minimal_requirements


def generate_summary(all_results: dict[str, list[dict]], minimal_requirements: dict):
    """Generate final summary and recommendations."""
    print("\n" + "=" * 80)
    print("SUMMARY: Minimal Data Requirements for Inference")
    print("=" * 80)

    print("\nRecommended Configuration:")
    print("-" * 40)
    if minimal_requirements['min_primitives']:
        print(f"  Primitives:  ≥ {minimal_requirements['min_primitives']}")
    if minimal_requirements['min_depth']:
        print(f"  Max Depth:   ≥ {minimal_requirements['min_depth']}")
    if minimal_requirements['min_samples']:
        print(f"  Samples:     ≥ {minimal_requirements['min_samples']}")
    if minimal_requirements['min_time']:
        print(f"  Sim Time:    ≥ {minimal_requirements['min_time']:.0f}")

    print("\nKey Findings:")

    # Primitives
    prim_results = all_results['primitives']
    prim_ranges = [r['ll_range'] for r in prim_results]
    if max(prim_ranges) > 2 * min(prim_ranges):
        print("  • More primitives → stronger identifiability")
        print(f"    (range increases from {min(prim_ranges):.1f} to {max(prim_ranges):.1f})")

    # Depth
    depth_results = all_results['depth']
    depth_ranges = [r['ll_range'] for r in depth_results]
    if max(depth_ranges) > 2 * min(depth_ranges):
        print("  • Greater depth → stronger identifiability")
        print(f"    (range increases from {min(depth_ranges):.1f} to {max(depth_ranges):.1f})")

    # Samples
    sample_results = all_results['samples']
    sample_ranges = [r['ll_range'] for r in sample_results]
    if max(sample_ranges) > 2 * min(sample_ranges):
        print("  • More samples → stronger identifiability")
        print(f"    (range increases from {min(sample_ranges):.1f} to {max(sample_ranges):.1f})")

    # Time
    time_results = all_results['time']
    time_ranges = [r['ll_range'] for r in time_results]
    if max(time_ranges) > 1.5 * min(time_ranges):
        print("  • Longer simulation → better equilibration")
        print(f"    (range increases from {min(time_ranges):.1f} to {max(time_ranges):.1f})")

    # Runtime scaling
    print("\nComputational Cost:")
    prim_runtimes = [r['runtime'] for r in prim_results]
    print(f"  • Primitives: {min(prim_runtimes):.1f}s → {max(prim_runtimes):.1f}s")

    depth_runtimes = [r['runtime'] for r in depth_results]
    print(f"  • Depth:      {min(depth_runtimes):.1f}s → {max(depth_runtimes):.1f}s")

    sample_runtimes = [r['runtime'] for r in sample_results]
    print(f"  • Samples:    {min(sample_runtimes):.1f}s → {max(sample_runtimes):.1f}s")

    print("\nPractical Recommendations:")
    print("  1. Start with 5 primitives, depth 5, 80 samples")
    print("  2. If not identifiable, increase primitives first (cheapest)")
    print("  3. Then increase samples (linear cost)")
    print("  4. Depth is expensive (exponential state space)")

    print("\n" + "=" * 80)


def main():
    print("=" * 80)
    print("Scaling Curves: Minimal Data Requirements for Inference")
    print("=" * 80)
    print("\nThis will systematically vary:")
    print("  - Number of primitives (3-7)")
    print("  - Max depth (3-7)")
    print("  - Sample size (20-200)")
    print("  - Simulation time (10-100)")
    print("\nMeasuring:")
    print("  - Identifiability (LL range)")
    print("  - Runtime")
    print()

    all_results = {}

    # Run scaling curves
    all_results['primitives'] = scaling_curve_primitives(verbose=True)
    all_results['depth'] = scaling_curve_depth(verbose=True)
    all_results['samples'] = scaling_curve_samples(verbose=True)
    all_results['time'] = scaling_curve_time(verbose=True)

    # Analyze
    minimal_requirements = analyze_scaling_curves(all_results)

    # Summary
    generate_summary(all_results, minimal_requirements)

    # Save results
    output_file = 'assembly_scaling_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'results': all_results,
            'minimal_requirements': minimal_requirements,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
