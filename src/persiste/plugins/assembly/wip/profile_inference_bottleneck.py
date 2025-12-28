"""
Profile the inference bottleneck to confirm where time is going.

This will instrument fit_with_null() to measure:
- Time in null fit
- Time in constrained fit
- Time in profile diagnostics
- Number of likelihood evaluations
- Time per likelihood evaluation
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
from persiste.plugins.assembly.inference.robust_inference import RobustConstraintInference


# Monkey-patch to count likelihood evaluations
original_neg_log_likelihood = RobustConstraintInference.neg_log_likelihood
call_count = 0
total_time = 0.0

def instrumented_neg_log_likelihood(self, params, observed_counts, constraint_features):
    global call_count, total_time
    call_count += 1
    t0 = time.time()
    result = original_neg_log_likelihood(self, params, observed_counts, constraint_features)
    total_time += time.time() - t0
    return result

RobustConstraintInference.neg_log_likelihood = instrumented_neg_log_likelihood


def generate_test_data():
    """Generate small test dataset."""
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


def profile_inference():
    """Profile the full inference pipeline."""
    global call_count, total_time
    
    print("=" * 80)
    print("PROFILING INFERENCE BOTTLENECK")
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
    
    inference = RobustConstraintInference(
        graph=graph,
        baseline_family=baseline_family,
        obs_model=obs_model,
        regularization=0.1,
        n_latent_samples=100,
    )
    
    # Profile null fit
    print("\n[3] Profiling NULL fit...")
    call_count = 0
    total_time = 0.0
    t0 = time.time()
    
    null_params = {'reuse_count': 0.0, 'depth_change': 0.0}
    _, _, _ = inference.fit(observed_counts, ['reuse_count', 'depth_change'], initial_guess=null_params)
    
    null_time = time.time() - t0
    null_calls = call_count
    null_ll_time = total_time
    
    print(f"  Total time: {null_time:.2f}s")
    print(f"  Likelihood calls: {null_calls}")
    print(f"  Time in likelihood: {null_ll_time:.2f}s ({100*null_ll_time/null_time:.1f}%)")
    print(f"  Time per call: {null_ll_time/null_calls:.3f}s")
    
    # Profile constrained fit
    print("\n[4] Profiling CONSTRAINED fit...")
    call_count = 0
    total_time = 0.0
    t0 = time.time()
    
    _, _, _ = inference.fit(observed_counts, ['reuse_count', 'depth_change'])
    
    constrained_time = time.time() - t0
    constrained_calls = call_count
    constrained_ll_time = total_time
    
    print(f"  Total time: {constrained_time:.2f}s")
    print(f"  Likelihood calls: {constrained_calls}")
    print(f"  Time in likelihood: {constrained_ll_time:.2f}s ({100*constrained_ll_time/constrained_time:.1f}%)")
    print(f"  Time per call: {constrained_ll_time/constrained_calls:.3f}s")
    
    # Profile ONE profile diagnostic
    print("\n[5] Profiling ONE profile diagnostic (9 grid points)...")
    call_count = 0
    total_time = 0.0
    t0 = time.time()
    
    base_params = {'reuse_count': 1.0, 'depth_change': -0.5}
    grid_values = np.linspace(-2.0, 2.0, 9)
    
    for value in grid_values:
        params_test = base_params.copy()
        params_test['reuse_count'] = value
        _, _, _ = inference.fit(observed_counts, ['reuse_count', 'depth_change'], initial_guess=params_test)
    
    profile_time = time.time() - t0
    profile_calls = call_count
    profile_ll_time = total_time
    
    print(f"  Total time: {profile_time:.2f}s")
    print(f"  Likelihood calls: {profile_calls}")
    print(f"  Time in likelihood: {profile_ll_time:.2f}s ({100*profile_ll_time/profile_time:.1f}%)")
    print(f"  Time per call: {profile_ll_time/profile_calls:.3f}s")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nNull fit:        {null_time:.2f}s ({null_calls} calls)")
    print(f"Constrained fit: {constrained_time:.2f}s ({constrained_calls} calls)")
    print(f"One profile:     {profile_time:.2f}s ({profile_calls} calls)")
    print(f"\nEstimated time for fit_with_null(profile_diagnostics=True, 2 features):")
    print(f"  Null + Constrained: {null_time + constrained_time:.2f}s")
    print(f"  + 2 profiles:       {null_time + constrained_time + 2*profile_time:.2f}s")
    print(f"\nBottleneck: Each likelihood call takes ~{(null_ll_time/null_calls):.3f}s")
    print(f"            This is dominated by Gillespie simulation (n_latent_samples={inference.n_latent_samples})")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("\n1. CACHE simulation results - don't re-simulate for every θ")
    print("2. Make profile diagnostics CONDITIONAL - only if ΔLL in uncertain range")
    print("3. Reduce n_latent_samples for screening (30-50 instead of 100)")
    print("4. Use deterministic approximation for initial search")


if __name__ == '__main__':
    profile_inference()
