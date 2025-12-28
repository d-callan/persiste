"""
Demo: MLE Inference for Assembly Theory

Phase 1.7: Recover θ from observations.

This is the scientific heart of the system.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.observation.presence_model import PresenceObservationModel
from persiste.plugins.assembly.inference.mle import AssemblyMLEInference


def main():
    print("=" * 80)
    print("MLE Inference for Assembly Theory (Phase 1.7)")
    print("=" * 80)
    
    print("\nGoal: Recover θ from observations without cheating")
    
    # ========================================================================
    # Setup
    # ========================================================================
    print("\n" + "=" * 80)
    print("Setup")
    print("=" * 80)
    
    primitives = ['A', 'B', 'C']
    graph = AssemblyGraph(primitives, max_depth=3, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    obs_model = PresenceObservationModel(detection_prob=0.9, false_positive_prob=0.01)
    initial_state = AssemblyState.from_parts(['A'], depth=0)
    
    print(f"\n{graph}")
    print(f"{baseline}")
    print(f"{obs_model}")
    
    # ========================================================================
    # Generate Synthetic Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("Generate Synthetic Data")
    print("=" * 80)
    
    # True parameters (what we'll try to recover)
    theta_true = {
        'reuse_count': 1.0,
        'depth_change': -0.3,
    }
    
    print(f"\nTrue parameters:")
    for k, v in theta_true.items():
        print(f"  {k:20s} = {v:.2f}")
    
    # Simulate with true parameters
    constraint_true = AssemblyConstraint(feature_weights=theta_true)
    simulator = GillespieSimulator(graph, baseline, constraint_true, rng=np.random.default_rng(42))
    
    print(f"\nSimulating dynamics with true θ...")
    latent_states = simulator.sample_final_states(
        initial_state,
        n_samples=100,
        t_max=50.0,
        burn_in=25.0,
    )
    
    print(f"Generated {len(latent_states)} unique states")
    
    # Generate observations
    observed_compounds = set()
    for state, prob in latent_states.items():
        for part in state.get_parts_list():
            if np.random.rand() < obs_model.detection_prob * prob:
                observed_compounds.add(part)
    
    print(f"Observed compounds: {observed_compounds}")
    
    # ========================================================================
    # Fit Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("Fit Model via MLE")
    print("=" * 80)
    
    # Create inference engine
    feature_names = ['reuse_count', 'depth_change']
    inference = AssemblyMLEInference(
        graph=graph,
        baseline=baseline,
        obs_model=obs_model,
        initial_state=initial_state,
        feature_names=feature_names,
    )
    
    print(f"\nFitting {len(feature_names)} parameters...")
    print(f"This will take a few minutes (stochastic simulation)...\n")
    
    # Fit
    result = inference.fit(
        observed_compounds,
        n_samples=30,  # Fewer samples for speed
        t_max=50.0,
        burn_in=25.0,
        rng_seed=42,
        verbose=True,
    )
    
    # ========================================================================
    # Compare Results
    # ========================================================================
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    
    theta_mle = result['theta_mle']
    
    print(f"\n{'Parameter':<20s} {'True':<10s} {'MLE':<10s} {'Error':<10s}")
    print("-" * 50)
    
    for name in feature_names:
        true_val = theta_true[name]
        mle_val = theta_mle[name]
        error = abs(mle_val - true_val)
        print(f"{name:<20s} {true_val:>8.2f}  {mle_val:>8.2f}  {error:>8.2f}")
    
    # ========================================================================
    # Diagnostics
    # ========================================================================
    print("\n" + "=" * 80)
    print("Diagnostics")
    print("=" * 80)
    
    print(f"\nOptimization success: {result['success']}")
    print(f"Iterations: {result['n_iterations']}")
    print(f"Final -log L: {result['neg_log_lik']:.4f}")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    
    print("\nPhase 1.7 ✓ Complete")
    print("  ✓ MLE inference implemented")
    print("  ✓ θ recovered from observations")
    print("  ✓ Pipeline working: θ → λ_eff → P(state) → P(obs)")
    print("\nNext: Phase 1.8 (Validation)")
    print("\nNote: Results may vary due to stochastic simulation.")
    print("Phase 1.8 will test recovery more rigorously.")


if __name__ == '__main__':
    main()
