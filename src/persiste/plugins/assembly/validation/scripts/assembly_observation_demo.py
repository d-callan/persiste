"""
Demo: Assembly Observation Models

Shows how observation models handle missingness and partial observations.
"""

import sys

sys.path.insert(0, 'src')

from persiste.plugins.assembly.observation.counts_model import AssemblyCountsModel
from persiste.plugins.assembly.states.assembly_state import AssemblyState


def main():
    print("=" * 70)
    print("Assembly Observation Models Demo")
    print("=" * 70)

    # 1. Create latent state distribution
    print("\n1. Latent State Distribution (Ground Truth)")
    print("-" * 70)

    # Imagine we have these states with these probabilities
    latent_states = {
        AssemblyState.from_parts(['A'], depth=0): 0.3,
        AssemblyState.from_parts(['B'], depth=0): 0.2,
        AssemblyState.from_parts(['A', 'B'], depth=1): 0.4,
        AssemblyState.from_parts(['A', 'B', 'C'], depth=2): 0.1,
    }

    print("  Latent states (what's actually there):")
    for state, prob in latent_states.items():
        print(f"    {prob:.1f} - {state}")

    # 2. Assembly counts observation model
    print("\n2. Assembly Counts Observation Model")
    print("-" * 70)

    counts_model = AssemblyCountsModel(
        detection_efficiency=0.9,
        background_noise=0.01,
    )

    print(f"  {counts_model}")

    # Imagine we use stable IDs for the states
    state_a = list(latent_states.keys())[0]
    state_b = list(latent_states.keys())[1]
    
    # Simulate counts
    observed_counts = {state_a.stable_id: 10, state_b.stable_id: 5}

    print(f"\n  Observed counts (by ID): {observed_counts}")

    # Convert latent states to IDs for the model
    latent_ids = {s.stable_id: p for s, p in latent_states.items()}

    # Compute likelihood
    log_lik = counts_model.compute_log_likelihood(observed_counts, latent_ids)
    print(f"  Log-likelihood: {log_lik:.4f}")

    # 3. Constraint Diagnostics
    print("\n3. Constraint Diagnostics")
    print("-" * 70)
    
    # Create a dummy graph for diagnostic lookups
    from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
    graph = AssemblyGraph(['A', 'B', 'C'], max_depth=5)
    counts_model.graph = graph
    
    # Register states in graph for diagnostics to work
    for state in latent_states:
        graph.register_state(state)

    diagnostics = counts_model.get_constraint_diagnostics(latent_ids)
    print("  Structural features inferred from occupancy:")
    for key, val in diagnostics.items():
        print(f"    {key}: {val:.4f}")

    # 4. Demonstrate missingness tolerance
    print("\n4. Missingness Tolerance")
    print("-" * 70)

    # We only observe A and B, completely missing C
    print("  Observation: We see A and B, but not C")
    print("  Latent truth: C is present in ABC state (10% probability)")
    print("\n  Counts model handles this naturally:")
    print("    - Explains what we see (A, B)")
    print("    - Doesn't penalize for not seeing C (low probability anyway)")
    print("    - Tolerates massive missingness")

    # 5. Comparison: Partial vs Full Observation
    print("\n5. Comparison: Partial vs Full Observation")
    print("-" * 70)

    # Partial observation (what we have)
    log_lik_partial = counts_model.compute_log_likelihood(observed_counts, latent_ids)

    # Full observation (if we saw everything)
    full_counts = observed_counts.copy()
    abc_state = list(latent_states.keys())[3]
    full_counts[abc_state.stable_id] = 2
    log_lik_full = counts_model.compute_log_likelihood(full_counts, latent_ids)

    print(f"  Partial observation (A, B):     log-lik = {log_lik_partial:.4f}")
    print(f"  Full observation (A, B, C):     log-lik = {log_lik_full:.4f}")
    print(f"  Difference: {log_lik_full - log_lik_partial:.4f}")
    print("\n  Full observation is better (saw more), but partial is still valid!")

    print("\n" + "=" * 70)
    print("Observation Models Demo Complete!")
    print("=" * 70)

    print("\nKey Takeaways:")
    print("  ✓ Counts model: P(counts | latent states)")
    print("  ✓ Structural diagnostics: Extract reuse/depth features from occupancy")
    print("  ✓ Tolerates missingness (only explain what we see)")
    print("  ✓ Detection efficiency < 1 (realistic)")
    print("  ✓ Background noise handled")
    print("\nNext: Integrate with PERSISTE inference to fit constraint parameters using counts!")


if __name__ == '__main__':
    main()
