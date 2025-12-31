"""
Demo: Assembly Observation Models

Shows how observation models handle missingness and partial observations.
"""

import sys
sys.path.insert(0, 'src')

from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.observation.presence_model import (
    PresenceObservationModel, FragmentObservationModel
)


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
    
    # 2. Presence observation model
    print("\n2. Presence Observation Model")
    print("-" * 70)
    
    presence_model = PresenceObservationModel(
        detection_prob=0.9,
        false_positive_prob=0.01,
    )
    
    print(f"  {presence_model}")
    
    # Simulate observations
    observed_compounds = {'A', 'B'}  # We observe A and B, but not C
    
    print(f"\n  Observed compounds: {observed_compounds}")
    
    # Compute likelihood
    log_lik = presence_model.compute_log_likelihood(observed_compounds, latent_states)
    print(f"  Log-likelihood: {log_lik:.4f}")
    
    # Predict presence probabilities
    print("\n  Predicted presence probabilities:")
    for compound in ['A', 'B', 'C', 'D']:
        p_obs = presence_model.predict_presence(latent_states, compound)
        print(f"    P(observe {compound}) = {p_obs:.4f}")
    
    # 3. Fragment observation model
    print("\n3. Fragment Observation Model")
    print("-" * 70)
    
    fragment_model = FragmentObservationModel(noise_level=0.1)
    print(f"  {fragment_model}")
    
    # Simulate fragment observations
    observed_fragments = {
        'A': 0.7,  # Intensity from states containing A
        'B': 0.6,  # Intensity from states containing B
        'C': 0.1,  # Small intensity from ABC state
    }
    
    print(f"\n  Observed fragments:")
    for frag, intensity in observed_fragments.items():
        print(f"    {frag}: {intensity:.2f}")
    
    # Predict expected fragments
    expected = fragment_model._predict_fragments(latent_states)
    print(f"\n  Expected fragments (from latent states):")
    for frag, intensity in expected.items():
        print(f"    {frag}: {intensity:.2f}")
    
    # Compute likelihood
    log_lik_frag = fragment_model.compute_log_likelihood(observed_fragments, latent_states)
    print(f"\n  Log-likelihood: {log_lik_frag:.4f}")
    
    # 4. Demonstrate missingness tolerance
    print("\n4. Missingness Tolerance")
    print("-" * 70)
    
    # We only observe A and B, completely missing C
    print("  Observation: We see A and B, but not C")
    print("  Latent truth: C is present in ABC state (10% probability)")
    print("\n  Presence model handles this naturally:")
    print("    - Explains what we see (A, B)")
    print("    - Doesn't penalize for not seeing C (low probability anyway)")
    print("    - Tolerates massive missingness")
    
    # 5. Compare to full observation
    print("\n5. Comparison: Partial vs Full Observation")
    print("-" * 70)
    
    # Partial observation (what we have)
    partial_obs = {'A', 'B'}
    log_lik_partial = presence_model.compute_log_likelihood(partial_obs, latent_states)
    
    # Full observation (if we saw everything)
    full_obs = {'A', 'B', 'C'}
    log_lik_full = presence_model.compute_log_likelihood(full_obs, latent_states)
    
    print(f"  Partial observation (A, B):     log-lik = {log_lik_partial:.4f}")
    print(f"  Full observation (A, B, C):     log-lik = {log_lik_full:.4f}")
    print(f"  Difference: {log_lik_full - log_lik_partial:.4f}")
    print("\n  Full observation is better (saw more), but partial is still valid!")
    
    print("\n" + "=" * 70)
    print("Observation Models Demo Complete!")
    print("=" * 70)
    
    print("\nKey Takeaways:")
    print("  ✓ Presence model: P(observe | latent states)")
    print("  ✓ Fragment model: P(fragments | latent states)")
    print("  ✓ Both tolerate missingness (only explain what we see)")
    print("  ✓ Detection probability < 1 (realistic)")
    print("  ✓ False positives handled")
    print("\nNext: Integrate with PERSISTE inference to fit constraint parameters!")


if __name__ == '__main__':
    main()
