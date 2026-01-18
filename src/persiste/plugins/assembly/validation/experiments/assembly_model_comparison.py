"""
Demo: Assembly Theory Model Comparison

Shows the three-layer architecture:
- Layer 0: PERSISTE core (math)
- Layer 1: Assembly mechanics (features)
- Layer 2: Assembly theories (constraints)

Compares:
1. Null model (θ = {})
2. Reuse-only model (θ = {'reuse_count': 1.0})
3. Assembly theory model (θ = {'reuse_count': 1.0, 'depth_change': -0.3})

This is science, not philosophy:
- We test hypotheses, not assert them
- We compare models, not assume one is right
- We let data decide which features matter
"""

import sys

sys.path.insert(0, 'src')

import numpy as np

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline, TransitionType
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.features.assembly_features import AssemblyFeatureExtractor
from persiste.plugins.assembly.states.assembly_state import AssemblyState


def main():
    print("=" * 80)
    print("Assembly Theory Model Comparison")
    print("=" * 80)

    print("\nThree-Layer Architecture:")
    print("  Layer 0: PERSISTE core (mathematical machinery)")
    print("  Layer 1: Assembly mechanics (features, not theories)")
    print("  Layer 2: Assembly theories (user-defined constraints)")

    # ========================================================================
    # Layer 1: Feature Extraction (Mechanics)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Layer 1: Feature Extraction (Hypothesis-Neutral)")
    print("=" * 80)

    feature_extractor = AssemblyFeatureExtractor()
    print(f"\n{feature_extractor}")
    print("\nAvailable features:")
    for feature in feature_extractor.get_feature_names():
        print(f"  - {feature}")

    # Example transition
    state_AB = AssemblyState.from_parts(['A', 'B'], depth=1)
    state_ABC = AssemblyState.from_parts(['A', 'B', 'C'], depth=2, motifs={'helix'})

    print(f"\nExample transition: {state_AB} → {state_ABC}")

    features = feature_extractor.extract_features(state_AB, state_ABC, TransitionType.JOIN)
    print("\nExtracted features:")
    for name, value in features.to_dict().items():
        print(f"  {name:25s} = {value:.3f}")

    print("\n→ Features are observables, not value judgments")
    print("→ Layer 1 doesn't say whether reuse is 'good' or 'bad'")

    # ========================================================================
    # Layer 2: Constraint Models (Theories)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Layer 2: Constraint Models (Hypothesis-Driven)")
    print("=" * 80)

    # Define three models
    models = {
        "Null": AssemblyConstraint.null_model(),
        "Reuse-only": AssemblyConstraint.reuse_only(reuse_weight=1.0),
        "Assembly Theory": AssemblyConstraint.assembly_theory(reuse=1.0, depth_penalty=-0.3),
    }

    print("\nThree competing hypotheses:\n")
    for name, model in models.items():
        params = model.get_parameters()
        if not params:
            print(f"1. {name:20s} θ = {{}}")
        else:
            params_str = ', '.join(f"{k}={v:.1f}" for k, v in params.items())
            print(f"   {name:20s} θ = {{{params_str}}}")

    print("\n→ These are hypotheses to test, not assumptions to bake in")

    # ========================================================================
    # Model Comparison: Rate Effects
    # ========================================================================
    print("\n" + "=" * 80)
    print("Model Comparison: How Constraints Affect Rates")
    print("=" * 80)

    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)

    # Test transitions
    test_transitions = [
        (state_AB, state_ABC, "AB → ABC (simple join)"),
        (state_AB, AssemblyState.from_parts(['A', 'B', 'C'], depth=2, motifs={'helix'}),
         "AB → helix (motif gain)"),
        (AssemblyState.from_parts(['A', 'A', 'B'], depth=2),
         AssemblyState.from_parts(['A', 'A', 'B', 'C'], depth=3),
         "AAB → AABC (reuse + depth)"),
    ]

    print(f"\n{'Transition':<30s} {'Model':<20s} {'C':<8s} {'λ_eff':<10s} {'Boost':<8s}")
    print("-" * 80)

    for source, target, desc in test_transitions:
        base_rate = baseline.get_assembly_rate(source, target, TransitionType.JOIN)

        for name, model in models.items():
            C = model.constraint_contribution(source, target, TransitionType.JOIN)
            eff_rate = base_rate * np.exp(C)
            boost = np.exp(C)

            print(f"{desc:<30s} {name:<20s} {C:>6.2f}  {eff_rate:>8.4f}  {boost:>6.2f}x")
        print()

    # ========================================================================
    # Scientific Questions
    # ========================================================================
    print("=" * 80)
    print("Scientific Questions (Not Philosophical Assertions)")
    print("=" * 80)

    print("\nWith this architecture, we can ask:")
    print("\n1. **Does reuse bias appear in this system?**")
    print("   - Fit null vs reuse-only models")
    print("   - Compare log-likelihoods")
    print("   - LRT: 2 × (ℓ_reuse - ℓ_null) ~ χ²(1)")

    print("\n2. **Which constraints emerge under inference?**")
    print("   - Start with θ = {}")
    print("   - Fit all features with regularization")
    print("   - See which weights are non-zero")

    print("\n3. **Do constraints strengthen over time?**")
    print("   - Fit θ_early and θ_late")
    print("   - Test: ||θ_late|| > ||θ_early||")

    print("\n4. **Are abiotic systems constraint-free?**")
    print("   - Fit θ_abiotic")
    print("   - Test: θ_abiotic ≈ 0")

    print("\n5. **Are early-life systems weakly constrained?**")
    print("   - Fit θ_early_life")
    print("   - Test: 0 < ||θ_early_life|| < ||θ_modern||")

    # ========================================================================
    # Inference Preview
    # ========================================================================
    print("\n" + "=" * 80)
    print("Inference Preview: Fitting Constraints from Data")
    print("=" * 80)

    print("\nInference pipeline:")
    print("  1. Observe assemblies in system")
    print("  2. Extract features for all transitions")
    print("  3. Fit weights: θ̂ = argmax_θ P(observations | θ)")
    print("  4. Compare models via AIC/BIC/LRT")
    print("  5. Report which features matter")

    print("\nExample output:")
    print("  Fitted weights (MLE):")
    print("    reuse_count:     1.2 ± 0.3  (p < 0.001) ✓ significant")
    print("    depth_change:   -0.4 ± 0.2  (p = 0.02)  ✓ significant")
    print("    symmetry_score:  0.1 ± 0.4  (p = 0.80)  ✗ not significant")
    print("\n  → Reuse and depth matter; symmetry doesn't")

    print("\nModel comparison:")
    print("  Null:            log-lik = -150.2, AIC = 300.4")
    print("  Reuse-only:      log-lik = -142.1, AIC = 288.2  (Δ = -12.2)")
    print("  Assembly theory: log-lik = -138.5, AIC = 283.0  (Δ = -17.4)")
    print("\n  → Assembly theory wins (lowest AIC)")

    # ========================================================================
    # Future: Suggested Constraints
    # ========================================================================
    print("\n" + "=" * 80)
    print("Future: Suggested Constraints (Once We Know More)")
    print("=" * 80)

    print("\nOnce we have validation, we can offer presets:")
    print("\n  # Null (always available)")
    print("  constraint = AssemblyConstraint.null_model()")
    print("\n  # Suggested (based on published studies)")
    print("  constraint = AssemblyConstraint.suggested(")
    print("      regime='early_life',  # or 'abiotic', 'modern'")
    print("      confidence='low'      # or 'medium', 'high'")
    print("  )")
    print("\n  # Custom (user-defined)")
    print("  constraint = AssemblyConstraint({")
    print("      'reuse_count': 1.5,")
    print("      'depth_change': -0.2,")
    print("  })")

    print("\nBut for now: Let data decide, not assumptions.")

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)

    print("\nKey Takeaways:")
    print("  ✓ Layer 1 (mechanics) is hypothesis-neutral")
    print("  ✓ Layer 2 (theories) is user-defined")
    print("  ✓ Features are observables, weights are hypotheses")
    print("  ✓ Null model is always available")
    print("  ✓ Model comparison via likelihood")
    print("  ✓ Science, not philosophy")
    print("\nNext: Implement CTMC dynamics for full inference!")


if __name__ == '__main__':
    main()
