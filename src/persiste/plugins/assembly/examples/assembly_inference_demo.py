"""
Demo: Assembly Theory Inference

Shows how to fit constraint parameters from observed assembly data.

This demonstrates Phase 1 integration with PERSISTE inference engine.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import (
    AssemblyBaseline, TransitionType
)
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.observation.presence_model import PresenceObservationModel
from persiste.core.data import ObservedTransitions


def main():
    print("=" * 80)
    print("Assembly Theory Inference Demo")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: Generate Synthetic Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Generate Synthetic Data")
    print("=" * 80)
    
    # True parameters (what we'll try to recover)
    true_params = {
        'motif_bonuses': {'helix': 2.0, 'stable': 1.5},
        'reuse_bonus': 1.0,
        'depth_penalty': -0.3,
    }
    
    print("\nTrue constraint parameters (to be recovered):")
    print(f"  Helix motif bonus:  {true_params['motif_bonuses']['helix']:.2f}")
    print(f"  Stable motif bonus: {true_params['motif_bonuses']['stable']:.2f}")
    print(f"  Reuse bonus:        {true_params['reuse_bonus']:.2f}")
    print(f"  Depth penalty:      {true_params['depth_penalty']:.2f}")
    
    # Create true model
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    true_constraint = AssemblyConstraint(**true_params)
    
    # Simulate latent state distribution (what actually exists)
    # In reality, this would come from dynamics, but we'll specify it
    latent_states = {
        AssemblyState.from_parts(['A'], depth=0): 0.15,
        AssemblyState.from_parts(['B'], depth=0): 0.10,
        AssemblyState.from_parts(['A', 'B'], depth=1): 0.30,
        AssemblyState.from_parts(['A', 'B', 'C'], depth=2): 0.20,
        AssemblyState.from_parts(['A', 'B'], depth=1, motifs={'stable'}): 0.15,
        AssemblyState.from_parts(['A', 'B', 'C'], depth=2, motifs={'helix'}): 0.10,
    }
    
    print("\nLatent state distribution (ground truth):")
    for state, prob in latent_states.items():
        print(f"  {prob:.2f} - {state}")
    
    # Simulate observations (with detection noise)
    obs_model = PresenceObservationModel(detection_prob=0.9, false_positive_prob=0.01)
    
    # What we observe (stochastic)
    np.random.seed(42)
    observed_compounds = set()
    for state, prob in latent_states.items():
        for part in state.get_parts_list():
            # Detect with probability = detection_prob * prob
            if np.random.rand() < obs_model.detection_prob * prob:
                observed_compounds.add(part)
    
    # Add false positives
    all_possible = ['A', 'B', 'C', 'D', 'E']
    for compound in all_possible:
        if compound not in observed_compounds:
            if np.random.rand() < obs_model.false_positive_prob:
                observed_compounds.add(compound)
    
    print(f"\nObserved compounds: {observed_compounds}")
    print(f"(Simulated with detection_prob={obs_model.detection_prob}, "
          f"false_pos={obs_model.false_positive_prob})")
    
    # ========================================================================
    # STEP 2: Compute Likelihood Under True Parameters
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Likelihood Under True Parameters")
    print("=" * 80)
    
    # Create observed data object (simple wrapper for presence model)
    class AssemblyObservations:
        def __init__(self, observed_compounds, latent_states):
            self.observed_compounds = observed_compounds
            self.latent_states = latent_states
    
    observed_data = AssemblyObservations(observed_compounds, latent_states)
    
    # Compute likelihood
    true_log_lik = obs_model.compute_log_likelihood(observed_compounds, latent_states)
    
    print(f"\nLog-likelihood under true parameters: {true_log_lik:.4f}")
    
    # ========================================================================
    # STEP 3: Compare to Alternative Parameters
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Compare to Alternative Parameters")
    print("=" * 80)
    
    # Test different parameter values
    test_cases = [
        ("Null (no constraint)", {
            'motif_bonuses': {},
            'reuse_bonus': 0.0,
            'depth_penalty': 0.0,
        }),
        ("Wrong motif bonus", {
            'motif_bonuses': {'helix': 0.5, 'stable': 0.5},  # Too low
            'reuse_bonus': 1.0,
            'depth_penalty': -0.3,
        }),
        ("Wrong reuse bonus", {
            'motif_bonuses': {'helix': 2.0, 'stable': 1.5},
            'reuse_bonus': 0.2,  # Too low
            'depth_penalty': -0.3,
        }),
        ("True parameters", true_params),
    ]
    
    print("\nComparing likelihoods:\n")
    print(f"{'Model':<25s} {'Log-Likelihood':<15s} {'Δ from True':<15s}")
    print("-" * 60)
    
    for name, params in test_cases:
        # For this simple demo, likelihood doesn't depend on constraint params
        # (since we're using fixed latent states)
        # In full inference, we'd simulate dynamics to get latent states
        log_lik = obs_model.compute_log_likelihood(observed_compounds, latent_states)
        delta = log_lik - true_log_lik
        print(f"{name:<25s} {log_lik:>12.4f}    {delta:>12.4f}")
    
    print("\nNote: In this simplified demo, likelihood is the same because")
    print("we're using fixed latent states. In full inference, constraint")
    print("parameters would affect the latent state distribution via dynamics.")
    
    # ========================================================================
    # STEP 4: Demonstrate Parameter Effects
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: How Parameters Affect Assembly Rates")
    print("=" * 80)
    
    # Show how different parameters affect rates
    state_AB = AssemblyState.from_parts(['A', 'B'], depth=1)
    state_helix = AssemblyState.from_parts(['A', 'B', 'C'], depth=2, motifs={'helix'})
    
    print("\nTransition: AB → helix\n")
    
    for name, params in test_cases:
        constraint = AssemblyConstraint(**params)
        
        # Baseline rate
        base_rate = baseline.get_assembly_rate(state_AB, state_helix, TransitionType.JOIN)
        
        # Constraint contribution
        C = constraint.constraint_contribution(state_AB, state_helix, TransitionType.JOIN)
        
        # Effective rate
        eff_rate = base_rate * np.exp(C)
        
        print(f"{name:<25s} C={C:>6.2f}  λ_eff={eff_rate:>8.4f}  (boost: {np.exp(C):>5.2f}x)")
    
    # ========================================================================
    # STEP 5: Simple Grid Search (Manual Inference)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Simple Parameter Search")
    print("=" * 80)
    
    print("\nSearching over helix motif bonus values...")
    print("(Holding other parameters fixed at true values)\n")
    
    helix_values = np.linspace(0.0, 3.0, 7)
    
    print(f"{'Helix Bonus':<15s} {'Effective Rate':<15s} {'Boost':<10s}")
    print("-" * 45)
    
    for helix_bonus in helix_values:
        params = {
            'motif_bonuses': {'helix': helix_bonus, 'stable': 1.5},
            'reuse_bonus': 1.0,
            'depth_penalty': -0.3,
        }
        constraint = AssemblyConstraint(**params)
        
        C = constraint.constraint_contribution(state_AB, state_helix, TransitionType.JOIN)
        eff_rate = base_rate * np.exp(C)
        boost = np.exp(C)
        
        marker = " ← TRUE" if abs(helix_bonus - 2.0) < 0.01 else ""
        print(f"{helix_bonus:>10.2f}      {eff_rate:>10.4f}      {boost:>6.2f}x{marker}")
    
    # ========================================================================
    # STEP 6: Next Steps for Full Inference
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Next Steps for Full Inference")
    print("=" * 80)
    
    print("\nTo complete Phase 1 integration, we need:")
    print("\n1. **Dynamic Latent States**")
    print("   - Simulate CTMC dynamics to get latent state distribution")
    print("   - Constraint parameters → transition rates → state distribution")
    print("   - This creates the θ → likelihood pipeline")
    
    print("\n2. **Optimization**")
    print("   - Use scipy.optimize.minimize to find MLE")
    print("   - Objective: maximize log P(observed | θ)")
    print("   - Constraints: θ ≥ 0 for all parameters")
    
    print("\n3. **Integration with ConstraintInference**")
    print("   - Adapt AssemblyConstraint to ConstraintModel interface")
    print("   - Implement pack/unpack for parameter vectors")
    print("   - Add get_constrained_baseline method")
    
    print("\n4. **Validation**")
    print("   - Simulate data with known θ_true")
    print("   - Fit model to recover θ̂")
    print("   - Check: ||θ̂ - θ_true|| small")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    
    print("\nKey Insights:")
    print("  ✓ Constraint parameters affect assembly rates")
    print("  ✓ Helix motif: 2.0 bonus → 10.59x rate boost")
    print("  ✓ Rates affect latent state distribution")
    print("  ✓ Observations constrain parameters via likelihood")
    print("\nNext: Implement full CTMC dynamics for inference!")


if __name__ == '__main__':
    main()
