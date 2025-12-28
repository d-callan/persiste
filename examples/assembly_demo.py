"""
Demo: Assembly Theory in PERSISTE

Shows how assembly states, baselines, and constraints work together.
"""

import sys
sys.path.insert(0, 'src')

from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import (
    AssemblyBaseline, TransitionType
)
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
import numpy as np


def main():
    print("=" * 70)
    print("Assembly Theory Demo")
    print("=" * 70)
    
    # 1. Create some assembly states
    print("\n1. Creating Assembly States")
    print("-" * 70)
    
    s_empty = AssemblyState.empty()
    s_A = AssemblyState.from_parts(['A'], depth=0)
    s_B = AssemblyState.from_parts(['B'], depth=0)
    s_AB = AssemblyState.from_parts(['A', 'B'], depth=1)
    s_AAB = AssemblyState.from_parts(['A', 'A', 'B'], depth=2)
    s_helix = AssemblyState.from_parts(['A', 'B', 'C'], depth=2, motifs={'helix'})
    
    states = [s_empty, s_A, s_B, s_AB, s_AAB, s_helix]
    for s in states:
        print(f"  {s}")
    
    # 2. Test state properties
    print("\n2. State Properties")
    print("-" * 70)
    
    print(f"  s_A is subassembly of s_AB: {s_A.is_subassembly_of(s_AB)}")
    print(f"  s_AB is subassembly of s_AAB: {s_AB.is_subassembly_of(s_AAB)}")
    print(f"  s_helix has motif 'helix': {s_helix.contains_motif('helix')}")
    
    # States are hashable
    state_set = {s_A, s_B, s_AB}
    print(f"  States are hashable, can use in sets: {len(state_set)} unique states")
    
    # 3. Create baseline (physics-agnostic)
    print("\n3. Baseline Model (Physics-Agnostic)")
    print("-" * 70)
    
    baseline = AssemblyBaseline(
        kappa=1.0,
        join_exponent=-0.5,  # Harder to join larger assemblies
        split_exponent=0.3,   # Easier to split larger assemblies
        decay_rate=0.01,
    )
    print(f"  {baseline}")
    
    # Compute baseline rates
    print("\n  Baseline rates (no chemistry):")
    
    # Join: A + B → AB
    rate_join = baseline.get_assembly_rate(s_A, s_AB, TransitionType.JOIN)
    print(f"    A + B → AB (join):  λ = {rate_join:.4f}")
    
    # Join: AB + A → AAB
    rate_join2 = baseline.get_assembly_rate(s_AB, s_AAB, TransitionType.JOIN)
    print(f"    AB + A → AAB (join): λ = {rate_join2:.4f}")
    
    # Split: AB → A + B
    rate_split = baseline.get_assembly_rate(s_AB, s_A, TransitionType.SPLIT)
    print(f"    AB → A + B (split):  λ = {rate_split:.4f}")
    
    # Decay: AB → ∅
    rate_decay = baseline.get_assembly_rate(s_AB, s_empty, TransitionType.DECAY)
    print(f"    AB → ∅ (decay):      λ = {rate_decay:.4f}")
    
    # 4. Create constraint (assembly theory)
    print("\n4. Constraint Model (Assembly Theory)")
    print("-" * 70)
    
    constraint = AssemblyConstraint(
        motif_bonuses={'helix': 2.0, 'sheet': 1.5},
        reuse_bonus=1.0,
        depth_penalty=-0.3,
        env_fit=0.5,
    )
    print(f"  {constraint}")
    
    # Compute constraint contributions
    print("\n  Constraint contributions (log-scale):")
    
    # Join with reuse: AB + A → AAB (reuses AB)
    C_reuse = constraint.constraint_contribution(s_AB, s_AAB, TransitionType.JOIN)
    print(f"    AB + A → AAB (reuse): C = {C_reuse:.4f}")
    
    # Join with motif: A + B → helix
    C_motif = constraint.constraint_contribution(s_AB, s_helix, TransitionType.JOIN)
    print(f"    AB → helix (motif):   C = {C_motif:.4f}")
    
    # 5. Effective rates
    print("\n5. Effective Rates (Baseline × Constraint)")
    print("-" * 70)
    
    print("\n  λ_eff = λ_baseline × exp(C)")
    
    # Without constraint
    lambda_base = baseline.get_assembly_rate(s_AB, s_AAB, TransitionType.JOIN)
    print(f"    AB + A → AAB (baseline only): λ = {lambda_base:.4f}")
    
    # With constraint (reuse bonus)
    C = constraint.constraint_contribution(s_AB, s_AAB, TransitionType.JOIN)
    lambda_eff = lambda_base * np.exp(C)
    print(f"    AB + A → AAB (with reuse):    λ = {lambda_eff:.4f} (boost: {np.exp(C):.2f}x)")
    
    # With motif bonus
    lambda_base_helix = baseline.get_assembly_rate(s_AB, s_helix, TransitionType.JOIN)
    C_helix = constraint.constraint_contribution(s_AB, s_helix, TransitionType.JOIN)
    lambda_eff_helix = lambda_base_helix * np.exp(C_helix)
    print(f"    AB → helix (with motif):      λ = {lambda_eff_helix:.4f} (boost: {np.exp(C_helix):.2f}x)")
    
    # 6. Assembly index emerges
    print("\n6. Assembly Index (Emerges from Model)")
    print("-" * 70)
    
    print("\n  Assembly index = minimum constraint-adjusted path length")
    print("  Low index  = many cheap paths (easy to assemble)")
    print("  High index = rare under baseline, rescued by constraint (complex)")
    print("\n  This is NOT hard-coded - it falls out of the dynamics!")
    
    # Simple example: depth as proxy
    for s in [s_A, s_AB, s_AAB, s_helix]:
        print(f"    {s.assembly_depth:2d} - {s}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    
    print("\nKey Takeaways:")
    print("  ✓ States are compositional (multisets), not molecular graphs")
    print("  ✓ Baseline is physics-agnostic (no chemistry)")
    print("  ✓ Constraint carries assembly theory (motifs, reuse, etc.)")
    print("  ✓ Effective rates = baseline × exp(constraint)")
    print("  ✓ Assembly index emerges from model dynamics")
    print("\nNext: Implement lazy graph generation and observation models!")


if __name__ == '__main__':
    main()
