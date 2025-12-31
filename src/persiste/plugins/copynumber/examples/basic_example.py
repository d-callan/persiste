#!/usr/bin/env python3
"""
Basic example of CopyNumberDynamics plugin usage.

Demonstrates:
1. Creating synthetic copy number data
2. Fitting null model (baseline only)
3. Fitting alternative model (with constraint)
4. Likelihood ratio test
5. Model comparison
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from persiste.core.trees import Tree
from persiste.plugins.copynumber import fit
from persiste.plugins.copynumber.cn_interface import (
    fit_null_model,
    likelihood_ratio_test,
)
from persiste.plugins.copynumber.states.cn_states import CopyNumberState


def create_synthetic_data(
    n_families: int = 50,
    n_taxa: int = 20,
    seed: int = 42
) -> tuple:
    """
    Create synthetic copy number data for testing.
    
    Returns:
        (cn_matrix, family_names, taxon_names, tree)
    """
    rng = np.random.default_rng(seed)
    
    # Generate random copy number states
    # Biased toward single copy (realistic)
    state_probs = [0.1, 0.6, 0.2, 0.1]  # ABSENT, SINGLE, LOW_MULTI, HIGH_MULTI
    
    cn_matrix = rng.choice(4, size=(n_families, n_taxa), p=state_probs)
    
    # Create names
    family_names = [f"family_{i:03d}" for i in range(n_families)]
    taxon_names = [f"taxon_{i:02d}" for i in range(n_taxa)]
    
    # Create simple tree (star tree for simplicity)
    # In real usage, you'd load a real phylogeny
    tree = create_star_tree(n_taxa)
    
    return cn_matrix, family_names, taxon_names, tree


def create_star_tree(n_taxa: int) -> Tree:
    """
    Create a simple star tree for testing.
    
    In real usage, you'd load a real phylogenetic tree.
    """
    # This is a placeholder - in real usage you'd use proper tree construction
    # For now, just create a simple Tree object
    from persiste.core.trees import Tree
    
    # Create a star tree structure
    # Root with n_taxa children, all with branch length 1.0
    tree = Tree()
    
    # Add root
    root_id = tree.add_node(is_leaf=False)
    
    # Add taxa as children of root
    for i in range(n_taxa):
        tree.add_node(parent=root_id, branch_length=1.0, is_leaf=True, name=f"taxon_{i:02d}")
    
    return tree


def main():
    """Run basic example."""
    
    print("=" * 70)
    print("COPY NUMBER DYNAMICS - BASIC EXAMPLE")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: Create synthetic data
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: CREATE SYNTHETIC DATA")
    print("=" * 70)
    
    cn_matrix, family_names, taxon_names, tree = create_synthetic_data(
        n_families=50,
        n_taxa=20,
        seed=42
    )
    
    print(f"\nGenerated synthetic data:")
    print(f"  {len(family_names)} families × {len(taxon_names)} taxa")
    print(f"\nState distribution:")
    for state in range(4):
        count = np.sum(cn_matrix == state)
        freq = count / cn_matrix.size
        state_name = CopyNumberState(state).name
        print(f"  {state_name:12s} ({state}): {freq:.3f}")
    
    # ========================================================================
    # STEP 2: Fit null model (baseline only)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: FIT NULL MODEL")
    print("=" * 70)
    print("\nFitting baseline-only model (no constraint)...")
    
    null_result = fit_null_model(
        cn_matrix=cn_matrix,
        family_names=family_names,
        taxon_names=taxon_names,
        tree=tree,
        baseline_type='global',  # Use global for speed in example
        verbose=True
    )
    
    # ========================================================================
    # STEP 3: Fit alternative model (with dosage stability constraint)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: FIT ALTERNATIVE MODEL")
    print("=" * 70)
    print("\nFitting model with dosage stability constraint...")
    print("Hypothesis: Genes are dosage-buffered (θ < 0)")
    
    alt_result = fit(
        cn_matrix=cn_matrix,
        family_names=family_names,
        taxon_names=taxon_names,
        tree=tree,
        baseline_type='global',
        constraint_type='dosage_stability',
        theta=-0.5,  # Buffered (stable dosage)
        verbose=True
    )
    
    # ========================================================================
    # STEP 4: Likelihood ratio test
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: LIKELIHOOD RATIO TEST")
    print("=" * 70)
    
    lrt_results = likelihood_ratio_test(
        alternative=alt_result,
        null=null_result,
        verbose=True
    )
    
    # ========================================================================
    # STEP 5: Try amplification bias constraint
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: ALTERNATIVE CONSTRAINT")
    print("=" * 70)
    print("\nFitting model with amplification bias constraint...")
    print("Hypothesis: Genes favor amplification (θ > 0)")
    
    amp_result = fit(
        cn_matrix=cn_matrix,
        family_names=family_names,
        taxon_names=taxon_names,
        tree=tree,
        baseline_type='global',
        constraint_type='amplification_bias',
        theta=0.3,  # Amplification favored
        verbose=True
    )
    
    # Compare to null
    print("\n" + "-" * 70)
    print("Comparing amplification bias model to null:")
    print("-" * 70)
    
    amp_lrt = likelihood_ratio_test(
        alternative=amp_result,
        null=null_result,
        verbose=True
    )
    
    # ========================================================================
    # STEP 6: Model comparison summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\nModel comparison (AIC):")
    models = [
        ("Null (baseline only)", null_result),
        ("Dosage stability", alt_result),
        ("Amplification bias", amp_result),
    ]
    
    for name, result in models:
        print(f"\n{name}:")
        print(f"  Log-likelihood: {result.log_likelihood:.2f}")
        print(f"  AIC: {result.aic:.2f}")
        print(f"  BIC: {result.bic:.2f}")
        if result.theta is not None:
            print(f"  θ: {result.theta:.4f}")
    
    # Find best model by AIC
    best_idx = np.argmin([r.aic for _, r in models])
    print(f"\nBest model by AIC: {models[best_idx][0]}")
    
    # ========================================================================
    # STEP 7: Binning example
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: BINNING RAW COPY NUMBERS")
    print("=" * 70)
    
    print("\nExample: Binning raw copy numbers to states")
    print("\nFor diploid organism (ploidy=2):")
    
    raw_counts = [0, 2, 4, 6, 8, 10]
    for count in raw_counts:
        state = CopyNumberState.from_raw_count(count, ploidy=2)
        print(f"  {count:2d} copies → {state.name:12s} ({state.value})")
    
    print("\nFor haploid organism (ploidy=1):")
    for count in raw_counts:
        state = CopyNumberState.from_raw_count(count, ploidy=1)
        print(f"  {count:2d} copies → {state.name:12s} ({state.value})")
    
    # ========================================================================
    # DONE
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)
    
    print("\nKey takeaways:")
    print("  1. Always fit null model first")
    print("  2. Test biologically motivated constraints")
    print("  3. Use LRT and AIC/BIC for model selection")
    print("  4. Interpret θ in biological context")
    print("  5. Hierarchical baseline recommended for real data")
    
    print("\nNext steps:")
    print("  - Try with real copy number data")
    print("  - Use hierarchical baseline for better inference")
    print("  - Test multiple constraint types")
    print("  - Integrate with GeneContent results")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
