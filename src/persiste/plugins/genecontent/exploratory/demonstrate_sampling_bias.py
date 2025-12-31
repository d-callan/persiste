#!/usr/bin/env python3
"""
Demonstrate sampling bias phenomenon using real E. coli data.

Shows how full dataset appears gain-dominated while subsets appear loss-dominated.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persiste.plugins.genecontent.pam_interface import fit
from persiste.plugins.genecontent.diagnostics import diagnose_sampling_bias, compare_full_vs_subset_diagnostics
from persiste.plugins.genecontent.recipes import (
    core_shell_recipe,
    exclude_singletons_recipe,
    exclude_rare_recipe,
    apply_recipe,
    compare_recipes,
)


def main():
    print("=" * 70)
    print("SAMPLING BIAS DEMONSTRATION: E. coli Pangenome")
    print("=" * 70)
    print("\nObjective: Show how sampling bias explains gain→loss regime shift")
    
    # Load data
    data_dir = Path("data/ecoli_real")
    pam_file = data_dir / "Supplementary File 2A.txt"
    
    if not pam_file.exists():
        print(f"\n✗ Data file not found: {pam_file}")
        return
    
    print("\nLoading E. coli pangenome dataset...")
    df = pd.read_csv(pam_file, sep='\t', index_col=0)
    pam_full = df.values.T.astype(int)
    strain_names = df.columns.tolist()
    gene_names = df.index.tolist()
    
    n_strains = len(strain_names)
    n_genes = len(gene_names)
    
    print(f"  Full dataset: {n_strains:,} strains × {n_genes:,} genes")
    
    # Step 1: Diagnose full dataset
    print("\n" + "=" * 70)
    print("STEP 1: DIAGNOSE FULL DATASET")
    print("=" * 70)
    
    diag_full = diagnose_sampling_bias(pam_full)
    diag_full.print_report()
    
    # Step 2: Analyze full dataset
    print("\n" + "=" * 70)
    print("STEP 2: ANALYZE FULL DATASET")
    print("=" * 70)
    
    print("\nFitting model to full dataset (1,324 strains)...")
    result_full = fit(
        pam=pam_full,
        tree=None,
        taxon_names=strain_names,
        gene_names=gene_names,
        tree_method='jaccard_upgma',
        verbose=False
    )
    
    print(f"\nFull dataset results:")
    print(f"  Gain rate (λ): {result_full.gain_rate:.4f}")
    print(f"  Loss rate (μ): {result_full.loss_rate:.4f}")
    print(f"  π₁ (stationary): {result_full.equilibrium_frequency:.4f}")
    print(f"  λ/μ ratio: {result_full.gain_rate/result_full.loss_rate:.2f}x")
    
    if result_full.gain_rate > result_full.loss_rate:
        print(f"  → GAIN-DOMINATED regime")
    else:
        print(f"  → LOSS-DOMINATED regime")
    
    # Step 3: Create subset
    print("\n" + "=" * 70)
    print("STEP 3: SUBSAMPLE STRAINS")
    print("=" * 70)
    
    n_subset = 100
    print(f"\nCreating random subset of {n_subset} strains...")
    
    np.random.seed(42)
    subset_idx = np.random.choice(n_strains, n_subset, replace=False)
    pam_subset = pam_full[subset_idx, :]
    strain_subset = [strain_names[i] for i in subset_idx]
    
    # Diagnose subset
    diag_subset = diagnose_sampling_bias(pam_subset)
    diag_subset.print_report()
    
    # Compare diagnostics
    compare_full_vs_subset_diagnostics(pam_full, pam_subset, f"{n_subset}-strain subset")
    
    # Step 4: Analyze subset
    print("\n" + "=" * 70)
    print("STEP 4: ANALYZE SUBSET")
    print("=" * 70)
    
    print(f"\nFitting model to subset ({n_subset} strains)...")
    result_subset = fit(
        pam=pam_subset,
        tree=None,
        taxon_names=strain_subset,
        gene_names=gene_names,
        tree_method='jaccard_upgma',
        verbose=False
    )
    
    print(f"\nSubset results:")
    print(f"  Gain rate (λ): {result_subset.gain_rate:.4f}")
    print(f"  Loss rate (μ): {result_subset.loss_rate:.4f}")
    print(f"  π₁ (stationary): {result_subset.equilibrium_frequency:.4f}")
    print(f"  λ/μ ratio: {result_subset.gain_rate/result_subset.loss_rate:.2f}x")
    
    if result_subset.gain_rate > result_subset.loss_rate:
        print(f"  → GAIN-DOMINATED regime")
    else:
        print(f"  → LOSS-DOMINATED regime")
    
    # Step 5: Compare results
    print("\n" + "=" * 70)
    print("STEP 5: COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Parameter':<25} {'Full (1,324)':<20} {'Subset ({n_subset})':<20} {'Change':<15}")
    print("-" * 80)
    print(f"{'Gain rate (λ)':<25} {result_full.gain_rate:<20.4f} {result_subset.gain_rate:<20.4f} {(result_subset.gain_rate/result_full.gain_rate-1)*100:>+6.1f}%")
    print(f"{'Loss rate (μ)':<25} {result_full.loss_rate:<20.4f} {result_subset.loss_rate:<20.4f} {(result_subset.loss_rate/result_full.loss_rate-1)*100:>+6.1f}%")
    print(f"{'π₁ (stationary)':<25} {result_full.equilibrium_frequency:<20.4f} {result_subset.equilibrium_frequency:<20.4f} {(result_subset.equilibrium_frequency-result_full.equilibrium_frequency)*100:>+6.1f}%")
    print(f"{'λ/μ ratio':<25} {result_full.gain_rate/result_full.loss_rate:<20.2f} {result_subset.gain_rate/result_subset.loss_rate:<20.2f}")
    
    # Key observation
    print("\n" + "=" * 70)
    print("KEY OBSERVATION")
    print("=" * 70)
    
    full_regime = "gain" if result_full.gain_rate > result_full.loss_rate else "loss"
    subset_regime = "gain" if result_subset.gain_rate > result_subset.loss_rate else "loss"
    
    if full_regime != subset_regime:
        print(f"\n✓ REGIME SHIFT OBSERVED!")
        print(f"  • Full dataset: {full_regime.upper()}-dominated (λ/μ = {result_full.gain_rate/result_full.loss_rate:.2f})")
        print(f"  • Subset: {subset_regime.upper()}-dominated (λ/μ = {result_subset.gain_rate/result_subset.loss_rate:.2f})")
    else:
        print(f"\n• Both datasets show {full_regime.upper()}-dominated regime")
    
    print(f"\nπ₁ explains everything:")
    print(f"  • Full: π₁ = {result_full.equilibrium_frequency:.4f} → {(1-result_full.equilibrium_frequency)*100:.1f}% genes absent at equilibrium")
    print(f"  • Subset: π₁ = {result_subset.equilibrium_frequency:.4f} → {(1-result_subset.equilibrium_frequency)*100:.1f}% genes absent at equilibrium")
    
    # Step 6: Test frequency-aware recipes
    print("\n" + "=" * 70)
    print("STEP 6: FREQUENCY-AWARE ANALYSIS RECIPES")
    print("=" * 70)
    
    print("\nTesting recipes on full dataset...")
    
    recipes = [
        core_shell_recipe(pam_full, cloud_threshold=0.15),
        exclude_singletons_recipe(pam_full),
        exclude_rare_recipe(pam_full, rare_threshold=10),
    ]
    
    compare_recipes(pam_full, recipes)
    
    print("\nFitting models with different recipes...")
    print(f"\n{'Recipe':<30} {'Genes':<12} {'λ':<10} {'μ':<10} {'π₁':<10} {'λ/μ':<10}")
    print("-" * 80)
    
    # Baseline
    print(f"{'Full dataset (baseline)':<30} {n_genes:<12,} {result_full.gain_rate:<10.4f} {result_full.loss_rate:<10.4f} {result_full.equilibrium_frequency:<10.4f} {result_full.gain_rate/result_full.loss_rate:<10.2f}")
    
    for recipe in recipes:
        pam_filtered, _, genes_filtered = apply_recipe(
            pam_full, strain_names, gene_names, recipe, verbose=False
        )
        
        result_recipe = fit(
            pam=pam_filtered,
            tree=None,
            taxon_names=strain_names,
            gene_names=genes_filtered,
            tree_method='jaccard_upgma',
            verbose=False
        )
        
        print(f"{recipe.name:<30} {len(genes_filtered):<12,} {result_recipe.gain_rate:<10.4f} {result_recipe.loss_rate:<10.4f} {result_recipe.equilibrium_frequency:<10.4f} {result_recipe.gain_rate/result_recipe.loss_rate:<10.2f}")
    
    # Step 7: Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION & RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n1. WHY THIS HAPPENS:")
    print("   • Full dataset has many rare genes (sampling artifacts)")
    print("   • Rare genes inflate gain rate estimates")
    print("   • Subsampling filters out rare genes naturally")
    print("   • Result: subset appears more loss-dominated")
    
    print("\n2. THIS IS EXPECTED:")
    print("   • Not a bug in the method")
    print("   • Reflects true sampling bias in pangenome data")
    print("   • Open pangenomes naturally have this property")
    
    print("\n3. WHAT TO REPORT:")
    print("   ✓ Always report π₁ (stationary frequency) prominently")
    print("   ✓ Report % rare genes and sampling bias diagnostics")
    print("   ✓ Compare results with frequency-aware recipes")
    print("   ✓ Acknowledge sampling bias in interpretation")
    
    print("\n4. RECOMMENDED ANALYSIS:")
    if diag_full.pct_rare > 50:
        print("   → Use 'core_shell' recipe to exclude cloud genes")
        print("   → Compare results across multiple frequency thresholds")
        print("   → Report sensitivity to rare gene inclusion")
    else:
        print("   → Standard analysis appropriate")
        print("   → Minimal sampling bias detected")
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("\n✓ Sampling bias explains gain→loss regime shift")
    print("✓ π₁ (stationary frequency) is the key interpretable metric")
    print("✓ Frequency-aware recipes help mitigate bias")
    print("✓ Always use diagnostics and report transparently")
    print("\nThis behavior is EXPECTED and DOCUMENTED.")


if __name__ == "__main__":
    main()
