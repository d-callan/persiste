#!/usr/bin/env python3
"""
Test all three strain-focused recipes on E. coli data.

Demonstrates:
1. Strain Heterogeneity Scan - diagnostic
2. Stratified Regime Modeling - explicit subpopulation modeling
3. Regime Contrast Test - hypothesis testing
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persiste.plugins.genecontent.strain_diagnostics import diagnose_strain_heterogeneity
from persiste.plugins.genecontent.strain_recipes import (
    strain_heterogeneity_scan,
    stratified_regime_modeling,
    regime_contrast_test,
)


def main():
    print("=" * 70)
    print("STRAIN-FOCUSED RECIPE DEMONSTRATION")
    print("=" * 70)
    print("\nTesting three recipes on E. coli pangenome data")
    
    # Load data
    data_dir = Path("data/ecoli_real")
    pam_file = data_dir / "Supplementary File 2A.txt"
    
    if not pam_file.exists():
        print(f"\n✗ Data file not found: {pam_file}")
        return
    
    print("\nLoading E. coli pangenome dataset...")
    df = pd.read_csv(pam_file, sep='\t', index_col=0)
    pam = df.values.T.astype(int)
    strain_names = df.columns.tolist()
    gene_names = df.index.tolist()
    
    print(f"  Dataset: {len(strain_names):,} strains × {len(gene_names):,} genes")
    
    # Step 0: Diagnostic
    print("\n" + "=" * 70)
    print("STEP 0: DIAGNOSTIC")
    print("=" * 70)
    
    diag = diagnose_strain_heterogeneity(pam)
    diag.print_report()
    
    # Recipe 1: Strain Heterogeneity Scan
    print("\n" + "=" * 70)
    print("RECIPE 1: STRAIN HETEROGENEITY SCAN")
    print("=" * 70)
    print("\nGoal: Detect whether multiple regimes are plausible")
    
    scan_result = strain_heterogeneity_scan(
        pam=pam,
        taxon_names=strain_names,
        gene_names=gene_names,
        tree_method='jaccard_upgma',
        verbose=True
    )
    
    scan_result.print_summary()
    
    # Recipe 2: Stratified Regime Modeling
    print("\n" + "=" * 70)
    print("RECIPE 2: STRATIFIED REGIME MODELING")
    print("=" * 70)
    print("\nGoal: Model regimes separately after detecting heterogeneity")
    
    stratified_result = stratified_regime_modeling(
        pam=pam,
        taxon_names=strain_names,
        gene_names=gene_names,
        threshold=None,  # Use median
        tree_method='jaccard_upgma',
        verbose=True
    )
    
    stratified_result.print_summary()
    
    # Recipe 3: Regime Contrast Test
    print("\n" + "=" * 70)
    print("RECIPE 3: REGIME CONTRAST TEST")
    print("=" * 70)
    print("\nGoal: Test whether regimes differ significantly")
    
    contrast_result = regime_contrast_test(
        pam=pam,
        taxon_names=strain_names,
        gene_names=gene_names,
        threshold=None,  # Use median
        tree_method='jaccard_upgma',
        verbose=True
    )
    
    contrast_result.print_summary()
    
    # Final summary
    print("\n" + "=" * 70)
    print("OVERALL CONCLUSIONS")
    print("=" * 70)
    
    print("\n1. DIAGNOSTIC:")
    print(f"   Heterogeneity level: {diag.heterogeneity_level.upper()}")
    print(f"   Top 10% contribution: {diag.top_10pct_contribution:.1f}%")
    
    print("\n2. HETEROGENEITY SCAN:")
    print(f"   {scan_result.interpretation}")
    max_shift = max(abs(v) for v in scan_result.parameter_shifts.values())
    print(f"   Maximum parameter shift: {max_shift:.1f}%")
    
    print("\n3. STRATIFIED MODELING:")
    high_regime = "gain" if stratified_result.high_accessory_result.gain_rate > stratified_result.high_accessory_result.loss_rate else "loss"
    low_regime = "gain" if stratified_result.low_accessory_result.gain_rate > stratified_result.low_accessory_result.loss_rate else "loss"
    print(f"   High-accessory: {high_regime.upper()}-dominated (π₁={stratified_result.high_accessory_result.equilibrium_frequency:.4f})")
    print(f"   Low-accessory:  {low_regime.upper()}-dominated (π₁={stratified_result.low_accessory_result.equilibrium_frequency:.4f})")
    
    print("\n4. REGIME CONTRAST TEST:")
    print(f"   Preferred model: {contrast_result.preferred_model.upper()}")
    print(f"   Evidence: {contrast_result.evidence_strength.upper()}")
    print(f"   ΔAIC: {contrast_result.delta_aic:+.2f}, ΔBIC: {contrast_result.delta_bic:+.2f}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    if contrast_result.preferred_model == "stratified":
        print("\n✓ Use stratified modeling for this dataset")
        print("  • Multiple evolutionary regimes detected")
        print("  • High-accessory and low-accessory strains evolve differently")
        print("  • Report results separately for each subpopulation")
    else:
        print("\n• Global model is adequate")
        print("  • Single regime describes data well")
        print("  • Standard analysis recommended")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
