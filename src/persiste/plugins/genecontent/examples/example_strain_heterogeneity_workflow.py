#!/usr/bin/env python3
"""
Example workflow for pangenome analysis with strain heterogeneity checking.

This demonstrates the recommended two-step approach:
1. Run heterogeneity scan (ALWAYS)
2. Use stratified modeling if needed
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persiste.plugins.genecontent.pam_interface import fit
from persiste.plugins.genecontent.strain_diagnostics import diagnose_strain_heterogeneity
from persiste.plugins.genecontent.strain_recipes import (
    strain_heterogeneity_scan,
    stratified_regime_modeling,
)


def main():
    """Example workflow for E. coli pangenome analysis."""
    
    print("=" * 70)
    print("PANGENOME ANALYSIS WITH STRAIN HETEROGENEITY CHECKING")
    print("=" * 70)
    
    # Load data
    data_dir = Path("data/ecoli_real")
    pam_file = data_dir / "Supplementary File 2A.txt"
    
    if not pam_file.exists():
        print(f"\n✗ Data file not found: {pam_file}")
        print("\nThis example requires E. coli data.")
        print("Replace with your own PAM data.")
        return
    
    print("\nLoading E. coli pangenome dataset...")
    df = pd.read_csv(pam_file, sep='\t', index_col=0)
    pam = df.values.T.astype(int)
    strain_names = df.columns.tolist()
    gene_names = df.index.tolist()
    
    print(f"  Dataset: {len(strain_names):,} strains × {len(gene_names):,} genes")
    
    # ========================================================================
    # STEP 1: DIAGNOSTIC (Quick check for heterogeneity)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DIAGNOSTIC")
    print("=" * 70)
    
    diag = diagnose_strain_heterogeneity(pam)
    diag.print_report()
    
    # ========================================================================
    # STEP 2: HETEROGENEITY SCAN (ALWAYS RECOMMENDED)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: HETEROGENEITY SCAN")
    print("=" * 70)
    print("\nThis is the hypothesis test for strain heterogeneity.")
    print("Parameter shifts >100% indicate significant heterogeneity.\n")
    
    scan = strain_heterogeneity_scan(
        pam=pam,
        taxon_names=strain_names,
        gene_names=gene_names,
        tree_method='jaccard_upgma',
        verbose=True
    )
    
    scan.print_summary()
    
    # ========================================================================
    # STEP 3: DECISION BASED ON SCAN RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: ANALYSIS DECISION")
    print("=" * 70)
    
    max_shift = max(abs(v) for v in scan.parameter_shifts.values())
    
    print(f"\nMaximum parameter shift: {max_shift:.1f}%")
    
    if max_shift > 100:
        print("\n→ EXTREME heterogeneity detected")
        print("→ Using stratified modeling\n")
        
        # Use stratified modeling
        stratified = stratified_regime_modeling(
            pam=pam,
            taxon_names=strain_names,
            gene_names=gene_names,
            threshold=None,  # Use median
            tree_method='jaccard_upgma',
            verbose=True
        )
        
        stratified.print_summary()
        
        # Report results
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        
        print("\nHigh-accessory strains:")
        print(f"  n = {stratified.n_high}")
        print(f"  λ = {stratified.high_accessory_result.gain_rate:.4f}")
        print(f"  μ = {stratified.high_accessory_result.loss_rate:.4f}")
        print(f"  π₁ = {stratified.high_accessory_result.equilibrium_frequency:.4f}")
        
        print("\nLow-accessory strains:")
        print(f"  n = {stratified.n_low}")
        print(f"  λ = {stratified.low_accessory_result.gain_rate:.4f}")
        print(f"  μ = {stratified.low_accessory_result.loss_rate:.4f}")
        print(f"  π₁ = {stratified.low_accessory_result.equilibrium_frequency:.4f}")
        
        print("\nInterpretation:")
        print("  Different strain groups exhibit distinct evolutionary regimes.")
        print("  Report results separately for each subpopulation.")
        
    elif max_shift > 50:
        print("\n→ STRONG heterogeneity detected")
        print("→ Consider stratified modeling or report with caveat\n")
        
        result = fit(
            pam=pam,
            taxon_names=strain_names,
            gene_names=gene_names,
            tree_method='jaccard_upgma',
            verbose=True
        )
        
        result.print_summary()
        
        print("\n⚠ CAVEAT: Results may be sensitive to strain sampling.")
        print("   Consider stratified analysis for robust inference.")
        
    elif max_shift > 20:
        print("\n→ MODERATE heterogeneity detected")
        print("→ Global model used, but report with caveat\n")
        
        result = fit(
            pam=pam,
            taxon_names=strain_names,
            gene_names=gene_names,
            tree_method='jaccard_upgma',
            verbose=True
        )
        
        result.print_summary()
        
        print("\n⚠ NOTE: Moderate strain heterogeneity detected.")
        print("   Results should be interpreted with caution.")
        
    else:
        print("\n→ STABLE parameters - homogeneous regime")
        print("→ Standard global model appropriate\n")
        
        result = fit(
            pam=pam,
            taxon_names=strain_names,
            gene_names=gene_names,
            tree_method='jaccard_upgma',
            verbose=True
        )
        
        result.print_summary()
        
        print("\n✓ No significant strain heterogeneity detected.")
        print("   Global model provides robust estimates.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
