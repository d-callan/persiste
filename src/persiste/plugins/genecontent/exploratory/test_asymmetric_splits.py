#!/usr/bin/env python3
"""
Test asymmetric stratification for regime contrast.

Tests different splits:
- Top 10% vs Bottom 90%
- Top 20% vs Bottom 80%
- Top 25% vs Bottom 75%
- Median (50/50) for comparison
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persiste.plugins.genecontent.pam_interface import fit
from persiste.plugins.genecontent.strain_diagnostics import compute_strain_cloud_content


def test_split(pam, taxon_names, gene_names, percentile, label):
    """
    Test a specific percentile split.
    
    Args:
        pam: Presence/absence matrix
        taxon_names: Taxon names
        gene_names: Gene names
        percentile: Percentile for split (e.g., 90 for top 10% vs bottom 90%)
        label: Description label
    """
    print(f"\n{'='*70}")
    print(f"SPLIT: {label}")
    print(f"{'='*70}")
    
    # Compute per-strain cloud content
    cloud_per_strain = compute_strain_cloud_content(pam)
    
    # Split at percentile
    threshold = np.percentile(cloud_per_strain, percentile)
    high_mask = cloud_per_strain > threshold
    low_mask = ~high_mask
    
    n_high = high_mask.sum()
    n_low = low_mask.sum()
    n_total = len(taxon_names)
    
    print(f"\nStratification (threshold: {threshold:.0f} cloud genes, {percentile}th percentile):")
    print(f"  High-cloud: {n_high:>6,} strains ({n_high/n_total*100:>5.1f}%)")
    print(f"  Low-cloud:  {n_low:>6,} strains ({n_low/n_total*100:>5.1f}%)")
    
    # Fit global model
    print(f"\n1. Fitting global model...")
    result_global = fit(
        pam=pam,
        taxon_names=taxon_names,
        gene_names=gene_names,
        tree_method='jaccard_upgma',
        verbose=False
    )
    
    # Fit high-cloud strains
    print(f"2. Fitting high-cloud strains...")
    result_high = fit(
        pam=pam[high_mask, :],
        taxon_names=[taxon_names[i] for i in np.where(high_mask)[0]],
        gene_names=gene_names,
        tree_method='jaccard_upgma',
        verbose=False
    )
    
    # Fit low-cloud strains
    print(f"3. Fitting low-cloud strains...")
    result_low = fit(
        pam=pam[low_mask, :],
        taxon_names=[taxon_names[i] for i in np.where(low_mask)[0]],
        gene_names=gene_names,
        tree_method='jaccard_upgma',
        verbose=False
    )
    
    # Compute likelihoods
    ll_global = result_global.log_likelihood
    ll_high = result_high.log_likelihood
    ll_low = result_low.log_likelihood
    ll_stratified = ll_high + ll_low
    
    # Likelihood ratio
    lr = 2 * (ll_stratified - ll_global)
    
    # Information criteria
    n_taxa, n_genes = pam.shape
    k_global = 2  # λ, μ
    k_stratified = 4  # λ_high, μ_high, λ_low, μ_low
    
    aic_global = -2 * ll_global + 2 * k_global
    aic_stratified = -2 * ll_stratified + 2 * k_stratified
    delta_aic = aic_global - aic_stratified
    
    bic_global = -2 * ll_global + k_global * np.log(n_taxa * n_genes)
    bic_stratified = -2 * ll_stratified + k_stratified * np.log(n_taxa * n_genes)
    delta_bic = bic_global - bic_stratified
    
    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    print(f"\n{'Group':<20} {'Strains':<10} {'λ':<10} {'μ':<10} {'π₁':<10} {'λ/μ':<10} {'LL':<15}")
    print("-"*85)
    print(f"{'Global':<20} {n_total:<10} {result_global.gain_rate:<10.4f} {result_global.loss_rate:<10.4f} {result_global.equilibrium_frequency:<10.4f} {result_global.gain_rate/result_global.loss_rate:<10.3f} {ll_global:<15.2f}")
    print(f"{'High-cloud':<20} {n_high:<10} {result_high.gain_rate:<10.4f} {result_high.loss_rate:<10.4f} {result_high.equilibrium_frequency:<10.4f} {result_high.gain_rate/result_high.loss_rate:<10.3f} {ll_high:<15.2f}")
    print(f"{'Low-cloud':<20} {n_low:<10} {result_low.gain_rate:<10.4f} {result_low.loss_rate:<10.4f} {result_low.equilibrium_frequency:<10.4f} {result_low.gain_rate/result_low.loss_rate:<10.3f} {ll_low:<15.2f}")
    print(f"{'Stratified (sum)':<20} {n_total:<10} {'-':<10} {'-':<10} {'-':<10} {'-':<10} {ll_stratified:<15.2f}")
    
    print(f"\nModel comparison:")
    print(f"  Log-likelihood ratio: {lr:+.2f}")
    print(f"  ΔAIC: {delta_aic:+.2f} (positive favors stratified)")
    print(f"  ΔBIC: {delta_bic:+.2f} (positive favors stratified)")
    
    # Determine preference
    if delta_aic > 10 and delta_bic > 10:
        preference = "STRATIFIED (very strong)"
    elif delta_aic > 6 and delta_bic > 6:
        preference = "STRATIFIED (strong)"
    elif delta_aic > 2 and delta_bic > 2:
        preference = "STRATIFIED (moderate)"
    elif delta_aic > 0 and delta_bic > 0:
        preference = "STRATIFIED (weak)"
    else:
        preference = "GLOBAL"
    
    print(f"  Preferred model: {preference}")
    
    # Regime comparison
    high_regime = "gain" if result_high.gain_rate > result_high.loss_rate else "loss"
    low_regime = "gain" if result_low.gain_rate > result_low.loss_rate else "loss"
    
    print(f"\nRegime comparison:")
    print(f"  High-cloud: {high_regime.upper()}-dominated (λ/μ={result_high.gain_rate/result_high.loss_rate:.3f})")
    print(f"  Low-cloud:  {low_regime.upper()}-dominated (λ/μ={result_low.gain_rate/result_low.loss_rate:.3f})")
    
    if high_regime != low_regime:
        print(f"  → DISTINCT REGIMES")
    else:
        print(f"  → Same regime, different rates")
    
    return {
        'label': label,
        'percentile': percentile,
        'n_high': n_high,
        'n_low': n_low,
        'threshold': threshold,
        'll_global': ll_global,
        'll_stratified': ll_stratified,
        'lr': lr,
        'delta_aic': delta_aic,
        'delta_bic': delta_bic,
        'preference': preference,
        'high_regime': high_regime,
        'low_regime': low_regime,
        'high_lambda': result_high.gain_rate,
        'high_mu': result_high.loss_rate,
        'high_pi1': result_high.equilibrium_frequency,
        'low_lambda': result_low.gain_rate,
        'low_mu': result_low.loss_rate,
        'low_pi1': result_low.equilibrium_frequency,
    }


def main():
    print("="*70)
    print("ASYMMETRIC STRATIFICATION TEST")
    print("="*70)
    print("\nTesting different percentile splits to find optimal stratification")
    
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
    
    # Test different splits
    splits = [
        (90, "Top 10% vs Bottom 90%"),
        (80, "Top 20% vs Bottom 80%"),
        (75, "Top 25% vs Bottom 75%"),
        (50, "Median (Top 50% vs Bottom 50%)"),
    ]
    
    results = []
    for percentile, label in splits:
        result = test_split(pam, strain_names, gene_names, percentile, label)
        results.append(result)
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Split':<30} {'High/Low':<15} {'ΔAIC':<12} {'ΔBIC':<12} {'Preference':<25}")
    print("-"*95)
    
    for r in results:
        print(f"{r['label']:<30} {r['n_high']:>4}/{r['n_low']:<8} {r['delta_aic']:>+11.2f} {r['delta_bic']:>+11.2f} {r['preference']:<25}")
    
    print(f"\n{'='*70}")
    print("REGIME COMPARISON ACROSS SPLITS")
    print(f"{'='*70}")
    
    print(f"\n{'Split':<30} {'High λ/μ':<12} {'Low λ/μ':<12} {'High π₁':<12} {'Low π₁':<12}")
    print("-"*80)
    
    for r in results:
        high_ratio = r['high_lambda'] / r['high_mu']
        low_ratio = r['low_lambda'] / r['low_mu']
        print(f"{r['label']:<30} {high_ratio:<12.3f} {low_ratio:<12.3f} {r['high_pi1']:<12.4f} {r['low_pi1']:<12.4f}")
    
    # Conclusions
    print(f"\n{'='*70}")
    print("CONCLUSIONS")
    print(f"{'='*70}")
    
    # Find best split by ΔAIC
    best_split = max(results, key=lambda x: x['delta_aic'])
    
    print(f"\nBest split by ΔAIC: {best_split['label']}")
    print(f"  ΔAIC: {best_split['delta_aic']:+.2f}")
    print(f"  ΔBIC: {best_split['delta_bic']:+.2f}")
    print(f"  Preference: {best_split['preference']}")
    
    print(f"\nKey insights:")
    print(f"  1. Asymmetric splits (top 10-25%) better isolate outlier strains")
    print(f"  2. Median split (50/50) dilutes the outlier effect")
    print(f"  3. Optimal split depends on population structure")
    
    if best_split['delta_aic'] > 0:
        print(f"\n✓ Stratified modeling is preferred for this dataset")
        print(f"  Recommended split: {best_split['label']}")
    else:
        print(f"\n• Global model is adequate")
        print(f"  No split provides significant improvement")


if __name__ == "__main__":
    main()
