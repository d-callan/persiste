#!/usr/bin/env python3
"""
Analyze strain-level heterogeneity in E. coli pangenome.

Key question: Do a few strains with many cloud genes drive the gain-dominated regime?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persiste.plugins.genecontent.pam_interface import fit


def classify_genes_by_frequency(pam, cloud_threshold=0.15):
    """
    Classify genes as core/shell/cloud based on frequency.
    
    Args:
        pam: Presence/absence matrix (n_strains × n_genes)
        cloud_threshold: Genes present in < this fraction are "cloud"
        
    Returns:
        Dictionary with gene classifications
    """
    n_strains = pam.shape[0]
    gene_counts = pam.sum(axis=0)
    
    # Classify genes
    core_mask = gene_counts == n_strains
    cloud_mask = gene_counts < (cloud_threshold * n_strains)
    shell_mask = ~core_mask & ~cloud_mask
    
    return {
        'core': core_mask,
        'shell': shell_mask,
        'cloud': cloud_mask,
        'gene_counts': gene_counts
    }


def analyze_strain_heterogeneity(pam, strain_names, gene_names):
    """
    Analyze how cloud gene content varies across strains.
    """
    print("=" * 70)
    print("STRAIN-LEVEL HETEROGENEITY ANALYSIS")
    print("=" * 70)
    
    n_strains, n_genes = pam.shape
    
    # Classify genes
    gene_classes = classify_genes_by_frequency(pam, cloud_threshold=0.15)
    
    n_core = gene_classes['core'].sum()
    n_shell = gene_classes['shell'].sum()
    n_cloud = gene_classes['cloud'].sum()
    
    print(f"\nDataset: {n_strains:,} strains × {n_genes:,} genes")
    print(f"\nGene classification:")
    print(f"  Core (100%):        {n_core:>8,} ({n_core/n_genes*100:>5.1f}%)")
    print(f"  Shell (15-95%):     {n_shell:>8,} ({n_shell/n_genes*100:>5.1f}%)")
    print(f"  Cloud (<15%):       {n_cloud:>8,} ({n_cloud/n_genes*100:>5.1f}%)")
    
    # Compute per-strain statistics
    print("\n" + "=" * 70)
    print("PER-STRAIN GENE CONTENT")
    print("=" * 70)
    
    strain_stats = []
    
    for i, strain in enumerate(strain_names):
        total_genes = pam[i, :].sum()
        core_genes = pam[i, gene_classes['core']].sum()
        shell_genes = pam[i, gene_classes['shell']].sum()
        cloud_genes = pam[i, gene_classes['cloud']].sum()
        
        strain_stats.append({
            'strain': strain,
            'total': total_genes,
            'core': core_genes,
            'shell': shell_genes,
            'cloud': cloud_genes,
            'pct_cloud': 100 * cloud_genes / total_genes if total_genes > 0 else 0,
        })
    
    df_strains = pd.DataFrame(strain_stats)
    
    # Summary statistics
    print(f"\nStrain gene content summary:")
    print(f"  Total genes per strain:")
    print(f"    Mean:   {df_strains['total'].mean():>8.1f}")
    print(f"    Median: {df_strains['total'].median():>8.1f}")
    print(f"    Range:  {df_strains['total'].min():>8.0f} - {df_strains['total'].max():>8.0f}")
    
    print(f"\n  Cloud genes per strain:")
    print(f"    Mean:   {df_strains['cloud'].mean():>8.1f}")
    print(f"    Median: {df_strains['cloud'].median():>8.1f}")
    print(f"    Range:  {df_strains['cloud'].min():>8.0f} - {df_strains['cloud'].max():>8.0f}")
    
    print(f"\n  % Cloud genes per strain:")
    print(f"    Mean:   {df_strains['pct_cloud'].mean():>8.1f}%")
    print(f"    Median: {df_strains['pct_cloud'].median():>8.1f}%")
    print(f"    Range:  {df_strains['pct_cloud'].min():>8.1f}% - {df_strains['pct_cloud'].max():>8.1f}%")
    
    # Identify outliers
    print("\n" + "=" * 70)
    print("STRAIN STRATIFICATION")
    print("=" * 70)
    
    # Quartiles
    q25 = df_strains['cloud'].quantile(0.25)
    q50 = df_strains['cloud'].quantile(0.50)
    q75 = df_strains['cloud'].quantile(0.75)
    q90 = df_strains['cloud'].quantile(0.90)
    q95 = df_strains['cloud'].quantile(0.95)
    
    print(f"\nCloud gene quartiles:")
    print(f"  Q25 (25th percentile): {q25:>8.0f}")
    print(f"  Q50 (median):          {q50:>8.0f}")
    print(f"  Q75 (75th percentile): {q75:>8.0f}")
    print(f"  Q90 (90th percentile): {q90:>8.0f}")
    print(f"  Q95 (95th percentile): {q95:>8.0f}")
    
    # Stratify strains
    low_cloud = df_strains['cloud'] <= q25
    med_cloud = (df_strains['cloud'] > q25) & (df_strains['cloud'] <= q75)
    high_cloud = df_strains['cloud'] > q75
    very_high_cloud = df_strains['cloud'] > q90
    
    print(f"\nStrain groups:")
    print(f"  Low cloud (≤Q25):     {low_cloud.sum():>6,} strains ({low_cloud.sum()/n_strains*100:>5.1f}%)")
    print(f"  Medium cloud (Q25-Q75): {med_cloud.sum():>6,} strains ({med_cloud.sum()/n_strains*100:>5.1f}%)")
    print(f"  High cloud (>Q75):    {high_cloud.sum():>6,} strains ({high_cloud.sum()/n_strains*100:>5.1f}%)")
    print(f"  Very high (>Q90):     {very_high_cloud.sum():>6,} strains ({very_high_cloud.sum()/n_strains*100:>5.1f}%)")
    
    # Show top strains
    print(f"\nTop 10 strains by cloud gene count:")
    top_strains = df_strains.nlargest(10, 'cloud')
    print(f"\n{'Strain':<20} {'Total':<10} {'Core':<10} {'Shell':<10} {'Cloud':<10} {'% Cloud':<10}")
    print("-" * 80)
    for _, row in top_strains.iterrows():
        print(f"{row['strain']:<20} {row['total']:<10.0f} {row['core']:<10.0f} {row['shell']:<10.0f} {row['cloud']:<10.0f} {row['pct_cloud']:<10.1f}")
    
    # Key insight: How much do high-cloud strains contribute to total cloud genes?
    print("\n" + "=" * 70)
    print("CLOUD GENE CONTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Total cloud gene observations (sum across all strains)
    total_cloud_observations = pam[:, gene_classes['cloud']].sum()
    
    # Cloud observations from high-cloud strains
    high_cloud_idx = df_strains[high_cloud].index
    cloud_from_high = pam[high_cloud_idx, :][:, gene_classes['cloud']].sum()
    
    very_high_cloud_idx = df_strains[very_high_cloud].index
    cloud_from_very_high = pam[very_high_cloud_idx, :][:, gene_classes['cloud']].sum()
    
    print(f"\nTotal cloud gene observations: {total_cloud_observations:,}")
    print(f"\nContribution by strain group:")
    print(f"  High cloud strains (>Q75, {high_cloud.sum()} strains):")
    print(f"    Cloud observations: {cloud_from_high:,} ({cloud_from_high/total_cloud_observations*100:.1f}%)")
    print(f"  Very high cloud strains (>Q90, {very_high_cloud.sum()} strains):")
    print(f"    Cloud observations: {cloud_from_very_high:,} ({cloud_from_very_high/total_cloud_observations*100:.1f}%)")
    
    return df_strains, gene_classes


def test_strain_removal(pam, strain_names, gene_names, df_strains, gene_classes):
    """
    Test how parameter estimates change when removing high-cloud strains.
    """
    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY TO HIGH-CLOUD STRAINS")
    print("=" * 70)
    
    # Baseline: full dataset
    print("\nBaseline: Full dataset (1,324 strains)")
    result_full = fit(
        pam=pam,
        tree=None,
        taxon_names=strain_names,
        gene_names=gene_names,
        tree_method='jaccard_upgma',
        verbose=False
    )
    
    print(f"  λ: {result_full.gain_rate:.4f}, μ: {result_full.loss_rate:.4f}")
    print(f"  π₁: {result_full.equilibrium_frequency:.4f}")
    print(f"  λ/μ: {result_full.gain_rate/result_full.loss_rate:.3f}")
    
    # Remove top 10% high-cloud strains
    q90 = df_strains['cloud'].quantile(0.90)
    keep_idx = df_strains['cloud'] <= q90
    
    pam_filtered = pam[keep_idx, :]
    strains_filtered = [strain_names[i] for i in df_strains[keep_idx].index]
    
    print(f"\nFiltered: Remove top 10% high-cloud strains ({(~keep_idx).sum()} removed)")
    print(f"  Remaining: {keep_idx.sum()} strains")
    
    result_filtered = fit(
        pam=pam_filtered,
        tree=None,
        taxon_names=strains_filtered,
        gene_names=gene_names,
        tree_method='jaccard_upgma',
        verbose=False
    )
    
    print(f"  λ: {result_filtered.gain_rate:.4f}, μ: {result_filtered.loss_rate:.4f}")
    print(f"  π₁: {result_filtered.equilibrium_frequency:.4f}")
    print(f"  λ/μ: {result_filtered.gain_rate/result_filtered.loss_rate:.3f}")
    
    # Remove top 25% high-cloud strains
    q75 = df_strains['cloud'].quantile(0.75)
    keep_idx_25 = df_strains['cloud'] <= q75
    
    pam_filtered_25 = pam[keep_idx_25, :]
    strains_filtered_25 = [strain_names[i] for i in df_strains[keep_idx_25].index]
    
    print(f"\nFiltered: Remove top 25% high-cloud strains ({(~keep_idx_25).sum()} removed)")
    print(f"  Remaining: {keep_idx_25.sum()} strains")
    
    result_filtered_25 = fit(
        pam=pam_filtered_25,
        tree=None,
        taxon_names=strains_filtered_25,
        gene_names=gene_names,
        tree_method='jaccard_upgma',
        verbose=False
    )
    
    print(f"  λ: {result_filtered_25.gain_rate:.4f}, μ: {result_filtered_25.loss_rate:.4f}")
    print(f"  π₁: {result_filtered_25.equilibrium_frequency:.4f}")
    print(f"  λ/μ: {result_filtered_25.gain_rate/result_filtered_25.loss_rate:.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Dataset':<30} {'Strains':<10} {'λ':<10} {'μ':<10} {'π₁':<10} {'λ/μ':<10}")
    print("-" * 80)
    print(f"{'Full':<30} {len(strain_names):<10} {result_full.gain_rate:<10.4f} {result_full.loss_rate:<10.4f} {result_full.equilibrium_frequency:<10.4f} {result_full.gain_rate/result_full.loss_rate:<10.3f}")
    print(f"{'Remove top 10% cloud':<30} {keep_idx.sum():<10} {result_filtered.gain_rate:<10.4f} {result_filtered.loss_rate:<10.4f} {result_filtered.equilibrium_frequency:<10.4f} {result_filtered.gain_rate/result_filtered.loss_rate:<10.3f}")
    print(f"{'Remove top 25% cloud':<30} {keep_idx_25.sum():<10} {result_filtered_25.gain_rate:<10.4f} {result_filtered_25.loss_rate:<10.4f} {result_filtered_25.equilibrium_frequency:<10.4f} {result_filtered_25.gain_rate/result_filtered_25.loss_rate:<10.3f}")


def main():
    print("=" * 70)
    print("E. COLI STRAIN HETEROGENEITY ANALYSIS")
    print("=" * 70)
    print("\nQuestion: Do a few strains with many cloud genes drive")
    print("          the apparent gain-dominated regime?")
    
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
    
    # Analyze strain heterogeneity
    df_strains, gene_classes = analyze_strain_heterogeneity(pam, strain_names, gene_names)
    
    # Test parameter sensitivity
    test_strain_removal(pam, strain_names, gene_names, df_strains, gene_classes)
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    
    print("\n1. Strain heterogeneity in cloud gene content:")
    print("   • Check if distribution is uniform or skewed")
    print("   • Identify if few strains have disproportionate cloud genes")
    
    print("\n2. Impact on parameter estimates:")
    print("   • Test if removing high-cloud strains changes regime")
    print("   • Quantify contribution of outlier strains")
    
    print("\n3. Biological interpretation:")
    print("   • High-cloud strains may be recent acquisitions")
    print("   • Or represent distinct ecological niches")
    print("   • Or have different HGT rates")


if __name__ == "__main__":
    main()
