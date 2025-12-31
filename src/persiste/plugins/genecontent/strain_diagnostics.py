"""
Strain heterogeneity diagnostics for pangenome analysis.

Detects whether different strains have different gene content dynamics,
which can cause regime shifts and parameter instability.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class StrainHeterogeneityDiagnostic:
    """
    Diagnostic information about strain-level heterogeneity in pangenome data.
    
    Attributes:
        n_taxa: Number of taxa/strains
        n_genes: Number of gene families
        n_core: Number of core genes (present in all strains)
        n_shell: Number of shell genes (15-95% prevalence)
        n_cloud: Number of cloud genes (<15% prevalence)
        cloud_per_strain_mean: Mean cloud genes per strain
        cloud_per_strain_median: Median cloud genes per strain
        cloud_per_strain_std: Standard deviation of cloud genes per strain
        cloud_per_strain_cv: Coefficient of variation (std/mean)
        cloud_per_strain_range: (min, max) cloud genes per strain
        heterogeneity_level: Severity (none, low, medium, high, extreme)
        top_10pct_contribution: % of cloud observations from top 10% strains
        recommendations: List of recommended analysis strategies
    """
    n_taxa: int
    n_genes: int
    n_core: int
    n_shell: int
    n_cloud: int
    cloud_per_strain_mean: float
    cloud_per_strain_median: float
    cloud_per_strain_std: float
    cloud_per_strain_cv: float
    cloud_per_strain_range: Tuple[int, int]
    heterogeneity_level: str
    top_10pct_contribution: float
    recommendations: list[str]
    
    def print_report(self):
        """Print diagnostic report."""
        print("=" * 70)
        print("STRAIN HETEROGENEITY DIAGNOSTIC")
        print("=" * 70)
        
        print(f"\nDataset:")
        print(f"  Taxa: {self.n_taxa:,}")
        print(f"  Gene families: {self.n_genes:,}")
        
        print(f"\nGene frequency distribution:")
        print(f"  Core (100%):        {self.n_core:>8,} ({self.n_core/self.n_genes*100:>5.1f}%)")
        print(f"  Shell (15-95%):     {self.n_shell:>8,} ({self.n_shell/self.n_genes*100:>5.1f}%)")
        print(f"  Cloud (<15%):       {self.n_cloud:>8,} ({self.n_cloud/self.n_genes*100:>5.1f}%)")
        
        print(f"\nPer-strain cloud gene content:")
        print(f"  Mean:   {self.cloud_per_strain_mean:>8.1f}")
        print(f"  Median: {self.cloud_per_strain_median:>8.1f}")
        print(f"  Std:    {self.cloud_per_strain_std:>8.1f}")
        print(f"  CV:     {self.cloud_per_strain_cv:>8.2f}")
        print(f"  Range:  {self.cloud_per_strain_range[0]:>8,} - {self.cloud_per_strain_range[1]:>8,}")
        
        print(f"\nStrain heterogeneity:")
        print(f"  Top 10% strains contribute: {self.top_10pct_contribution:.1f}% of cloud observations")
        print(f"  Heterogeneity level: {self.heterogeneity_level.upper()}")
        
        if self.heterogeneity_level != "none":
            print("\n⚠ POTENTIAL REGIME HETEROGENEITY DETECTED")
            print("\nImplications:")
            if self.cloud_per_strain_cv > 0.5:
                print(f"  • High variance in cloud gene content (CV={self.cloud_per_strain_cv:.2f})")
                print("    → Different strains may have different gene dynamics")
            if self.top_10pct_contribution > 20:
                print(f"  • Top 10% strains contribute {self.top_10pct_contribution:.1f}% of cloud genes")
                print("    → Parameter estimates may be driven by outlier strains")
            if self.heterogeneity_level in ["high", "extreme"]:
                print("  • Strong evidence for multiple evolutionary regimes")
                print("    → Single global model may be inappropriate")
        
        if self.recommendations:
            print("\nRecommended analysis strategies:")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "=" * 70)


def diagnose_strain_heterogeneity(
    pam: np.ndarray,
    cloud_threshold: float = 0.15
) -> StrainHeterogeneityDiagnostic:
    """
    Diagnose strain-level heterogeneity in presence/absence matrix.
    
    Args:
        pam: Presence/absence matrix (n_taxa × n_genes)
        cloud_threshold: Genes present in < this fraction are "cloud"
        
    Returns:
        StrainHeterogeneityDiagnostic with warnings and recommendations
    """
    n_taxa, n_genes = pam.shape
    
    # Classify genes by frequency
    gene_counts = pam.sum(axis=0)
    n_core = np.sum(gene_counts == n_taxa)
    n_cloud = np.sum(gene_counts < (cloud_threshold * n_taxa))
    n_shell = n_genes - n_core - n_cloud
    
    # Compute per-strain cloud gene counts
    cloud_mask = gene_counts < (cloud_threshold * n_taxa)
    cloud_per_strain = pam[:, cloud_mask].sum(axis=1)
    
    mean_cloud = cloud_per_strain.mean()
    median_cloud = np.median(cloud_per_strain)
    std_cloud = cloud_per_strain.std()
    cv_cloud = std_cloud / mean_cloud if mean_cloud > 0 else 0
    range_cloud = (int(cloud_per_strain.min()), int(cloud_per_strain.max()))
    
    # Compute top 10% contribution
    q90 = np.percentile(cloud_per_strain, 90)
    top_10pct_mask = cloud_per_strain > q90
    total_cloud_obs = pam[:, cloud_mask].sum()
    top_10pct_obs = pam[top_10pct_mask, :][:, cloud_mask].sum()
    top_10pct_contribution = 100 * top_10pct_obs / total_cloud_obs if total_cloud_obs > 0 else 0
    
    # Determine heterogeneity level
    if cv_cloud < 0.3 and top_10pct_contribution < 15:
        heterogeneity_level = "none"
    elif cv_cloud < 0.5 and top_10pct_contribution < 20:
        heterogeneity_level = "low"
    elif cv_cloud < 0.8 and top_10pct_contribution < 25:
        heterogeneity_level = "medium"
    elif cv_cloud < 1.2 or top_10pct_contribution < 30:
        heterogeneity_level = "high"
    else:
        heterogeneity_level = "extreme"
    
    # Generate recommendations
    recommendations = []
    
    if heterogeneity_level in ["none", "low"]:
        recommendations.append("Standard global model appropriate - low strain heterogeneity")
    
    if heterogeneity_level in ["medium", "high", "extreme"]:
        recommendations.append(
            "Run 'strain_heterogeneity_scan' to test parameter stability"
        )
    
    if heterogeneity_level in ["high", "extreme"]:
        recommendations.append(
            "Consider 'stratified_regime_modeling' to model subpopulations separately"
        )
        recommendations.append(
            "Use 'regime_contrast_test' to formally test for multiple regimes"
        )
    
    if top_10pct_contribution > 20:
        recommendations.append(
            f"Top 10% strains contribute {top_10pct_contribution:.1f}% of cloud genes - check for outliers"
        )
    
    if cv_cloud > 0.8:
        recommendations.append(
            f"High variance in cloud content (CV={cv_cloud:.2f}) suggests distinct lineages"
        )
    
    return StrainHeterogeneityDiagnostic(
        n_taxa=n_taxa,
        n_genes=n_genes,
        n_core=n_core,
        n_shell=n_shell,
        n_cloud=n_cloud,
        cloud_per_strain_mean=mean_cloud,
        cloud_per_strain_median=median_cloud,
        cloud_per_strain_std=std_cloud,
        cloud_per_strain_cv=cv_cloud,
        cloud_per_strain_range=range_cloud,
        heterogeneity_level=heterogeneity_level,
        top_10pct_contribution=top_10pct_contribution,
        recommendations=recommendations,
    )


def compute_strain_cloud_content(
    pam: np.ndarray,
    cloud_threshold: float = 0.15
) -> np.ndarray:
    """
    Compute per-strain cloud gene counts.
    
    Args:
        pam: Presence/absence matrix (n_taxa × n_genes)
        cloud_threshold: Genes present in < this fraction are "cloud"
        
    Returns:
        Array of cloud gene counts per strain (n_taxa,)
    """
    n_taxa = pam.shape[0]
    gene_counts = pam.sum(axis=0)
    cloud_mask = gene_counts < (cloud_threshold * n_taxa)
    return pam[:, cloud_mask].sum(axis=1)
