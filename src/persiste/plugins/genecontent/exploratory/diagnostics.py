"""
Diagnostic tools for detecting strain heterogeneity in pangenome analysis.

Focuses on detecting whether different strains have different gene content dynamics,
which can cause regime shifts and parameter instability.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class StrainHeterogeneityDiagnostic:
    """
    Diagnostic information about sampling bias in pangenome data.
    
    Attributes:
        n_taxa: Number of taxa/strains
        n_genes: Number of gene families
        n_singletons: Number of genes present in exactly 1 strain
        n_rare: Number of genes present in ≤ rare_threshold strains
        rare_threshold: Threshold for defining "rare" genes
        pct_singletons: Percentage of genes that are singletons
        pct_rare: Percentage of genes that are rare
        mean_prevalence: Mean number of strains per gene
        core_genes: Number of genes present in all strains
        shell_genes: Number of genes present in 15-95% of strains
        cloud_genes: Number of genes present in <15% of strains
        warning_level: Severity of sampling bias (none, low, medium, high)
        recommendations: List of recommended analysis strategies
    """
    n_taxa: int
    n_genes: int
    n_singletons: int
    n_rare: int
    rare_threshold: int
    pct_singletons: float
    pct_rare: float
    mean_prevalence: float
    core_genes: int
    shell_genes: int
    cloud_genes: int
    warning_level: str
    recommendations: list[str]
    
    def print_report(self):
        """Print diagnostic report."""
        print("=" * 70)
        print("SAMPLING BIAS DIAGNOSTIC")
        print("=" * 70)
        
        print(f"\nDataset:")
        print(f"  Taxa: {self.n_taxa:,}")
        print(f"  Gene families: {self.n_genes:,}")
        print(f"  Mean prevalence: {self.mean_prevalence:.1f} strains/gene")
        
        print(f"\nGene frequency distribution:")
        print(f"  Core (100%):        {self.core_genes:>8,} ({self.core_genes/self.n_genes*100:>5.1f}%)")
        print(f"  Shell (15-95%):     {self.shell_genes:>8,} ({self.shell_genes/self.n_genes*100:>5.1f}%)")
        print(f"  Cloud (<15%):       {self.cloud_genes:>8,} ({self.cloud_genes/self.n_genes*100:>5.1f}%)")
        print(f"  Singletons (1):     {self.n_singletons:>8,} ({self.pct_singletons:>5.1f}%)")
        print(f"  Rare (≤{self.rare_threshold}):        {self.n_rare:>8,} ({self.pct_rare:>5.1f}%)")
        
        print(f"\n⚠ Warning level: {self.warning_level.upper()}")
        
        if self.warning_level != "none":
            print("\nPotential issues:")
            if self.pct_rare > 50:
                print(f"  • {self.pct_rare:.1f}% of genes are rare (≤{self.rare_threshold} strains)")
                print("    → Gain rate estimates may be inflated")
                print("    → Sampling bias toward recently acquired genes")
            if self.pct_singletons > 30:
                print(f"  • {self.pct_singletons:.1f}% of genes are singletons")
                print("    → Strong sampling bias toward strain-specific genes")
            if self.core_genes < 100:
                print(f"  • Only {self.core_genes} core genes")
                print("    → Dataset may be too diverse or incomplete")
        
        if self.recommendations:
            print("\nRecommended analysis strategies:")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "=" * 70)


def diagnose_sampling_bias(
    pam: np.ndarray,
    rare_threshold: Optional[int] = None
) -> SamplingDiagnostic:
    """
    Diagnose sampling bias in presence/absence matrix.
    
    Args:
        pam: Presence/absence matrix (n_taxa × n_genes)
        rare_threshold: Threshold for "rare" genes (default: 5% of taxa or 10, whichever is larger)
        
    Returns:
        SamplingDiagnostic with warnings and recommendations
    """
    n_taxa, n_genes = pam.shape
    
    # Compute gene frequencies
    gene_counts = pam.sum(axis=0)
    
    # Set rare threshold
    if rare_threshold is None:
        rare_threshold = max(10, int(0.05 * n_taxa))
    
    # Count singletons and rare genes
    n_singletons = np.sum(gene_counts == 1)
    n_rare = np.sum(gene_counts <= rare_threshold)
    
    pct_singletons = 100 * n_singletons / n_genes
    pct_rare = 100 * n_rare / n_genes
    
    mean_prevalence = gene_counts.mean()
    
    # Pangenome structure (Tettelin et al. definitions)
    core_genes = np.sum(gene_counts == n_taxa)
    shell_genes = np.sum((gene_counts >= 0.15 * n_taxa) & (gene_counts < 0.95 * n_taxa))
    cloud_genes = np.sum(gene_counts < 0.15 * n_taxa)
    
    # Determine warning level
    if pct_rare > 70:
        warning_level = "high"
    elif pct_rare > 50:
        warning_level = "medium"
    elif pct_rare > 30:
        warning_level = "low"
    else:
        warning_level = "none"
    
    # Generate recommendations
    recommendations = []
    
    if pct_rare > 50:
        recommendations.append(
            "Use 'core_shell' recipe to exclude cloud genes (reduces sampling bias)"
        )
    
    if pct_singletons > 30:
        recommendations.append(
            "Use 'exclude_singletons' recipe to remove strain-specific genes"
        )
    
    if pct_rare > 70:
        recommendations.append(
            "Use 'downweight_rare' recipe to reduce influence of rare genes"
        )
    
    if warning_level != "none":
        recommendations.append(
            "Compare results across multiple frequency thresholds"
        )
        recommendations.append(
            "Report π₁ (stationary frequency) prominently - it explains the regime"
        )
    
    if not recommendations:
        recommendations.append("No sampling bias detected - standard analysis recommended")
    
    return SamplingDiagnostic(
        n_taxa=n_taxa,
        n_genes=n_genes,
        n_singletons=n_singletons,
        n_rare=n_rare,
        rare_threshold=rare_threshold,
        pct_singletons=pct_singletons,
        pct_rare=pct_rare,
        mean_prevalence=mean_prevalence,
        core_genes=core_genes,
        shell_genes=shell_genes,
        cloud_genes=cloud_genes,
        warning_level=warning_level,
        recommendations=recommendations,
    )


def compare_full_vs_subset_diagnostics(
    pam_full: np.ndarray,
    pam_subset: np.ndarray,
    subset_name: str = "subset"
):
    """
    Compare diagnostics between full dataset and subset.
    
    Useful for understanding how sampling affects inferred parameters.
    
    Args:
        pam_full: Full presence/absence matrix
        pam_subset: Subset presence/absence matrix
        subset_name: Name for the subset
    """
    print("=" * 70)
    print("FULL vs SUBSET COMPARISON")
    print("=" * 70)
    
    diag_full = diagnose_sampling_bias(pam_full)
    diag_subset = diagnose_sampling_bias(pam_subset)
    
    print(f"\n{'Metric':<30} {'Full':<15} {subset_name:<15} {'Change':<10}")
    print("-" * 70)
    
    print(f"{'Taxa':<30} {diag_full.n_taxa:<15,} {diag_subset.n_taxa:<15,}")
    print(f"{'Genes':<30} {diag_full.n_genes:<15,} {diag_subset.n_genes:<15,}")
    print(f"{'Mean prevalence':<30} {diag_full.mean_prevalence:<15.1f} {diag_subset.mean_prevalence:<15.1f}")
    
    print(f"\n{'% Singletons':<30} {diag_full.pct_singletons:<15.1f} {diag_subset.pct_singletons:<15.1f} {diag_subset.pct_singletons - diag_full.pct_singletons:+.1f}")
    print(f"{'% Rare':<30} {diag_full.pct_rare:<15.1f} {diag_subset.pct_rare:<15.1f} {diag_subset.pct_rare - diag_full.pct_rare:+.1f}")
    print(f"{'% Core':<30} {diag_full.core_genes/diag_full.n_genes*100:<15.1f} {diag_subset.core_genes/diag_subset.n_genes*100:<15.1f}")
    
    print(f"\n{'Warning level':<30} {diag_full.warning_level:<15} {diag_subset.warning_level:<15}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if diag_subset.pct_rare > diag_full.pct_rare:
        print("\n⚠ Subset has MORE rare genes than full dataset")
        print("  → Sampling bias is STRONGER in subset")
        print("  → Gain rate will appear HIGHER in subset")
        print("  → This is expected: subsampling enriches for common genes")
    
    if diag_subset.pct_singletons > diag_full.pct_singletons:
        print("\n⚠ Subset has MORE singletons than full dataset")
        print("  → Strain-specific genes are overrepresented")
        print("  → Use stratified sampling or frequency-aware recipes")
