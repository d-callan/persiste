"""
Analysis recipes for frequency-aware pangenome analysis.

Provides explicit strategies for handling sampling bias and rare genes.
These are NOT defaults - they are explicit analysis choices.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class AnalysisRecipe:
    """
    Recipe for frequency-aware analysis.
    
    Attributes:
        name: Recipe name
        description: What this recipe does
        rationale: Why you would use this recipe
        gene_mask: Boolean mask for genes to include (True = include)
        gene_weights: Optional weights for genes (None = equal weights)
    """
    name: str
    description: str
    rationale: str
    gene_mask: np.ndarray
    gene_weights: Optional[np.ndarray] = None
    
    def print_summary(self, n_genes_total: int):
        """Print recipe summary."""
        n_included = self.gene_mask.sum()
        pct_included = 100 * n_included / n_genes_total
        
        print(f"\n{self.name}")
        print(f"  {self.description}")
        print(f"  Rationale: {self.rationale}")
        print(f"  Genes included: {n_included:,} / {n_genes_total:,} ({pct_included:.1f}%)")
        
        if self.gene_weights is not None:
            print(f"  Weighting: Custom (downweights rare genes)")


def core_shell_recipe(pam: np.ndarray, cloud_threshold: float = 0.15) -> AnalysisRecipe:
    """
    Include only core and shell genes, exclude cloud genes.
    
    Reduces sampling bias by excluding very rare genes that inflate gain rates.
    
    Args:
        pam: Presence/absence matrix (n_taxa × n_genes)
        cloud_threshold: Genes present in < this fraction are "cloud" (default: 0.15)
        
    Returns:
        AnalysisRecipe with cloud genes excluded
        
    Example:
        >>> recipe = core_shell_recipe(pam)
        >>> pam_filtered = pam[:, recipe.gene_mask]
    """
    n_taxa = pam.shape[0]
    gene_counts = pam.sum(axis=0)
    
    # Include genes present in ≥ cloud_threshold of strains
    gene_mask = gene_counts >= (cloud_threshold * n_taxa)
    
    return AnalysisRecipe(
        name="Core + Shell Only",
        description=f"Exclude cloud genes (present in <{cloud_threshold*100:.0f}% of strains)",
        rationale="Reduces sampling bias from rare genes that inflate gain rate estimates",
        gene_mask=gene_mask,
    )


def exclude_singletons_recipe(pam: np.ndarray) -> AnalysisRecipe:
    """
    Exclude singleton genes (present in exactly 1 strain).
    
    Removes strain-specific genes that may represent sequencing artifacts,
    recent HGT, or sampling bias.
    
    Args:
        pam: Presence/absence matrix (n_taxa × n_genes)
        
    Returns:
        AnalysisRecipe with singletons excluded
    """
    gene_counts = pam.sum(axis=0)
    gene_mask = gene_counts > 1
    
    return AnalysisRecipe(
        name="Exclude Singletons",
        description="Remove genes present in exactly 1 strain",
        rationale="Eliminates strain-specific genes that may be artifacts or recent acquisitions",
        gene_mask=gene_mask,
    )


def exclude_rare_recipe(
    pam: np.ndarray,
    rare_threshold: Optional[int] = None
) -> AnalysisRecipe:
    """
    Exclude rare genes (present in ≤ threshold strains).
    
    More aggressive than exclude_singletons - removes all rare genes.
    
    Args:
        pam: Presence/absence matrix (n_taxa × n_genes)
        rare_threshold: Genes present in ≤ this many strains are excluded
                       (default: 5% of taxa or 10, whichever is larger)
        
    Returns:
        AnalysisRecipe with rare genes excluded
    """
    n_taxa = pam.shape[0]
    
    if rare_threshold is None:
        rare_threshold = max(10, int(0.05 * n_taxa))
    
    gene_counts = pam.sum(axis=0)
    gene_mask = gene_counts > rare_threshold
    
    return AnalysisRecipe(
        name=f"Exclude Rare (≤{rare_threshold})",
        description=f"Remove genes present in ≤{rare_threshold} strains",
        rationale="Aggressive filtering to minimize sampling bias",
        gene_mask=gene_mask,
    )


def downweight_rare_recipe(
    pam: np.ndarray,
    rare_threshold: Optional[int] = None,
    rare_weight: float = 0.1
) -> AnalysisRecipe:
    """
    Downweight rare genes instead of excluding them.
    
    Keeps all genes but reduces influence of rare ones. More conservative
    than exclusion - retains information while reducing bias.
    
    Args:
        pam: Presence/absence matrix (n_taxa × n_genes)
        rare_threshold: Genes present in ≤ this many strains are downweighted
                       (default: 5% of taxa or 10, whichever is larger)
        rare_weight: Weight for rare genes (default: 0.1)
        
    Returns:
        AnalysisRecipe with rare genes downweighted
        
    Note:
        This recipe requires weighted likelihood computation (not yet implemented).
        For now, it returns a mask like other recipes.
    """
    n_taxa = pam.shape[0]
    
    if rare_threshold is None:
        rare_threshold = max(10, int(0.05 * n_taxa))
    
    gene_counts = pam.sum(axis=0)
    
    # Create weights: 1.0 for common genes, rare_weight for rare genes
    weights = np.ones(len(gene_counts))
    weights[gene_counts <= rare_threshold] = rare_weight
    
    # For now, still use a mask (include all genes)
    # In future, pass weights to likelihood computation
    gene_mask = np.ones(len(gene_counts), dtype=bool)
    
    return AnalysisRecipe(
        name=f"Downweight Rare (≤{rare_threshold})",
        description=f"Reduce weight of genes present in ≤{rare_threshold} strains to {rare_weight}",
        rationale="Conservative approach: retains information while reducing bias",
        gene_mask=gene_mask,
        gene_weights=weights,
    )


def frequency_stratified_recipe(
    pam: np.ndarray,
    n_bins: int = 5
) -> list[AnalysisRecipe]:
    """
    Create multiple recipes stratified by gene frequency.
    
    Useful for understanding how parameter estimates vary with gene frequency.
    
    Args:
        pam: Presence/absence matrix (n_taxa × n_genes)
        n_bins: Number of frequency bins (default: 5)
        
    Returns:
        List of AnalysisRecipe objects, one per frequency bin
        
    Example:
        >>> recipes = frequency_stratified_recipe(pam, n_bins=3)
        >>> for recipe in recipes:
        >>>     pam_bin = pam[:, recipe.gene_mask]
        >>>     # Analyze each frequency bin separately
    """
    gene_counts = pam.sum(axis=0)
    
    # Create frequency bins
    bin_edges = np.percentile(gene_counts, np.linspace(0, 100, n_bins + 1))
    
    recipes = []
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        
        if i == n_bins - 1:
            # Last bin: include upper edge
            gene_mask = (gene_counts >= lower) & (gene_counts <= upper)
        else:
            gene_mask = (gene_counts >= lower) & (gene_counts < upper)
        
        recipes.append(AnalysisRecipe(
            name=f"Frequency Bin {i+1}/{n_bins}",
            description=f"Genes present in {lower:.0f}-{upper:.0f} strains",
            rationale=f"Stratified analysis by gene frequency (bin {i+1}/{n_bins})",
            gene_mask=gene_mask,
        ))
    
    return recipes


def apply_recipe(
    pam: np.ndarray,
    taxon_names: list[str],
    gene_names: list[str],
    recipe: AnalysisRecipe,
    verbose: bool = True
) -> Tuple[np.ndarray, list[str], list[str]]:
    """
    Apply recipe to filter PAM.
    
    Args:
        pam: Presence/absence matrix (n_taxa × n_genes)
        taxon_names: Taxon names
        gene_names: Gene names
        recipe: AnalysisRecipe to apply
        verbose: Print summary
        
    Returns:
        Tuple of (filtered_pam, taxon_names, filtered_gene_names)
    """
    if verbose:
        recipe.print_summary(pam.shape[1])
    
    # Apply gene mask
    pam_filtered = pam[:, recipe.gene_mask]
    gene_names_filtered = [gene_names[i] for i in np.where(recipe.gene_mask)[0]]
    
    if verbose:
        print(f"  Filtered PAM: {pam_filtered.shape[0]} taxa × {pam_filtered.shape[1]} genes")
    
    return pam_filtered, taxon_names, gene_names_filtered


def compare_recipes(
    pam: np.ndarray,
    recipes: list[AnalysisRecipe],
    recipe_names: Optional[list[str]] = None
):
    """
    Compare multiple recipes side-by-side.
    
    Args:
        pam: Presence/absence matrix (n_taxa × n_genes)
        recipes: List of recipes to compare
        recipe_names: Optional custom names for recipes
    """
    if recipe_names is None:
        recipe_names = [r.name for r in recipes]
    
    print("=" * 70)
    print("RECIPE COMPARISON")
    print("=" * 70)
    
    n_genes = pam.shape[1]
    
    print(f"\n{'Recipe':<30} {'Genes included':<20} {'% of total':<15}")
    print("-" * 70)
    
    for name, recipe in zip(recipe_names, recipes):
        n_included = recipe.gene_mask.sum()
        pct = 100 * n_included / n_genes
        print(f"{name:<30} {n_included:>10,} / {n_genes:<6,} {pct:>10.1f}%")
    
    print("\n" + "=" * 70)
