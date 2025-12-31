"""
Recipe 3: Lineage-Conditioned CN Volatility

Question: "Do some clades experience elevated dosage turnover?"

This is descriptive, not causal: "This lineage exhibits elevated CN volatility."
"""

from typing import Dict, Optional, Union, List
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from persiste.core.trees import Tree
from persiste.plugins.copynumber.cn_interface import (
    fit,
    fit_null_model,
    likelihood_ratio_test,
)


@dataclass
class LineageVolatilityReport:
    """
    Report from lineage-conditioned volatility analysis.
    
    Attributes:
        theta: Estimated constraint parameter
        theta_ci: Confidence interval for theta
        target_lineage: Lineage identifier
        log_likelihood_null: Null model log-likelihood
        log_likelihood_alt: Alternative model log-likelihood
        lrt_statistic: Likelihood ratio test statistic
        p_value: LRT p-value
        interpretation: Human-readable interpretation
        recommendation: What to do next
    """
    theta: float
    theta_ci: tuple
    target_lineage: str
    log_likelihood_null: float
    log_likelihood_alt: float
    lrt_statistic: float
    p_value: float
    interpretation: str
    recommendation: str


def lineage_volatility_test(
    cn_matrix: Union[np.ndarray, str, Path],
    family_names: Optional[List[str]] = None,
    taxon_names: Optional[List[str]] = None,
    tree: Optional[Union[Tree, str, Path]] = None,
    target_lineage: str = "host_associated",
    baseline_type: str = 'hierarchical',
    theta_grid: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> LineageVolatilityReport:
    """
    Recipe 3: Lineage-Conditioned CN Volatility
    
    Tests whether specific clades/lineages exhibit elevated copy number
    turnover relative to the rest of the tree.
    
    Constraint: HostConditionedVolatilityConstraint
    - θ < 0 → suppressed CN volatility in target lineage
    - θ = 0 → neutral (no lineage effect)
    - θ > 0 → elevated CN volatility in target lineage
    
    Important framing: This is DESCRIPTIVE, not causal.
    Say: "This lineage exhibits elevated CN volatility"
    NOT: "Host association causes CN volatility"
    
    Interpretation:
    - Elevated volatility → Environmental adaptation, niche-specific genes
    - Suppressed volatility → Stable environment, core functions
    
    Args:
        cn_matrix: Copy number matrix (n_families, n_taxa) or path to file
        family_names: List of gene family names
        taxon_names: List of taxon names
        tree: Phylogenetic tree or path to tree file
        target_lineage: Lineage identifier (e.g., "host_associated", "clade_A")
        baseline_type: 'hierarchical' (recommended) or 'global'
        theta_grid: Grid of theta values to test (default: -2 to 2)
        verbose: Print interpretation
    
    Returns:
        LineageVolatilityReport with test results and interpretation
    
    Example:
        >>> report = lineage_volatility_test(
        ...     cn_matrix="data/cn_matrix.tsv",
        ...     tree="data/tree.nwk",
        ...     target_lineage="host_associated",
        ... )
        >>> if report.p_value < 0.05:
        ...     print(f"Lineage {report.target_lineage} shows elevated CN volatility")
    
    Note:
        Currently uses a placeholder for lineage identification.
        In production, this should integrate with tree metadata/annotations.
    """
    # Fit null model
    null_result = fit_null_model(
        cn_matrix=cn_matrix,
        family_names=family_names,
        taxon_names=taxon_names,
        tree=tree,
        baseline_type=baseline_type,
    )
    
    # Grid search for best theta
    if theta_grid is None:
        theta_grid = np.linspace(-2, 2, 21)
    
    best_theta = None
    best_ll = -np.inf
    
    for theta in theta_grid:
        alt_result = fit(
            cn_matrix=cn_matrix,
            family_names=family_names,
            taxon_names=taxon_names,
            tree=tree,
            baseline_type=baseline_type,
            constraint_type='host_conditioned',
            theta=theta,
        )
        
        if alt_result.log_likelihood > best_ll:
            best_ll = alt_result.log_likelihood
            best_theta = theta
    
    # Fit with best theta
    alt_result = fit(
        cn_matrix=cn_matrix,
        family_names=family_names,
        taxon_names=taxon_names,
        tree=tree,
        baseline_type=baseline_type,
        constraint_type='host_conditioned',
        theta=best_theta,
    )
    
    # Likelihood ratio test
    lrt_result = likelihood_ratio_test(alt_result, null_result)
    
    # Estimate confidence interval
    theta_se = 0.3  # Placeholder
    theta_ci = (best_theta - 1.96 * theta_se, best_theta + 1.96 * theta_se)
    
    # Generate interpretation
    interpretation = _generate_lineage_interpretation(
        best_theta, target_lineage, lrt_result['p_value'], lrt_result['lrt_statistic']
    )
    
    # Generate recommendation
    recommendation = _generate_lineage_recommendation(
        best_theta, target_lineage, lrt_result['p_value']
    )
    
    if verbose:
        print("=" * 70)
        print("Recipe 3: Lineage-Conditioned CN Volatility")
        print("=" * 70)
        print(interpretation)
        print("\n" + recommendation)
        print("=" * 70)
    
    return LineageVolatilityReport(
        theta=best_theta,
        theta_ci=theta_ci,
        target_lineage=target_lineage,
        log_likelihood_null=null_result.log_likelihood,
        log_likelihood_alt=alt_result.log_likelihood,
        lrt_statistic=lrt_result['lrt_statistic'],
        p_value=lrt_result['p_value'],
        interpretation=interpretation,
        recommendation=recommendation,
    )


def _generate_lineage_interpretation(
    theta: float,
    lineage: str,
    p_value: float,
    lrt_stat: float,
) -> str:
    """Generate human-readable interpretation."""
    
    lines = []
    lines.append("\n## Lineage-Conditioned CN Volatility")
    lines.append(f"\nTarget lineage: {lineage}")
    lines.append(f"Estimated θ = {theta:.3f}")
    lines.append(f"LRT statistic = {lrt_stat:.2f}")
    lines.append(f"p-value = {p_value:.4f}")
    
    lines.append("\n### Statistical Significance")
    if p_value < 0.001:
        lines.append("  *** Highly significant (p < 0.001)")
    elif p_value < 0.01:
        lines.append("  ** Significant (p < 0.01)")
    elif p_value < 0.05:
        lines.append("  * Significant (p < 0.05)")
    else:
        lines.append("  Not significant (p ≥ 0.05)")
    
    lines.append("\n### Biological Interpretation")
    
    if p_value >= 0.05:
        lines.append(f"  No evidence for lineage-specific CN volatility in {lineage}.")
        lines.append("  Copy number evolves similarly across the tree.")
    elif theta > 0.5:
        lines.append(f"  **Elevated CN volatility in {lineage}**")
        lines.append(f"  Rate multiplier: {np.exp(theta):.2f}× (all CN transitions)")
        lines.append("\n  Descriptive statement:")
        lines.append(f"  '{lineage} exhibits elevated copy number turnover'")
        lines.append("\n  Possible biological context:")
        lines.append("  • Environmental adaptation (niche-specific genes)")
        lines.append("  • Host-pathogen arms race (if host-associated)")
        lines.append("  • Relaxed constraint in new environment")
        lines.append("  • Adaptive gene family expansion/contraction")
        lines.append("\n  Important: This is DESCRIPTIVE, not causal.")
        lines.append("  Cannot conclude that lineage/host *causes* volatility.")
    elif theta > 0.1:
        lines.append(f"  **Moderate CN volatility in {lineage}**")
        lines.append(f"  Rate multiplier: {np.exp(theta):.2f}×")
    elif theta < -0.5:
        lines.append(f"  **Suppressed CN volatility in {lineage}**")
        lines.append(f"  Rate multiplier: {np.exp(theta):.2f}× (all CN transitions)")
        lines.append("\n  Possible biological context:")
        lines.append("  • Stable environment (reduced selection pressure)")
        lines.append("  • Core functions (dosage-sensitive)")
        lines.append("  • Reduced genome plasticity")
    elif theta < -0.1:
        lines.append(f"  **Moderate CN suppression in {lineage}**")
        lines.append(f"  Rate multiplier: {np.exp(theta):.2f}×")
    else:
        lines.append("  **Weak lineage effect detected**")
        lines.append("  Effect is statistically significant but biologically small.")
    
    return "\n".join(lines)


def _generate_lineage_recommendation(
    theta: float,
    lineage: str,
    p_value: float,
) -> str:
    """Generate recommendation for next steps."""
    
    lines = []
    lines.append("### Recommendations")
    
    if p_value >= 0.05:
        lines.append("  1. No lineage-specific effect detected")
        lines.append("  2. CN evolution is similar across the tree")
        lines.append("  3. Consider testing other lineages/clades")
    elif theta > 0.3:
        lines.append(f"  1. {lineage} shows elevated CN volatility")
        lines.append("  2. Investigate which gene families drive the signal")
        lines.append("  3. Check for:")
        lines.append("     • Niche-specific genes")
        lines.append("     • Environmental adaptation signatures")
        lines.append("     • Host-pathogen interaction genes (if applicable)")
        lines.append("  4. Compare to other lineages for contrast")
        lines.append("  5. Consider Recipe 2 (Amplification Bias) within this lineage")
        lines.append("\n  Important: Frame results descriptively, not causally")
    elif theta < -0.3:
        lines.append(f"  1. {lineage} shows suppressed CN volatility")
        lines.append("  2. May indicate stable environment or core functions")
        lines.append("  3. Check for enrichment of essential genes")
    else:
        lines.append("  1. Weak effect - interpret cautiously")
        lines.append("  2. May reflect subtle differences in selection regime")
        lines.append("  3. Consider testing with more taxa or longer branches")
    
    return "\n".join(lines)
