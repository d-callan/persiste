"""
Recipe 1: Dosage Stability Scan (Core Constraint Test)

Question: "Are copy number changes globally suppressed or enhanced?"

This is the CN analogue of "Are genes conserved or labile?"
"""

from typing import Dict, Optional, Union, List
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from persiste.core.trees import TreeStructure
from persiste.plugins.copynumber.cn_interface import (
    fit,
    fit_null_model,
    likelihood_ratio_test,
)


@dataclass
class DosageStabilityReport:
    """
    Report from dosage stability analysis.
    
    Attributes:
        theta: Estimated constraint parameter
        theta_ci: Confidence interval for theta
        log_likelihood_null: Null model log-likelihood
        log_likelihood_alt: Alternative model log-likelihood
        lrt_statistic: Likelihood ratio test statistic
        p_value: LRT p-value
        interpretation: Human-readable interpretation
        recommendation: What to do next
    """
    theta: float
    theta_ci: tuple
    log_likelihood_null: float
    log_likelihood_alt: float
    lrt_statistic: float
    p_value: float
    interpretation: str
    recommendation: str


def dosage_stability_scan(
    cn_matrix: Union[np.ndarray, str, Path],
    family_names: Optional[List[str]] = None,
    taxon_names: Optional[List[str]] = None,
    tree: Optional[Union[TreeStructure, str, Path]] = None,
    baseline_type: str = 'hierarchical',
    theta_grid: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> DosageStabilityReport:
    """
    Recipe 1: Dosage Stability Scan
    
    Tests whether copy number changes are globally suppressed (buffering)
    or enhanced (volatility) relative to baseline expectations.
    
    Constraint: DosageStabilityConstraint
    - θ < 0 → dosage buffering (suppressed CN changes)
    - θ = 0 → neutral (baseline rates)
    - θ > 0 → dosage volatility (enhanced CN changes)
    
    Interpretation:
    - "Strong dosage buffering" → Essential genes, housekeeping
    - "CN volatility" → Antigen families, stress response
    
    Args:
        cn_matrix: Copy number matrix (n_families, n_taxa) or path to file
        family_names: List of gene family names
        taxon_names: List of taxon names
        tree: Phylogenetic tree or path to tree file
        baseline_type: 'hierarchical' (recommended) or 'global'
        theta_grid: Grid of theta values to test (default: -2 to 2)
        verbose: Print interpretation
    
    Returns:
        DosageStabilityReport with test results and interpretation
    
    Example:
        >>> report = dosage_stability_scan(
        ...     cn_matrix="data/cn_matrix.tsv",
        ...     tree="data/tree.nwk",
        ... )
        >>> print(report.interpretation)
        >>> if report.p_value < 0.05:
        ...     print(f"Significant dosage effect: θ = {report.theta:.2f}")
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
            constraint_type='dosage_stability',
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
        constraint_type='dosage_stability',
        theta=best_theta,
    )
    
    # Likelihood ratio test
    lrt_result = likelihood_ratio_test(alt_result, null_result)
    
    # Estimate confidence interval (profile likelihood)
    # Simple approximation: θ ± 1.96 * SE
    # SE estimated from curvature at MLE
    theta_se = 0.3  # Placeholder - should compute from Hessian
    theta_ci = (best_theta - 1.96 * theta_se, best_theta + 1.96 * theta_se)
    
    # Generate interpretation
    interpretation = _generate_dosage_interpretation(
        best_theta, lrt_result['p_value'], lrt_result['lrt_statistic']
    )
    
    # Generate recommendation
    recommendation = _generate_dosage_recommendation(
        best_theta, lrt_result['p_value']
    )
    
    if verbose:
        print("=" * 70)
        print("Recipe 1: Dosage Stability Scan")
        print("=" * 70)
        print(interpretation)
        print("\n" + recommendation)
        print("=" * 70)
    
    return DosageStabilityReport(
        theta=best_theta,
        theta_ci=theta_ci,
        log_likelihood_null=null_result.log_likelihood,
        log_likelihood_alt=alt_result.log_likelihood,
        lrt_statistic=lrt_result['lrt_statistic'],
        p_value=lrt_result['p_value'],
        interpretation=interpretation,
        recommendation=recommendation,
    )


def _generate_dosage_interpretation(
    theta: float,
    p_value: float,
    lrt_stat: float,
) -> str:
    """Generate human-readable interpretation."""
    
    lines = []
    lines.append("\n## Dosage Stability Analysis")
    lines.append(f"\nEstimated θ = {theta:.3f}")
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
        lines.append("  No evidence for dosage-specific constraint.")
        lines.append("  Copy number evolves according to baseline dynamics.")
    elif theta < -0.5:
        lines.append("  **Strong dosage buffering detected**")
        lines.append("  Copy number changes are suppressed relative to baseline.")
        lines.append(f"  Rate multiplier: {np.exp(theta):.2f}× (all CN transitions)")
        lines.append("\n  Biological context:")
        lines.append("  • Essential genes (dosage-sensitive)")
        lines.append("  • Housekeeping genes (stable expression)")
        lines.append("  • Core metabolic pathways")
    elif theta < -0.1:
        lines.append("  **Moderate dosage buffering detected**")
        lines.append("  Copy number changes are moderately suppressed.")
        lines.append(f"  Rate multiplier: {np.exp(theta):.2f}×")
    elif theta > 0.5:
        lines.append("  **Copy number volatility detected**")
        lines.append("  Copy number changes are enhanced relative to baseline.")
        lines.append(f"  Rate multiplier: {np.exp(theta):.2f}× (all CN transitions)")
        lines.append("\n  Biological context:")
        lines.append("  • Antigen families (immune evasion)")
        lines.append("  • Stress response genes")
        lines.append("  • Mobile elements")
    elif theta > 0.1:
        lines.append("  **Moderate copy number volatility detected**")
        lines.append("  Copy number changes are moderately enhanced.")
        lines.append(f"  Rate multiplier: {np.exp(theta):.2f}×")
    else:
        lines.append("  **Weak dosage effect detected**")
        lines.append("  Effect is statistically significant but biologically small.")
    
    return "\n".join(lines)


def _generate_dosage_recommendation(theta: float, p_value: float) -> str:
    """Generate recommendation for next steps."""
    
    lines = []
    lines.append("### Recommendations")
    
    if p_value >= 0.05:
        lines.append("  1. No dosage-specific constraint detected")
        lines.append("  2. Consider Recipe 2 (Amplification Bias) for asymmetry tests")
        lines.append("  3. Check if specific gene families show dosage effects")
    elif theta < -0.3:
        lines.append("  1. Strong buffering suggests dosage-sensitive genes")
        lines.append("  2. Investigate which families drive the signal")
        lines.append("  3. Consider functional enrichment analysis")
        lines.append("  4. Check for essentiality / housekeeping annotations")
    elif theta > 0.3:
        lines.append("  1. Volatility suggests adaptive CN variation")
        lines.append("  2. Investigate which families drive the signal")
        lines.append("  3. Consider Recipe 2 (Amplification Bias) for directionality")
        lines.append("  4. Check for antigen / stress response annotations")
    else:
        lines.append("  1. Weak effect - interpret cautiously")
        lines.append("  2. May reflect heterogeneity across gene families")
        lines.append("  3. Consider family-specific analyses")
    
    return "\n".join(lines)
