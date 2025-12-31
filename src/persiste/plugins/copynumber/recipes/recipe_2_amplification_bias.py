"""
Recipe 2: Amplification Bias Test

Question: "Are increases favored over decreases beyond baseline?"

Tests asymmetry between amplification and contraction.
This is dosage-specific - NOT redundant with gain/loss.
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
class AmplificationBiasReport:
    """
    Report from amplification bias analysis.
    
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


def amplification_bias_test(
    cn_matrix: Union[np.ndarray, str, Path],
    family_names: Optional[List[str]] = None,
    taxon_names: Optional[List[str]] = None,
    tree: Optional[Union[Tree, str, Path]] = None,
    baseline_type: str = 'hierarchical',
    theta_grid: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> AmplificationBiasReport:
    """
    Recipe 2: Amplification Bias Test
    
    Tests whether copy number increases (1→2, 2→3) are favored over
    decreases (2→1, 3→2) beyond baseline expectations.
    
    Constraint: AmplificationBiasConstraint (bidirectional)
    - θ < 0 → contraction favored (amplification suppressed)
    - θ = 0 → neutral (baseline rates)
    - θ > 0 → amplification favored (contraction suppressed)
    
    Note: Does NOT affect gene birth (0→1) - that's biologically distinct.
    
    Interpretation:
    - θ > 0 → Drug resistance, stress response, adaptive CNV
    - θ < 0 → Dosage constraint on multi-copy genes
    
    Args:
        cn_matrix: Copy number matrix (n_families, n_taxa) or path to file
        family_names: List of gene family names
        taxon_names: List of taxon names
        tree: Phylogenetic tree or path to tree file
        baseline_type: 'hierarchical' (recommended) or 'global'
        theta_grid: Grid of theta values to test (default: -2 to 2)
        verbose: Print interpretation
    
    Returns:
        AmplificationBiasReport with test results and interpretation
    
    Example:
        >>> report = amplification_bias_test(
        ...     cn_matrix="data/cn_matrix.tsv",
        ...     tree="data/tree.nwk",
        ... )
        >>> if report.p_value < 0.05 and report.theta > 0:
        ...     print("Amplification bias detected - check for drug resistance")
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
            constraint_type='amplification_bias',
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
        constraint_type='amplification_bias',
        theta=best_theta,
    )
    
    # Likelihood ratio test
    lrt_result = likelihood_ratio_test(alt_result, null_result)
    
    # Estimate confidence interval
    theta_se = 0.3  # Placeholder
    theta_ci = (best_theta - 1.96 * theta_se, best_theta + 1.96 * theta_se)
    
    # Generate interpretation
    interpretation = _generate_amplification_interpretation(
        best_theta, lrt_result['p_value'], lrt_result['lrt_statistic']
    )
    
    # Generate recommendation
    recommendation = _generate_amplification_recommendation(
        best_theta, lrt_result['p_value']
    )
    
    if verbose:
        print("=" * 70)
        print("Recipe 2: Amplification Bias Test")
        print("=" * 70)
        print(interpretation)
        print("\n" + recommendation)
        print("=" * 70)
    
    return AmplificationBiasReport(
        theta=best_theta,
        theta_ci=theta_ci,
        log_likelihood_null=null_result.log_likelihood,
        log_likelihood_alt=alt_result.log_likelihood,
        lrt_statistic=lrt_result['lrt_statistic'],
        p_value=lrt_result['p_value'],
        interpretation=interpretation,
        recommendation=recommendation,
    )


def _generate_amplification_interpretation(
    theta: float,
    p_value: float,
    lrt_stat: float,
) -> str:
    """Generate human-readable interpretation."""
    
    lines = []
    lines.append("\n## Amplification Bias Analysis")
    lines.append(f"\nEstimated θ = {theta:.3f}")
    lines.append(f"LRT statistic = {lrt_stat:.2f}")
    lines.append(f"p-value = {p_value:.4f}")
    
    lines.append("\n### Constraint Effect")
    lines.append(f"  Amplification (1→2, 2→3): {np.exp(theta):.2f}× baseline")
    lines.append(f"  Contraction (2→1, 3→2):   {np.exp(-theta):.2f}× baseline")
    lines.append("  Gene birth (0→1):          unchanged (biologically distinct)")
    
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
        lines.append("  No evidence for amplification bias.")
        lines.append("  Amplification and contraction are balanced.")
    elif theta > 0.5:
        lines.append("  **Strong amplification bias detected**")
        lines.append("  Copy number increases are favored over decreases.")
        lines.append("\n  Biological context:")
        lines.append("  • Drug resistance genes (adaptive CNV)")
        lines.append("  • Efflux pumps (antibiotic resistance)")
        lines.append("  • Stress response genes")
        lines.append("  • Virulence factors")
        lines.append("\n  Mechanism:")
        lines.append("  • Positive selection for increased dosage")
        lines.append("  • Relaxed constraint on multi-copy state")
    elif theta > 0.1:
        lines.append("  **Moderate amplification bias detected**")
        lines.append("  Copy number increases are moderately favored.")
    elif theta < -0.5:
        lines.append("  **Contraction bias detected**")
        lines.append("  Copy number decreases are favored over increases.")
        lines.append("\n  Biological context:")
        lines.append("  • Dosage constraint on multi-copy genes")
        lines.append("  • Cost of maintaining high copy number")
        lines.append("  • Selection against gene dosage imbalance")
    elif theta < -0.1:
        lines.append("  **Moderate contraction bias detected**")
        lines.append("  Copy number decreases are moderately favored.")
    else:
        lines.append("  **Weak asymmetry detected**")
        lines.append("  Effect is statistically significant but biologically small.")
    
    return "\n".join(lines)


def _generate_amplification_recommendation(theta: float, p_value: float) -> str:
    """Generate recommendation for next steps."""
    
    lines = []
    lines.append("### Recommendations")
    
    if p_value >= 0.05:
        lines.append("  1. No amplification bias detected")
        lines.append("  2. Amplification and contraction are balanced")
        lines.append("  3. Consider Recipe 1 (Dosage Stability) for global effects")
    elif theta > 0.3:
        lines.append("  1. Strong amplification bias suggests adaptive CNV")
        lines.append("  2. Investigate which families drive the signal")
        lines.append("  3. Check for:")
        lines.append("     • Drug resistance genes")
        lines.append("     • Stress response pathways")
        lines.append("     • Virulence factors")
        lines.append("  4. Consider environmental/clinical context")
        lines.append("  5. Look for recent selective sweeps")
    elif theta < -0.3:
        lines.append("  1. Contraction bias suggests dosage constraint")
        lines.append("  2. Multi-copy genes may be costly to maintain")
        lines.append("  3. Check for metabolic burden or dosage imbalance")
    else:
        lines.append("  1. Weak effect - interpret cautiously")
        lines.append("  2. May reflect heterogeneity across gene families")
        lines.append("  3. Consider family-specific analyses")
    
    return "\n".join(lines)
