"""
Diagnostic utilities for CopyNumberDynamics plugin.

Critical diagnostic: Expected vs Observed CN per family.
If these diverge wildly under null, something is wrong:
- Binning is incorrect
- Baseline is mis-specified
- Data quality is poor
"""

from typing import Optional, Union, List, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from persiste.core.trees import Tree
from persiste.plugins.copynumber.cn_interface import fit_null_model


def expected_vs_observed_cn(
    cn_matrix: Union[np.ndarray, str, Path],
    family_names: Optional[List[str]] = None,
    taxon_names: Optional[List[str]] = None,
    tree: Optional[Union[Tree, str, Path]] = None,
    baseline_type: str = 'hierarchical',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Diagnostic plot: Expected vs Observed CN per family.
    
    This is the CN equivalent of "tree-sequence mismatch check".
    
    If expected and observed CN diverge wildly under the null model:
    - Binning may be wrong (check state definitions)
    - Baseline may be mis-specified (try hierarchical)
    - Data quality may be poor (check for errors)
    
    Args:
        cn_matrix: Copy number matrix (n_families, n_taxa) or path to file
        family_names: List of gene family names
        taxon_names: List of taxon names
        tree: Phylogenetic tree or path to tree file
        baseline_type: 'hierarchical' (recommended) or 'global'
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> fig = expected_vs_observed_cn(
        ...     cn_matrix="data/cn_matrix.tsv",
        ...     tree="data/tree.nwk",
        ...     save_path="diagnostics/expected_vs_observed.png"
        ... )
    """
    # Load data
    if isinstance(cn_matrix, np.ndarray):
        cn_data = cn_matrix
    else:
        cn_data = np.loadtxt(cn_matrix, delimiter='\t', skiprows=1)
    
    # Fit null model
    result = fit_null_model(
        cn_matrix=cn_matrix,
        family_names=family_names,
        taxon_names=taxon_names,
        tree=tree,
        baseline_type=baseline_type,
    )
    
    # Compute observed mean CN per family
    observed_mean = np.mean(cn_data, axis=1)
    
    # Compute expected CN from stationary distribution
    baseline = result.baseline
    Q = baseline.build_rate_matrix()
    
    # Stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(Q.T)
    idx = np.argmin(np.abs(eigenvalues))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)
    
    # Expected CN (using state midpoints: 0, 1, 4, 10)
    state_midpoints = np.array([0, 1, 4, 10])
    expected_cn = np.sum(pi * state_midpoints)
    
    # For hierarchical baseline, compute per-family expectations
    if baseline_type == 'hierarchical':
        expected_mean = np.full(len(observed_mean), expected_cn)
        # TODO: Could compute family-specific expectations if rates are stored
    else:
        expected_mean = np.full(len(observed_mean), expected_cn)
    
    # Create diagnostic plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel 1: Scatter plot
    ax = axes[0]
    ax.scatter(expected_mean, observed_mean, alpha=0.5, s=20)
    
    # Add diagonal line (perfect agreement)
    min_val = min(expected_mean.min(), observed_mean.min())
    max_val = max(expected_mean.max(), observed_mean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect agreement')
    
    ax.set_xlabel('Expected CN (from null model)', fontsize=12)
    ax.set_ylabel('Observed mean CN', fontsize=12)
    ax.set_title('Expected vs Observed CN per Family', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Residuals
    ax = axes[1]
    residuals = observed_mean - expected_mean
    ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', lw=2, label='Zero residual')
    ax.set_xlabel('Residual (Observed - Expected)', fontsize=12)
    ax.set_ylabel('Number of families', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add summary statistics
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    fig.text(0.5, 0.02, 
             f'RMSE: {rmse:.3f} | MAE: {mae:.3f} | Baseline: {baseline_type}',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Diagnostic plot saved to: {save_path}")
    
    return fig


def interpret_diagnostic(
    cn_matrix: Union[np.ndarray, str, Path],
    family_names: Optional[List[str]] = None,
    taxon_names: Optional[List[str]] = None,
    tree: Optional[Union[Tree, str, Path]] = None,
    baseline_type: str = 'hierarchical',
) -> str:
    """
    Interpret the expected vs observed CN diagnostic.
    
    Provides guidance on whether the model fit is reasonable.
    
    Args:
        cn_matrix: Copy number matrix
        family_names: List of gene family names
        taxon_names: List of taxon names
        tree: Phylogenetic tree
        baseline_type: 'hierarchical' or 'global'
    
    Returns:
        Interpretation string
    """
    # Load data
    if isinstance(cn_matrix, np.ndarray):
        cn_data = cn_matrix
    else:
        cn_data = np.loadtxt(cn_matrix, delimiter='\t', skiprows=1)
    
    # Fit null model
    result = fit_null_model(
        cn_matrix=cn_matrix,
        family_names=family_names,
        taxon_names=taxon_names,
        tree=tree,
        baseline_type=baseline_type,
    )
    
    # Compute observed mean CN per family
    observed_mean = np.mean(cn_data, axis=1)
    
    # Compute expected CN
    baseline = result.baseline
    Q = baseline.build_rate_matrix()
    eigenvalues, eigenvectors = np.linalg.eig(Q.T)
    idx = np.argmin(np.abs(eigenvalues))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)
    state_midpoints = np.array([0, 1, 4, 10])
    expected_cn = np.sum(pi * state_midpoints)
    expected_mean = np.full(len(observed_mean), expected_cn)
    
    # Compute diagnostics
    residuals = observed_mean - expected_mean
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    max_residual = np.max(np.abs(residuals))
    
    # Generate interpretation
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("DIAGNOSTIC: Expected vs Observed CN")
    lines.append("=" * 70)
    
    lines.append(f"\nBaseline model: {baseline_type}")
    lines.append(f"Expected CN (equilibrium): {expected_cn:.2f}")
    lines.append(f"Observed mean CN: {np.mean(observed_mean):.2f}")
    lines.append(f"\nRMSE: {rmse:.3f}")
    lines.append(f"MAE: {mae:.3f}")
    lines.append(f"Max absolute residual: {max_residual:.3f}")
    
    lines.append("\n### Interpretation")
    
    if rmse < 0.5:
        lines.append("  ✓ Excellent fit - model captures CN distribution well")
    elif rmse < 1.0:
        lines.append("  ✓ Good fit - reasonable agreement between expected and observed")
    elif rmse < 2.0:
        lines.append("  ⚠ Moderate fit - some discrepancy present")
        lines.append("    Consider:")
        lines.append("    • Using hierarchical baseline (if using global)")
        lines.append("    • Checking for data quality issues")
        lines.append("    • Verifying state binning is appropriate")
    else:
        lines.append("  ✗ Poor fit - substantial discrepancy")
        lines.append("    Likely issues:")
        lines.append("    • Binning may be incorrect (check state definitions)")
        lines.append("    • Baseline model mis-specified (try hierarchical)")
        lines.append("    • Data quality problems (check for errors)")
        lines.append("    • Tree may not match data")
    
    if max_residual > 5:
        lines.append("\n  ⚠ Some families have very large residuals")
        lines.append("    • Check for outliers or data errors")
        lines.append("    • May indicate regime heterogeneity")
    
    lines.append("\n### Recommendations")
    
    if rmse < 1.0:
        lines.append("  • Model fit is acceptable - proceed with constraint tests")
        lines.append("  • Run Recipe 1 (Dosage Stability)")
        lines.append("  • Run Recipe 2 (Amplification Bias) if multi-copy genes present")
    else:
        lines.append("  • Investigate model fit before running constraint tests")
        if baseline_type == 'global':
            lines.append("  • Try hierarchical baseline for better fit")
        lines.append("  • Check data quality and binning")
        lines.append("  • Verify tree is appropriate for data")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)
