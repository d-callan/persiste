"""
Recipe 0: Null CN Dynamics (Descriptive Baseline)

Question: "What does copy number evolution look like under no hypothesis?"

This is the descriptive anchor - always run this first to understand baseline dynamics.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from persiste.core.trees import TreeStructure
from persiste.plugins.copynumber.cn_interface import fit_null_model


@dataclass
class NullCNReport:
    """
    Report from null CN dynamics analysis.

    Attributes:
        log_likelihood: Model log-likelihood
        stationary_distribution: π(state) for each CN state
        expected_cn: Expected copy number at equilibrium
        transition_fluxes: Expected flux between states
        family_summaries: Per-family statistics
        interpretation: Human-readable interpretation
    """

    log_likelihood: float
    stationary_distribution: np.ndarray
    expected_cn: float
    transition_fluxes: dict[str, float]
    family_summaries: dict[str, np.ndarray]
    interpretation: str


def null_cn_dynamics(
    cn_matrix: np.ndarray | str | Path,
    family_names: list[str] | None = None,
    taxon_names: list[str] | None = None,
    tree: TreeStructure | str | Path | None = None,
    baseline_type: str = "hierarchical",
    verbose: bool = True,
) -> NullCNReport:
    """
    Recipe 0: Null CN Dynamics (Descriptive Baseline)

    Fits a hierarchical baseline model with no constraints to describe
    the baseline copy number evolutionary dynamics.

    This is your descriptive anchor - run this first to understand:
    - What is the equilibrium CN distribution?
    - What is the expected copy number?
    - What are the dominant transition types?

    Args:
        cn_matrix: Copy number matrix (n_families, n_taxa) or path to file
        family_names: List of gene family names
        taxon_names: List of taxon names
        tree: Phylogenetic tree or path to tree file
        baseline_type: 'hierarchical' (recommended) or 'global'
        verbose: Print interpretation

    Returns:
        NullCNReport with baseline dynamics summary

    Example:
        >>> report = null_cn_dynamics(
        ...     cn_matrix="data/cn_matrix.tsv",
        ...     tree="data/tree.nwk",
        ...     baseline_type='hierarchical'
        ... )
        >>> print(report.interpretation)
    """
    # Fit null model
    result = fit_null_model(
        cn_matrix=cn_matrix,
        family_names=family_names,
        taxon_names=taxon_names,
        tree=tree,
        baseline_type=baseline_type,
    )

    # Extract baseline rate matrix (use global mean parameters for summary)
    Q = _build_rate_matrix_from_params(baseline_type, result.baseline_params)

    # Compute stationary distribution
    # Solve πQ = 0 with Σπ = 1
    eigenvalues, eigenvectors = np.linalg.eig(Q.T)
    idx = np.argmin(np.abs(eigenvalues))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)

    # Expected copy number at equilibrium
    # State 0=Absent, 1=Single, 2=Low-Multi, 3=High-Multi
    # Use midpoints: 0, 1, 4, 10
    state_midpoints = np.array([0, 1, 4, 10])
    expected_cn = np.sum(pi * state_midpoints)

    # Transition fluxes (expected transitions per unit time)
    transition_fluxes = {}
    transition_names = {
        (0, 1): "gain",
        (1, 0): "loss",
        (1, 2): "amplify_low",
        (2, 1): "contract_low",
        (2, 3): "amplify_high",
        (3, 2): "contract_high",
    }

    for (i, j), name in transition_names.items():
        flux = pi[i] * Q[i, j]
        transition_fluxes[name] = flux

    # Per-family summaries
    if isinstance(cn_matrix, np.ndarray):
        cn_data = cn_matrix
    else:
        # Load from file
        cn_data = np.loadtxt(cn_matrix, delimiter="\t", skiprows=1)

    family_summaries = {
        "mean_cn": np.mean(cn_data, axis=1),
        "std_cn": np.std(cn_data, axis=1),
        "presence_fraction": np.mean(cn_data > 0, axis=1),
    }

    # Generate interpretation
    interpretation = _generate_null_interpretation(
        pi, expected_cn, transition_fluxes, baseline_type
    )

    if verbose:
        print("=" * 70)
        print("Recipe 0: Null CN Dynamics (Descriptive Baseline)")
        print("=" * 70)
        print(interpretation)
        print("=" * 70)

    return NullCNReport(
        log_likelihood=result.log_likelihood,
        stationary_distribution=pi,
        expected_cn=expected_cn,
        transition_fluxes=transition_fluxes,
        family_summaries=family_summaries,
        interpretation=interpretation,
    )


def _generate_null_interpretation(
    pi: np.ndarray,
    expected_cn: float,
    fluxes: dict[str, float],
    baseline_type: str,
) -> str:
    """Generate human-readable interpretation of null dynamics."""

    lines = []
    lines.append("\n## Baseline Copy Number Dynamics")
    lines.append(f"\nModel: {baseline_type} baseline (no constraints)")
    lines.append("\n### Equilibrium Distribution")
    lines.append(f"  Absent (state 0):      {pi[0]:.1%}")
    lines.append(f"  Single-copy (state 1): {pi[1]:.1%}")
    lines.append(f"  Low-multi (state 2):   {pi[2]:.1%}")
    lines.append(f"  High-multi (state 3):  {pi[3]:.1%}")

    lines.append(f"\n### Expected Copy Number: {expected_cn:.2f}")
    lines.append("\n### Dominant Transitions (flux per unit time)")
    sorted_fluxes = sorted(fluxes.items(), key=lambda x: x[1], reverse=True)
    for name, flux in sorted_fluxes[:3]:
        lines.append(f"  {name:20s}: {flux:.4f}")

    lines.append("\n### Interpretation")

    # Presence/absence dominance
    if pi[0] > 0.5:
        lines.append("  • Most gene families are absent at equilibrium (accessory genome)")
    elif pi[1] > 0.5:
        lines.append("  • Most gene families are single-copy (core genome)")
    else:
        lines.append("  • Mixed equilibrium (diverse CN states)")

    # Multi-copy prevalence
    multi_fraction = pi[2] + pi[3]
    if multi_fraction > 0.3:
        lines.append(f"  • Substantial multi-copy fraction ({multi_fraction:.1%})")
    elif multi_fraction > 0.1:
        lines.append(f"  • Moderate multi-copy fraction ({multi_fraction:.1%})")
    else:
        lines.append(f"  • Low multi-copy fraction ({multi_fraction:.1%})")

    # Gain/loss balance
    gain_flux = fluxes["gain"]
    loss_flux = fluxes["loss"]
    if gain_flux > loss_flux * 1.5:
        lines.append("  • Gain-biased dynamics (genes accumulate)")
    elif loss_flux > gain_flux * 1.5:
        lines.append("  • Loss-biased dynamics (genes erode)")
    else:
        lines.append("  • Balanced gain/loss dynamics")

    # Amplification activity
    amplify_flux = fluxes["amplify_low"] + fluxes["amplify_high"]
    contract_flux = fluxes["contract_low"] + fluxes["contract_high"]
    if amplify_flux > 0.01:
        if amplify_flux > contract_flux * 1.5:
            lines.append("  • Active amplification (CN increases favored)")
        else:
            lines.append("  • Active CN turnover (amplification and contraction)")
    else:
        lines.append("  • Low amplification activity (mostly single-copy)")

    lines.append("\n### Next Steps")
    lines.append("  1. Check diagnostic plot: expected vs observed CN")
    lines.append("  2. Run Recipe 1 (Dosage Stability) to test for buffering/volatility")
    lines.append("  3. Run Recipe 2 (Amplification Bias) if multi-copy fraction > 10%")

    return "\n".join(lines)


def _build_rate_matrix_from_params(baseline_type: str, params: dict[str, float]) -> np.ndarray:
    """
    Reconstruct baseline rate matrix from stored parameters.

    Args:
        baseline_type: 'global' or 'hierarchical'
        params: Baseline parameter dictionary from CopyNumberResult
    """
    if params is None:
        raise ValueError(
            "baseline_params missing from CopyNumberResult; cannot summarize baseline dynamics"
        )

    if baseline_type == "global":
        gain = params["gain_rate"]
        loss = params["loss_rate"]
        amplify = params["amplify_rate"]
        contract = params["contract_rate"]
    else:
        gain = params["global_gain_rate"]
        loss = params["global_loss_rate"]
        amplify = params["global_amplify_rate"]
        contract = params["global_contract_rate"]

    rate_matrix = np.zeros((4, 4))
    transitions = {
        (0, 1): gain,
        (1, 0): loss,
        (1, 2): amplify,
        (2, 1): contract,
        (2, 3): amplify,
        (3, 2): contract,
    }
    for (i, j), rate in transitions.items():
        rate_matrix[i, j] = rate
    for i in range(4):
        rate_matrix[i, i] = -np.sum(rate_matrix[i, :])
    return rate_matrix
