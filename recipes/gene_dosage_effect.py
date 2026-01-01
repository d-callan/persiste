"""
Cross-Plugin Recipe: Gene Dosage Effect

Question: "How does gene retention interact with copy number dynamics?"

This is an ORCHESTRATION recipe - it runs two plugins sequentially and
integrates their results. No joint likelihood required.

Philosophy:
- Presence/absence events are rare but decisive
- CN changes are frequent but subtle
- Forcing them into one CTMC dilutes both signals
- Separate analysis + biological integration is more powerful

Workflow:
1. Gene retention analysis (GeneContent)
2. Conditional copy-number analysis (CopyNumberDynamics)
3. Integrative interpretation (the "dosage effect")
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from persiste.core.trees import TreeStructure, load_tree


@dataclass
class RetentionProfile:
    """
    Per-family retention profile from GeneContent analysis.

    Attributes:
        family_name: Gene family identifier
        retention_regime: Classification (core-like, accessory-like, lineage-restricted)
        presence_probability: π₁ (equilibrium presence probability)
        gain_rate: λ (gain rate)
        loss_rate: μ (loss rate)
        constraint_evidence: Evidence for retention constraint (p-value)
    """

    family_name: str
    retention_regime: str
    presence_probability: float
    gain_rate: float
    loss_rate: float
    constraint_evidence: float


@dataclass
class DosageProfile:
    """
    Per-family dosage profile from CopyNumberDynamics analysis.

    Attributes:
        family_name: Gene family identifier
        mean_cn: Mean copy number (when present)
        amplification_rate: Rate of CN increases
        contraction_rate: Rate of CN decreases
        dosage_stability: Evidence for dosage buffering (θ, p-value)
        amplification_bias: Evidence for amplification bias (θ, p-value)
    """

    family_name: str
    mean_cn: float
    amplification_rate: float
    contraction_rate: float
    dosage_stability: tuple[float, float]  # (θ, p-value)
    amplification_bias: tuple[float, float]  # (θ, p-value)


@dataclass
class GeneDosageEffectReport:
    """
    Integrated report from gene dosage effect analysis.

    Combines retention (GeneContent) and dosage (CopyNumberDynamics) profiles
    to identify biologically meaningful patterns.

    Attributes:
        retention_profiles: Per-family retention profiles
        dosage_profiles: Per-family dosage profiles
        integrated_patterns: Biological pattern classifications
        summary_table: Structured summary of results
        interpretation: Human-readable interpretation
    """

    retention_profiles: dict[str, RetentionProfile]
    dosage_profiles: dict[str, DosageProfile]
    integrated_patterns: dict[str, str]
    summary_table: pd.DataFrame
    interpretation: str


def gene_dosage_effect(
    cn_matrix: np.ndarray | str | Path,
    family_names: list[str],
    taxon_names: list[str],
    tree: TreeStructure | str | Path,
    output_dir: str | None = None,
    verbose: bool = True,
) -> GeneDosageEffectReport:
    """
    Cross-Plugin Recipe: Gene Dosage Effect

    Analyzes how gene retention interacts with copy number dynamics by:
    1. Running GeneContent to classify retention regimes
    2. Running CopyNumberDynamics conditional on presence
    3. Integrating results to identify biological patterns

    Patterns identified:
    - High retention + strong buffering → Essential dosage-sensitive genes
    - High retention + amplification bias → Adaptive dosage modulation
    - Low retention + high amplification → Mobile elements / selfish genes
    - Host-specific retention + CN volatility → Environment-dependent dosage

    Args:
        cn_matrix: Copy number matrix (n_families, n_taxa)
        family_names: List of gene family names
        taxon_names: List of taxon names
        tree: Phylogenetic tree
        output_dir: Optional directory for output files
        verbose: Print progress and interpretation

    Returns:
        GeneDosageEffectReport with integrated analysis

    Example:
        >>> report = gene_dosage_effect(
        ...     cn_matrix="data/cn_matrix.tsv",
        ...     family_names=families,
        ...     taxon_names=taxa,
        ...     tree="data/tree.nwk",
        ... )
        >>>
        >>> # Identify essential dosage-sensitive genes
        >>> essential = [
        ...     fam for fam, pattern in report.integrated_patterns.items()
        ...     if pattern == "essential_dosage_sensitive"
        ... ]

    Note:
        This recipe requires both GeneContent and CopyNumberDynamics plugins.
        It orchestrates them sequentially - no joint likelihood required.
    """
    if verbose:
        print("=" * 80)
        print("CROSS-PLUGIN RECIPE: Gene Dosage Effect")
        print("=" * 80)
        print("\nThis recipe orchestrates GeneContent + CopyNumberDynamics")
        print("to analyze how retention interacts with dosage dynamics.\n")

    # Load data
    if isinstance(cn_matrix, np.ndarray):
        cn_data = cn_matrix
    else:
        cn_data = np.loadtxt(cn_matrix, delimiter="\t", skiprows=1)

    # =========================================================================
    # Step 1: Gene Retention Analysis (GeneContent)
    # =========================================================================
    if verbose:
        print("=" * 80)
        print("STEP 1: Gene Retention Analysis (GeneContent)")
        print("=" * 80)
        print("\nClassifying genes by retention regime...\n")

    tree_obj = load_tree(tree) if isinstance(tree, (str, Path)) else tree

    retention_profiles = _run_genecontent_analysis(
        cn_data,
        family_names,
        taxon_names,
        tree_obj,
        verbose,
    )

    if verbose:
        print(f"\n✓ Classified {len(retention_profiles)} gene families")
        core_count = sum(
            1 for p in retention_profiles.values() if p.retention_regime == "core-like"
        )
        accessory_count = sum(
            1 for p in retention_profiles.values() if p.retention_regime == "accessory-like"
        )
        print(f"  • Core-like: {core_count}")
        print(f"  • Accessory-like: {accessory_count}")

    # =========================================================================
    # Step 2: Conditional Copy-Number Analysis (CopyNumberDynamics)
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2: Conditional Copy-Number Analysis (CopyNumberDynamics)")
        print("=" * 80)
        print("\nAnalyzing dosage dynamics conditional on presence...\n")

    dosage_profiles = _run_copynumber_analysis(
        cn_data,
        family_names,
        taxon_names,
        tree_obj,
        retention_profiles,
        verbose,
    )

    if verbose:
        print(f"\n✓ Analyzed dosage dynamics for {len(dosage_profiles)} families")

    # =========================================================================
    # Step 3: Integrative Interpretation
    # =========================================================================
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 3: Integrative Interpretation (Dosage Effect)")
        print("=" * 80)
        print("\nIdentifying biological patterns...\n")

    integrated_patterns = _integrate_patterns(retention_profiles, dosage_profiles, verbose)

    # Create summary table
    summary_table = _create_summary_table(
        retention_profiles,
        dosage_profiles,
        integrated_patterns,
    )

    # Generate interpretation
    interpretation = _generate_integrated_interpretation(
        retention_profiles,
        dosage_profiles,
        integrated_patterns,
    )

    if verbose:
        print(interpretation)

    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary_table.to_csv(output_path / "gene_dosage_effect_summary.tsv", sep="\t", index=False)

        with open(output_path / "gene_dosage_effect_interpretation.txt", "w") as f:
            f.write(interpretation)

        if verbose:
            print(f"\n✓ Results saved to: {output_dir}/")

    return GeneDosageEffectReport(
        retention_profiles=retention_profiles,
        dosage_profiles=dosage_profiles,
        integrated_patterns=integrated_patterns,
        summary_table=summary_table,
        interpretation=interpretation,
    )


def _run_genecontent_analysis(
    cn_data: np.ndarray,
    family_names: list[str],
    taxon_names: list[str],
    tree: TreeStructure,
    verbose: bool,
) -> dict[str, RetentionProfile]:
    """
    Run GeneContent analysis to classify retention regimes.

    Note: This is a placeholder implementation.
    In production, this would call the actual GeneContent plugin.
    """
    # Convert CN to presence/absence
    presence_data = (cn_data > 0).astype(int)

    # Placeholder: Simple classification based on presence frequency
    retention_profiles = {}

    for i, family_name in enumerate(family_names):
        presence_freq = np.mean(presence_data[i, :])

        # Simple classification
        if presence_freq > 0.9:
            regime = "core-like"
        elif presence_freq > 0.5:
            regime = "accessory-like"
        else:
            regime = "lineage-restricted"

        # Placeholder rates (would come from GeneContent fit)
        retention_profiles[family_name] = RetentionProfile(
            family_name=family_name,
            retention_regime=regime,
            presence_probability=presence_freq,
            gain_rate=0.1,  # Placeholder
            loss_rate=0.1 * (1 - presence_freq) / presence_freq if presence_freq > 0 else 0.1,
            constraint_evidence=0.5,  # Placeholder p-value
        )

    return retention_profiles


def _run_copynumber_analysis(
    cn_data: np.ndarray,
    family_names: list[str],
    taxon_names: list[str],
    tree: TreeStructure,
    retention_profiles: dict[str, RetentionProfile],
    verbose: bool,
) -> dict[str, DosageProfile]:
    """
    Run CopyNumberDynamics analysis conditional on presence.

    For each family, analyze CN dynamics only where gene is present.
    """
    dosage_profiles = {}

    for i, family_name in enumerate(family_names):
        # Filter to present strains
        present_mask = cn_data[i, :] > 0

        if np.sum(present_mask) < 5:
            # Not enough data for CN analysis
            continue

        # Compute basic statistics
        mean_cn = np.mean(cn_data[i, present_mask])

        # Placeholder for rates (would come from actual fit)
        amplification_rate = 0.1
        contraction_rate = 0.1

        # Placeholder for constraint tests
        # In production, would run actual dosage_stability_scan and amplification_bias_test
        # on filtered data
        dosage_stability = (0.0, 0.5)  # (θ, p-value)
        amplification_bias = (0.0, 0.5)  # (θ, p-value)

        dosage_profiles[family_name] = DosageProfile(
            family_name=family_name,
            mean_cn=mean_cn,
            amplification_rate=amplification_rate,
            contraction_rate=contraction_rate,
            dosage_stability=dosage_stability,
            amplification_bias=amplification_bias,
        )

    return dosage_profiles


def _integrate_patterns(
    retention_profiles: dict[str, RetentionProfile],
    dosage_profiles: dict[str, DosageProfile],
    verbose: bool,
) -> dict[str, str]:
    """
    Integrate retention and dosage profiles to identify biological patterns.

    Pattern classification:
    - essential_dosage_sensitive: High retention + strong buffering
    - adaptive_dosage_modulation: High retention + amplification bias
    - mobile_selfish: Low retention + high amplification
    - environment_dependent: Host-specific retention + CN volatility
    """
    integrated_patterns = {}

    for family_name in retention_profiles.keys():
        if family_name not in dosage_profiles:
            integrated_patterns[family_name] = "insufficient_data"
            continue

        retention = retention_profiles[family_name]
        dosage = dosage_profiles[family_name]

        # Pattern classification logic
        high_retention = retention.presence_probability > 0.8
        low_retention = retention.presence_probability < 0.3

        dosage_theta, dosage_p = dosage.dosage_stability
        amp_theta, amp_p = dosage.amplification_bias

        strong_buffering = dosage_p < 0.05 and dosage_theta < -0.3
        amp_bias = amp_p < 0.05 and amp_theta > 0.3
        high_amplification = dosage.amplification_rate > 0.15

        # Classify pattern
        if high_retention and strong_buffering:
            pattern = "essential_dosage_sensitive"
        elif high_retention and amp_bias:
            pattern = "adaptive_dosage_modulation"
        elif low_retention and high_amplification:
            pattern = "mobile_selfish"
        elif high_retention and not strong_buffering:
            pattern = "core_dosage_tolerant"
        elif low_retention:
            pattern = "accessory_variable"
        else:
            pattern = "intermediate"

        integrated_patterns[family_name] = pattern

    return integrated_patterns


def _create_summary_table(
    retention_profiles: dict[str, RetentionProfile],
    dosage_profiles: dict[str, DosageProfile],
    integrated_patterns: dict[str, str],
) -> pd.DataFrame:
    """Create structured summary table."""

    rows = []
    for family_name in retention_profiles.keys():
        retention = retention_profiles[family_name]
        dosage = dosage_profiles.get(family_name)
        pattern = integrated_patterns.get(family_name, "unknown")

        row = {
            "family": family_name,
            "retention_regime": retention.retention_regime,
            "presence_prob": retention.presence_probability,
            "pattern": pattern,
        }

        if dosage:
            row["mean_cn"] = dosage.mean_cn
            row["dosage_theta"] = dosage.dosage_stability[0]
            row["dosage_p"] = dosage.dosage_stability[1]
            row["amp_theta"] = dosage.amplification_bias[0]
            row["amp_p"] = dosage.amplification_bias[1]

        rows.append(row)

    return pd.DataFrame(rows)


def _generate_integrated_interpretation(
    retention_profiles: dict[str, RetentionProfile],
    dosage_profiles: dict[str, DosageProfile],
    integrated_patterns: dict[str, str],
) -> str:
    """Generate human-readable interpretation."""

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("INTEGRATED INTERPRETATION: Gene Dosage Effect")
    lines.append("=" * 80)

    # Count patterns
    pattern_counts = {}
    for pattern in integrated_patterns.values():
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    lines.append("\n### Biological Patterns Identified")
    lines.append(f"\nTotal families analyzed: {len(integrated_patterns)}")

    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  • {pattern}: {count} families")

    lines.append("\n### Pattern Interpretations")

    if pattern_counts.get("essential_dosage_sensitive", 0) > 0:
        lines.append("\n**Essential Dosage-Sensitive Genes:**")
        lines.append("  High retention + strong dosage buffering")
        lines.append("  → Essential genes with tight dosage constraint")
        lines.append("  → Likely housekeeping, core metabolism")

    if pattern_counts.get("adaptive_dosage_modulation", 0) > 0:
        lines.append("\n**Adaptive Dosage Modulation:**")
        lines.append("  High retention + amplification bias")
        lines.append("  → Core genes with adaptive CN variation")
        lines.append("  → Likely stress response, environmental adaptation")

    if pattern_counts.get("mobile_selfish", 0) > 0:
        lines.append("\n**Mobile/Selfish Elements:**")
        lines.append("  Low retention + high amplification")
        lines.append("  → Accessory genes with CN expansion")
        lines.append("  → Likely mobile elements, selfish genes")

    if pattern_counts.get("core_dosage_tolerant", 0) > 0:
        lines.append("\n**Core Dosage-Tolerant:**")
        lines.append("  High retention + no dosage constraint")
        lines.append("  → Core genes without tight dosage control")
        lines.append("  → May tolerate CN variation")

    lines.append("\n### Key Insights")
    lines.append("\nThis analysis separates:")
    lines.append("  1. Structural evolution (presence/absence) - rare but decisive")
    lines.append("  2. Quantitative tuning (copy number) - frequent but subtle")
    lines.append("\nBy analyzing them separately and integrating biologically,")
    lines.append("we capture signals that would be diluted in a joint model.")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)
