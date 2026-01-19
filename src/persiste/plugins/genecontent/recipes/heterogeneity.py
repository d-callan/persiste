"""Heterogeneity-focused recipes for GeneContent analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..pam_interface import PAMAnalysisResult, fit
from ..strain_diagnostics import compute_strain_cloud_content

__all__ = [
    "HeterogeneityScanResult",
    "StratifiedRegimeResult",
    "run_heterogeneity_diagnostic",
    "strain_heterogeneity_scan",
    "stratified_regime_modeling",
]


@dataclass
class HeterogeneityScanResult:
    """Results from the strain heterogeneity scan."""

    full_result: PAMAnalysisResult
    remove_top10_result: PAMAnalysisResult
    remove_bottom10_result: PAMAnalysisResult
    parameter_shifts: dict[str, float]
    interpretation: str

    def print_summary(self) -> None:
        """Print scan summary."""

        print("=" * 70)
        print("STRAIN HETEROGENEITY SCAN RESULTS")
        print("=" * 70)

        print(f"\n{'Dataset':<30} {'Strains':<10} {'λ':<10} {'μ':<10} {'π₁':<10} {'λ/μ':<10}")
        print("-" * 80)

        r = self.full_result
        print(
            f"{'Full dataset':<30} {r.data.n_taxa:<10} {r.gain_rate:<10.4f} {r.loss_rate:<10.4f} "
            f"{r.equilibrium_frequency:<10.4f} {r.gain_rate / r.loss_rate:<10.3f}"
        )

        r = self.remove_top10_result
        print(
            f"{'Remove top 10% cloud':<30} {r.data.n_taxa:<10} {r.gain_rate:<10.4f} {r.loss_rate:<10.4f} "
            f"{r.equilibrium_frequency:<10.4f} {r.gain_rate / r.loss_rate:<10.3f}"
        )

        r = self.remove_bottom10_result
        print(
            f"{'Remove bottom 10% cloud':<30} {r.data.n_taxa:<10} {r.gain_rate:<10.4f} {r.loss_rate:<10.4f} "
            f"{r.equilibrium_frequency:<10.4f} {r.gain_rate / r.loss_rate:<10.3f}"
        )

        print("\nParameter shifts:")
        for param, shift in self.parameter_shifts.items():
            print(f"  {param}: {shift:+.1f}%")

        print("\nInterpretation:")
        print(f"  {self.interpretation}")
        print("\n" + "=" * 70)


@dataclass
class StratifiedRegimeResult:
    """Results from stratified regime modeling."""

    high_accessory_result: PAMAnalysisResult
    low_accessory_result: PAMAnalysisResult
    n_high: int
    n_low: int
    threshold: float
    regime_comparison: dict[str, tuple[float, float]]

    def print_summary(self) -> None:
        """Print stratified modeling summary."""

        print("=" * 70)
        print("STRATIFIED REGIME MODELING RESULTS")
        print("=" * 70)

        print("\nStratification:")
        print(f"  High-accessory strains: {self.n_high:,} (cloud genes > {self.threshold:.0f})")
        print(f"  Low-accessory strains:  {self.n_low:,} (cloud genes ≤ {self.threshold:.0f})")

        print(f"\n{'Parameter':<20} {'High-accessory':<20} {'Low-accessory':<20} {'Difference':<15}")
        print("-" * 80)

        for param, (high_val, low_val) in self.regime_comparison.items():
            diff = high_val - low_val
            print(f"{param:<20} {high_val:<20.4f} {low_val:<20.4f} {diff:+.4f}")

        print("\nInterpretation:")
        high_regime = (
            "gain" if self.high_accessory_result.gain_rate > self.high_accessory_result.loss_rate else "loss"
        )
        low_regime = (
            "gain" if self.low_accessory_result.gain_rate > self.low_accessory_result.loss_rate else "loss"
        )

        if high_regime != low_regime:
            print("  ✓ DISTINCT REGIMES DETECTED")
            print(f"    High-accessory: {high_regime.upper()}-dominated")
            print(f"    Low-accessory:  {low_regime.upper()}-dominated")
        else:
            print(f"  • Both groups show {high_regime.upper()}-dominated dynamics")
            print("    But with different rates")

        print("\n" + "=" * 70)


def run_heterogeneity_diagnostic(
    pam: np.ndarray,
    taxon_names: list[str],
    gene_names: list[str],
    tree_method: str = "jaccard_upgma",
    verbose: bool = True,
) -> HeterogeneityScanResult:
    """User-facing entry point for the strain heterogeneity scan."""

    return strain_heterogeneity_scan(
        pam=pam,
        taxon_names=taxon_names,
        gene_names=gene_names,
        tree_method=tree_method,
        verbose=verbose,
    )


def strain_heterogeneity_scan(
    pam: np.ndarray,
    taxon_names: list[str],
    gene_names: list[str],
    tree_method: str = "jaccard_upgma",
    verbose: bool = True,
) -> HeterogeneityScanResult:
    """Diagnostic that tests parameter stability across strain subsets."""

    if verbose:
        print("=" * 70)
        print("STRAIN HETEROGENEITY SCAN")
        print("=" * 70)
        print("\nTesting parameter stability across strain subsets...")

    cloud_per_strain = compute_strain_cloud_content(pam)

    if verbose:
        print("\n1. Fitting full dataset...")
    result_full = fit(
        pam=pam,
        taxon_names=taxon_names,
        gene_names=gene_names,
        tree_method=tree_method,
        verbose=False,
    )

    q90 = np.percentile(cloud_per_strain, 90)
    keep_top = cloud_per_strain <= q90

    if verbose:
        print(f"2. Removing top 10% high-cloud strains ({(~keep_top).sum()} strains)...")

    result_remove_top = fit(
        pam=pam[keep_top, :],
        taxon_names=[taxon_names[i] for i in np.where(keep_top)[0]],
        gene_names=gene_names,
        tree_method=tree_method,
        verbose=False,
    )

    q10 = np.percentile(cloud_per_strain, 10)
    keep_bottom = cloud_per_strain >= q10

    if verbose:
        print(f"3. Removing bottom 10% low-cloud strains ({(~keep_bottom).sum()} strains)...")

    result_remove_bottom = fit(
        pam=pam[keep_bottom, :],
        taxon_names=[taxon_names[i] for i in np.where(keep_bottom)[0]],
        gene_names=gene_names,
        tree_method=tree_method,
        verbose=False,
    )

    lambda_shift_top = 100 * (result_remove_top.gain_rate - result_full.gain_rate) / result_full.gain_rate
    mu_shift_top = 100 * (result_remove_top.loss_rate - result_full.loss_rate) / result_full.loss_rate
    pi1_shift_top = (
        100
        * (result_remove_top.equilibrium_frequency - result_full.equilibrium_frequency)
        / result_full.equilibrium_frequency
    )

    lambda_shift_bottom = (
        100 * (result_remove_bottom.gain_rate - result_full.gain_rate) / result_full.gain_rate
    )
    mu_shift_bottom = 100 * (result_remove_bottom.loss_rate - result_full.loss_rate) / result_full.loss_rate
    pi1_shift_bottom = (
        100
        * (result_remove_bottom.equilibrium_frequency - result_full.equilibrium_frequency)
        / result_full.equilibrium_frequency
    )

    parameter_shifts = {
        "λ (remove top 10%)": lambda_shift_top,
        "μ (remove top 10%)": mu_shift_top,
        "π₁ (remove top 10%)": pi1_shift_top,
        "λ (remove bottom 10%)": lambda_shift_bottom,
        "μ (remove bottom 10%)": mu_shift_bottom,
        "π₁ (remove bottom 10%)": pi1_shift_bottom,
    }

    max_shift = max(
        abs(lambda_shift_top),
        abs(mu_shift_top),
        abs(pi1_shift_top),
        abs(lambda_shift_bottom),
        abs(mu_shift_bottom),
        abs(pi1_shift_bottom),
    )

    if max_shift > 100:
        interpretation = "EXTREME regime heterogeneity - parameter estimates highly unstable"
    elif max_shift > 50:
        interpretation = "STRONG regime heterogeneity - stratified modeling recommended"
    elif max_shift > 20:
        interpretation = "MODERATE regime heterogeneity - consider stratified analysis"
    elif max_shift > 10:
        interpretation = "WEAK regime heterogeneity - global model likely adequate"
    else:
        interpretation = "STABLE parameters - homogeneous regime"

    return HeterogeneityScanResult(
        full_result=result_full,
        remove_top10_result=result_remove_top,
        remove_bottom10_result=result_remove_bottom,
        parameter_shifts=parameter_shifts,
        interpretation=interpretation,
    )


def stratified_regime_modeling(
    pam: np.ndarray,
    taxon_names: list[str],
    gene_names: list[str],
    threshold: Optional[float] = None,
    tree_method: str = "jaccard_upgma",
    verbose: bool = True,
) -> StratifiedRegimeResult:
    """Model high- and low-accessory strains separately for interpretation."""

    if verbose:
        print("=" * 70)
        print("STRATIFIED REGIME MODELING")
        print("=" * 70)

    cloud_per_strain = compute_strain_cloud_content(pam)

    if threshold is None:
        threshold = np.median(cloud_per_strain)

    high_mask = cloud_per_strain > threshold
    low_mask = ~high_mask

    n_high = high_mask.sum()
    n_low = low_mask.sum()

    if verbose:
        print(f"\nStratification (threshold: {threshold:.0f} cloud genes):")
        print(f"  High-accessory: {n_high:,} strains")
        print(f"  Low-accessory:  {n_low:,} strains")

    if verbose:
        print("\n1. Fitting high-accessory strains...")
    result_high = fit(
        pam=pam[high_mask, :],
        taxon_names=[taxon_names[i] for i in np.where(high_mask)[0]],
        gene_names=gene_names,
        tree_method=tree_method,
        verbose=False,
    )

    if verbose:
        print("2. Fitting low-accessory strains...")
    result_low = fit(
        pam=pam[low_mask, :],
        taxon_names=[taxon_names[i] for i in np.where(low_mask)[0]],
        gene_names=gene_names,
        tree_method=tree_method,
        verbose=False,
    )

    regime_comparison = {
        "Gain rate (λ)": (result_high.gain_rate, result_low.gain_rate),
        "Loss rate (μ)": (result_high.loss_rate, result_low.loss_rate),
        "π₁": (
            result_high.equilibrium_frequency,
            result_low.equilibrium_frequency,
        ),
        "λ/μ ratio": (
            result_high.gain_rate / result_high.loss_rate,
            result_low.gain_rate / result_low.loss_rate,
        ),
    }

    return StratifiedRegimeResult(
        high_accessory_result=result_high,
        low_accessory_result=result_low,
        n_high=n_high,
        n_low=n_low,
        threshold=threshold,
        regime_comparison=regime_comparison,
    )
