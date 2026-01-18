"""
Safety Report for Assembly Plugin.

Consolidates all Tier 1 safety checks into a single report
with machine-readable status for UI/CLI integration.
"""

from dataclasses import dataclass

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.safety.baseline_check import (
    BaselineSanityResult,
    check_baseline_sanity,
)
from persiste.plugins.assembly.safety.cache_reliability import (
    CacheReliabilityResult,
    check_cache_reliability,
)
from persiste.plugins.assembly.safety.identifiability import (
    IdentifiabilityResult,
    check_identifiability,
)
from persiste.plugins.assembly.screening.screening import ScreeningResult
from persiste.plugins.assembly.states.assembly_state import AssemblyState

BASE_DELTA_LL_THRESHOLD = 2.0


@dataclass
class SafetyReport:
    """
    Consolidated safety report from Tier 1 checks.

    Attributes:
        overall_status: Machine-readable severity ('ok', 'warning', 'unsafe')
        baseline_check: BaselineSanityResult
        identifiability_check: IdentifiabilityResult
        cache_check: CacheReliabilityResult
        overall_safe: Whether all checks passed (deprecated, use overall_status)
        warnings: List of warning messages
        recommendations: List of actionable recommendations
        adjusted_delta_ll_threshold: ΔLL threshold adjusted by baseline multiplier
    """

    overall_status: str
    baseline_check: BaselineSanityResult
    identifiability_check: IdentifiabilityResult
    cache_check: CacheReliabilityResult
    overall_safe: bool
    warnings: list[str]
    recommendations: list[str]
    adjusted_delta_ll_threshold: float

    def print_summary(self):
        """Print safety summary."""
        print("=" * 70)
        print("SAFETY REPORT")
        print("=" * 70)

        status_icon = {"ok": "✓", "warning": "⚠", "unsafe": "✗"}
        icon = status_icon.get(self.overall_status, "?")
        print(f"\nOverall Status: {icon} {self.overall_status.upper()}")

        baseline_status = "✓ PASS" if self.baseline_check.baseline_ok else "⚠ WARNING"
        print(f"\n1. Baseline Sanity: {baseline_status}")
        print(f"   {self.baseline_check.message}")

        ident_status = "✓ PASS" if self.identifiability_check.identifiable else "⚠ WARNING"
        print(f"\n2. Identifiability: {ident_status}")
        print(f"   {self.identifiability_check.message}")

        cache_status = "✓ PASS" if self.cache_check.inference_stable else "⚠ WARNING"
        print(f"\n3. Cache Reliability: {cache_status}")
        print(f"   {self.cache_check.message}")

        if not self.overall_safe:
            print("\n⚠ INTERPRETATION CAUTION ADVISED")
            print(f"   ΔLL threshold adjusted: {self.adjusted_delta_ll_threshold:.1f}")
            if self.warnings:
                print("\n   Warnings:")
                for w in self.warnings:
                    print(f"   - {w}")
            if self.recommendations:
                print("\n   Recommendations:")
                for r in self.recommendations:
                    print(f"   - {r}")
        else:
            print("\n✓ SAFE FOR INTERPRETATION")

        print("=" * 70)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "overall_status": self.overall_status,
            "baseline_check": self.baseline_check.to_dict(),
            "identifiability_check": self.identifiability_check.to_dict(),
            "cache_check": self.cache_check.to_dict(),
            "overall_safe": self.overall_safe,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "adjusted_delta_ll_threshold": self.adjusted_delta_ll_threshold,
        }


def run_safety_checks(
    observed_compounds: set[str],
    primitives: list[str],
    baseline: AssemblyBaseline,
    initial_state: AssemblyState,
    theta_hat: dict[str, float],
    screening_results: list[ScreeningResult],
    cache_stats: dict,
    n_baseline_samples: int = 100,
    max_depth: int = 5,
    seed: int = 42,
) -> SafetyReport:
    """
    Run all Tier 1 safety checks.

    Args:
        observed_compounds: Set of observed compound identifiers
        primitives: List of primitive building blocks
        baseline: AssemblyBaseline instance
        initial_state: Initial assembly state
        theta_hat: Estimated feature weights
        screening_results: List of ScreeningResult from screening
        cache_stats: Dictionary with cache statistics
        n_baseline_samples: Samples for baseline check (default: 100)
        max_depth: Maximum assembly depth (default: 5)
        seed: RNG seed (default: 42)

    Returns:
        SafetyReport with consolidated results
    """
    # Run individual checks
    baseline_check = check_baseline_sanity(
        observed_compounds=observed_compounds,
        primitives=primitives,
        baseline=baseline,
        initial_state=initial_state,
        n_samples=n_baseline_samples,
        max_depth=max_depth,
        seed=seed,
    )

    identifiability_check = check_identifiability(
        screening_results=screening_results,
        theta_hat=theta_hat,
    )

    cache_check = check_cache_reliability(cache_stats=cache_stats)

    # Aggregate with machine-readable status
    overall_safe = (
        baseline_check.baseline_ok
        and identifiability_check.identifiable
        and cache_check.inference_stable
    )

    # Determine overall_status
    if not baseline_check.baseline_ok and baseline_check.warning_level == "severe":
        overall_status = "unsafe"
    elif (
        not identifiability_check.identifiable
        and identifiability_check.status == "collapse_to_null"
    ):
        overall_status = "unsafe"
    elif cache_check.status == "severe":
        overall_status = "unsafe"
    elif not overall_safe:
        overall_status = "warning"
    else:
        overall_status = "ok"

    # Collect warnings and recommendations
    warnings: list[str] = []
    recommendations: list[str] = []

    if not baseline_check.baseline_ok:
        warnings.append(baseline_check.message)
    if not identifiability_check.identifiable:
        warnings.append(identifiability_check.message)
        recommendations.append(identifiability_check.recommendation)
    if not cache_check.inference_stable:
        warnings.append(cache_check.message)

    # Adjust ΔLL threshold using multiplier
    adjusted_threshold = BASE_DELTA_LL_THRESHOLD * baseline_check.delta_ll_multiplier

    return SafetyReport(
        overall_status=overall_status,
        baseline_check=baseline_check,
        identifiability_check=identifiability_check,
        cache_check=cache_check,
        overall_safe=overall_safe,
        warnings=warnings,
        recommendations=recommendations,
        adjusted_delta_ll_threshold=adjusted_threshold,
    )
