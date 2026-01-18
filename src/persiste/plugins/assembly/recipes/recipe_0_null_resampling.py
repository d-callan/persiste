"""
Recipe 0: Null Resampling Diagnostic

Question: "Is θ̂ significantly different from θ=0?"

Tests whether inferred constraints are real or noise by
resampling the null distribution via importance weights.
"""

from dataclasses import dataclass

import numpy as np

from persiste.plugins.assembly.diagnostics.artifacts import InferenceArtifacts
from persiste.plugins.assembly.diagnostics.suite import (
    CachedPathData,
    null_resampling,
)
from persiste.plugins.assembly.recipes.base import DiagnosticReport


@dataclass
class NullResamplingReport(DiagnosticReport):
    """
    Report from null resampling diagnostic.

    Attributes:
        severity: 'ok', 'warning', 'fail'
        observed_delta_ll: Observed ΔLL at θ̂
        null_distribution: Array of ΔLL values under null
        p_value: Fraction of null samples ≥ observed
        interpretation: Human-readable interpretation
        recommendation: What to do next
    """

    severity: str
    observed_delta_ll: float
    null_distribution: np.ndarray
    p_value: float
    interpretation: str
    recommendation: str

    def print_summary(self) -> None:
        """Print null resampling summary."""
        print("=" * 70)
        print("NULL RESAMPLING DIAGNOSTIC")
        print("=" * 70)
        print(f"\nObserved ΔLL: {self.observed_delta_ll:.2f}")
        null_mean = float(np.mean(self.null_distribution))
        null_std = float(np.std(self.null_distribution))
        print(f"Null distribution: mean={null_mean:.2f}, std={null_std:.2f}")
        print(f"P-value: {self.p_value:.3f}")
        print(f"\nSeverity: {self.severity.upper()}")
        print(f"Interpretation: {self.interpretation}")
        print(f"Recommendation: {self.recommendation}")
        print("=" * 70)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "severity": self.severity,
            "observed_delta_ll": self.observed_delta_ll,
            "null_distribution": self.null_distribution.tolist(),
            "p_value": self.p_value,
            "interpretation": self.interpretation,
            "recommendation": self.recommendation,
        }


def null_resampling_diagnostic(
    artifacts: InferenceArtifacts,
    cache: CachedPathData,
    n_resamples: int = 1000,
) -> NullResamplingReport:
    """
    Recipe 0: Null Resampling Diagnostic

    Tests whether θ̂ is significantly better than θ=0 by
    resampling the null distribution via importance weights.

    Args:
        artifacts: InferenceArtifacts from fit_assembly_constraints
        cache: CachedPathData from inference
        n_resamples: Number of bootstrap resamples (default: 1000)

    Returns:
        NullResamplingReport with p-value and interpretation

    Example:
        >>> from persiste.plugins.assembly.recipes import null_resampling_diagnostic
        >>> report = null_resampling_diagnostic(artifacts, cache)
        >>> report.print_summary()
        >>> if report.p_value < 0.05:
        ...     print("Constraints are significant!")
    """
    # Use existing implementation from diagnostics/suite.py
    result = null_resampling(artifacts, cache, n_resamples)

    # Generate interpretation with severity
    if result.p_value < 0.01:
        severity = "ok"
        interpretation = "STRONG evidence for constraints (p < 0.01)"
        recommendation = "Constraints are well-supported; proceed with interpretation"
    elif result.p_value < 0.05:
        severity = "ok"
        interpretation = "MODERATE evidence for constraints (p < 0.05)"
        recommendation = "Constraints likely real; consider additional validation"
    else:
        severity = "warning"
        interpretation = "WEAK evidence for constraints (p ≥ 0.05)"
        recommendation = "Constraints may be noise; do not over-interpret"

    return NullResamplingReport(
        severity=severity,
        observed_delta_ll=result.observed_delta_ll,
        null_distribution=np.array(result.null_distribution),
        p_value=result.p_value,
        interpretation=interpretation,
        recommendation=recommendation,
    )
