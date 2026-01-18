"""
Identifiability Screen for Assembly Plugin.

Detects flat likelihood surfaces or collapse-to-null behavior
by reusing screening results.
"""

from dataclasses import dataclass

import numpy as np

from persiste.plugins.assembly.screening.screening import ScreeningResult


@dataclass
class IdentifiabilityResult:
    """
    Result from identifiability screen.

    Attributes:
        status: Failure mode ('ok', 'flat', 'collapse_to_null')
        identifiable: Whether constraints appear identifiable
        evidence: Dictionary of evidence metrics
        message: Human-readable message
        recommendation: Actionable recommendation
    """

    status: str
    identifiable: bool
    evidence: dict[str, float]
    message: str
    recommendation: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status,
            "identifiable": self.identifiable,
            "evidence": self.evidence,
            "message": self.message,
            "recommendation": self.recommendation,
        }


def check_identifiability(
    screening_results: list[ScreeningResult],
    theta_hat: dict[str, float],
) -> IdentifiabilityResult:
    """
    Identifiability screen from screening results.

    Distinguishes two failure modes:
    - 'flat': Flat likelihood surface → needs more data
    - 'collapse_to_null': θ̂ collapsed to zero → constraints unsupported

    Cost: Zero (reuses screening data)

    Args:
        screening_results: List of ScreeningResult from adaptive screening
        theta_hat: Estimated feature weights

    Returns:
        IdentifiabilityResult with status and recommendation
    """
    if not screening_results:
        # No screening data - assume identifiable (conservative)
        return IdentifiabilityResult(
            status="ok",
            identifiable=True,
            evidence={
                "screening_variance": 0.0,
                "theta_hat_near_zero": False,
                "curvature_proxy": 0.0,
                "top_k_separation": 0.0,
            },
            message="No screening data available; identifiability unknown",
            recommendation="Run screening to assess identifiability",
        )

    # Variance in normalized ΔLL
    delta_lls = [r.normalized_delta_ll for r in screening_results]
    screening_variance = float(np.var(delta_lls))

    # Check if θ̂ near zero
    theta_values = list(theta_hat.values()) if theta_hat else [0.0]
    theta_norm = float(np.linalg.norm(theta_values))
    theta_hat_near_zero = theta_norm < 0.1

    # Curvature proxy: difference between top-1 and top-3
    if len(screening_results) >= 3:
        curvature_proxy = (
            screening_results[0].normalized_delta_ll
            - screening_results[2].normalized_delta_ll
        )
    else:
        curvature_proxy = 0.0

    # Top-k separation (same as curvature for now)
    top_k_separation = curvature_proxy

    # Classify with explicit failure modes
    top_delta_ll = screening_results[0].normalized_delta_ll if screening_results else 0.0

    if theta_hat_near_zero and top_delta_ll < 2.0:
        status = "collapse_to_null"
        identifiable = False
        message = "θ̂ collapsed to zero; no constraint signal detected"
        recommendation = "Constraints unsupported or baseline dominant; do not interpret θ"
    elif screening_variance < 1.0 and curvature_proxy < 2.0:
        status = "flat"
        identifiable = False
        message = "Flat likelihood surface detected; constraints may not be identifiable"
        recommendation = "Needs more data or stronger constraints; increase sample size"
    else:
        status = "ok"
        identifiable = True
        message = "Constraints appear identifiable from screening"
        recommendation = "Proceed with interpretation"

    return IdentifiabilityResult(
        status=status,
        identifiable=identifiable,
        evidence={
            "screening_variance": screening_variance,
            "theta_hat_near_zero": theta_hat_near_zero,
            "curvature_proxy": curvature_proxy,
            "top_k_separation": top_k_separation,
        },
        message=message,
        recommendation=recommendation,
    )
