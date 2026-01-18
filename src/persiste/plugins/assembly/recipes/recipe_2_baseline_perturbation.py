"""
Recipe 2: Baseline Perturbation Sensitivity

Question: "How sensitive is θ̂ to baseline misspecification?"

Stress tests inference by perturbing baseline parameters and
measuring how θ̂ changes.
"""

from dataclasses import dataclass

import numpy as np

from persiste.plugins.assembly.diagnostics.suite import baseline_sensitivity
from persiste.plugins.assembly.recipes.base import DiagnosticReport


@dataclass
class BaselinePerturbationReport(DiagnosticReport):
    """
    Report from baseline perturbation analysis.

    Attributes:
        severity: 'ok', 'warning', 'fail'
        perturbations: List of perturbed baselines tested
        theta_hats: θ̂ under each perturbation
        log_likelihoods: Log-likelihood under each perturbation
        stable: Whether θ̂ is stable across perturbations
        interpretation: Human-readable interpretation
        recommendation: What to do next
    """

    severity: str
    perturbations: list[dict]
    theta_hats: list[dict[str, float]]
    log_likelihoods: list[float]
    stable: bool
    interpretation: str
    recommendation: str

    def print_summary(self) -> None:
        """Print baseline perturbation summary."""
        print("=" * 70)
        print("BASELINE PERTURBATION SENSITIVITY")
        print("=" * 70)
        print(f"\nTested {len(self.perturbations)} perturbations:")
        for i, (pert, ll) in enumerate(zip(self.perturbations, self.log_likelihoods)):
            print(f"  {i+1}. {pert}: LL = {ll:.2f}")
        print(f"\nStable: {self.stable}")
        print(f"\nSeverity: {self.severity.upper()}")
        print(f"Interpretation: {self.interpretation}")
        print(f"Recommendation: {self.recommendation}")
        print("=" * 70)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "severity": self.severity,
            "perturbations": self.perturbations,
            "theta_hats": self.theta_hats,
            "log_likelihoods": self.log_likelihoods,
            "stable": self.stable,
            "interpretation": self.interpretation,
            "recommendation": self.recommendation,
        }


def baseline_perturbation_sensitivity(
    inference_result: dict,
    perturbations: list[dict] | None = None,
    stability_threshold: float = 0.5,
) -> BaselinePerturbationReport:
    """
    Recipe 2: Baseline Perturbation Sensitivity

    Tests how θ̂ changes when baseline parameters are perturbed.
    Reveals fragility under model misspecification.

    Uses importance reweighting to evaluate θ̂ under perturbed baselines.

    Args:
        inference_result: Result dict from fit_assembly_constraints
            (must contain 'artifacts' and 'cache')
        perturbations: List of baseline perturbation dicts (default: kappa scaling)
        stability_threshold: Max θ deviation to be considered stable (default: 0.5)

    Returns:
        BaselinePerturbationReport with sensitivity analysis

    Example:
        >>> result = fit_assembly_constraints(...)
        >>> report = baseline_perturbation_sensitivity(result)
        >>> report.print_summary()
    """
    artifacts = inference_result.get("artifacts")
    cache = inference_result.get("cache")

    if artifacts is None or cache is None:
        msg = "Inference result must contain 'artifacts' and 'cache' for this diagnostic."
        raise ValueError(msg)

    # Use existing implementation from diagnostics/suite.py
    result = baseline_sensitivity(
        artifacts=artifacts,
        cache=cache,
        perturbations=perturbations,
        stability_threshold=stability_threshold,
    )

    # Generate interpretation with severity
    if not result.stable:
        # Check how unstable by looking at LL variance
        ll_std = float(np.std(result.log_likelihoods)) if result.log_likelihoods else 0.0
        if ll_std > 5.0:
            severity = "fail"
            interpretation = "EXTREME sensitivity to baseline"
            recommendation = "Validate baseline before trusting constraints"
        else:
            severity = "warning"
            interpretation = "MODERATE sensitivity to baseline"
            recommendation = "Use conservative ΔLL thresholds"
    else:
        severity = "ok"
        interpretation = "LOW sensitivity to baseline (robust)"
        recommendation = "Constraints appear stable under perturbations"

    return BaselinePerturbationReport(
        severity=severity,
        perturbations=result.perturbations,
        theta_hats=result.theta_estimates,
        log_likelihoods=result.log_likelihoods,
        stable=result.stable,
        interpretation=interpretation,
        recommendation=recommendation,
    )
