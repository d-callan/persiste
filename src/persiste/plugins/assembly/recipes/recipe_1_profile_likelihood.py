"""
Recipe 1: Profile Likelihood Sweeps

Question: "How well-constrained is each parameter?"

Characterizes uncertainty in θ̂ by computing profile likelihood
for individual features.
"""

from dataclasses import dataclass

import numpy as np

from persiste.plugins.assembly.diagnostics.suite import profile_likelihood
from persiste.plugins.assembly.recipes.base import DiagnosticReport


@dataclass
class ProfileLikelihoodReport(DiagnosticReport):
    """
    Report from profile likelihood analysis.

    Attributes:
        severity: 'ok', 'warning', 'fail'
        feature_name: Feature being profiled
        mle: Maximum likelihood estimate
        confidence_interval: 95% CI
        ci_width: Width of confidence interval
        grid_values: Grid of θ values tested
        log_likelihoods: Log-likelihood at each grid point
        interpretation: Human-readable interpretation
        recommendation: What to do next
    """

    severity: str
    feature_name: str
    mle: float
    confidence_interval: tuple[float, float]
    ci_width: float
    grid_values: np.ndarray
    log_likelihoods: np.ndarray
    interpretation: str
    recommendation: str

    def print_summary(self) -> None:
        """Print profile likelihood summary."""
        print("=" * 70)
        print(f"PROFILE LIKELIHOOD: {self.feature_name}")
        print("=" * 70)
        print(f"\nMLE: {self.mle:.2f}")
        ci_lo, ci_hi = self.confidence_interval
        print(f"95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
        print(f"CI width: {self.ci_width:.2f}")
        print(f"\nSeverity: {self.severity.upper()}")
        print(f"Interpretation: {self.interpretation}")
        print(f"Recommendation: {self.recommendation}")
        print("=" * 70)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "severity": self.severity,
            "feature_name": self.feature_name,
            "mle": self.mle,
            "confidence_interval": list(self.confidence_interval),
            "ci_width": self.ci_width,
            "grid_values": self.grid_values.tolist(),
            "log_likelihoods": self.log_likelihoods.tolist(),
            "interpretation": self.interpretation,
            "recommendation": self.recommendation,
        }

    def plot(self, save_path: str | None = None):
        """
        Plot profile likelihood curve.

        Args:
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.grid_values, self.log_likelihoods, "b-", lw=2)
        ax.axvline(self.mle, color="r", linestyle="--", label=f"MLE = {self.mle:.2f}")
        ax.axvline(self.confidence_interval[0], color="g", linestyle=":", label="95% CI")
        ax.axvline(self.confidence_interval[1], color="g", linestyle=":")
        ax.set_xlabel(f"{self.feature_name}", fontsize=12)
        ax.set_ylabel("Log-likelihood", fontsize=12)
        ax.set_title(
            f"Profile Likelihood: {self.feature_name}", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


def profile_likelihood_sweep(
    inference_result: dict,
    feature_name: str,
    n_grid: int = 21,
) -> ProfileLikelihoodReport:
    """
    Recipe 1: Profile Likelihood Sweep

    Computes profile likelihood for a single feature to
    characterize uncertainty in θ̂.

    Args:
        inference_result: Result dict from fit_assembly_constraints
            (must contain 'artifacts' and 'cache')
        feature_name: Feature to profile
        n_grid: Number of grid points (default: 21)

    Returns:
        ProfileLikelihoodReport with CI and interpretation

    Example:
        >>> result = fit_assembly_constraints(...)
        >>> report = profile_likelihood_sweep(result, 'reuse_count')
        >>> report.print_summary()
    """
    artifacts = inference_result.get("artifacts")
    cache = inference_result.get("cache")

    if artifacts is None or cache is None:
        msg = "Inference result must contain 'artifacts' and 'cache' for this diagnostic."
        raise ValueError(msg)

    # Use existing implementation from diagnostics/suite.py
    result = profile_likelihood(artifacts, cache, feature_name, n_grid)

    # Calculate CI width
    ci_width = result.confidence_interval[1] - result.confidence_interval[0]

    # Generate interpretation with severity
    if ci_width < 0.5:
        severity = "ok"
        interpretation = "WELL-CONSTRAINED parameter (narrow CI)"
        recommendation = "Parameter estimate is reliable"
    elif ci_width < 1.5:
        severity = "ok"
        interpretation = "MODERATELY constrained parameter"
        recommendation = "Parameter estimate has moderate uncertainty"
    else:
        severity = "warning"
        interpretation = "POORLY constrained parameter (wide CI)"
        recommendation = "Parameter estimate is uncertain; interpret cautiously"

    return ProfileLikelihoodReport(
        severity=severity,
        feature_name=feature_name,
        mle=result.mle,
        confidence_interval=result.confidence_interval,
        ci_width=ci_width,
        grid_values=np.array(result.grid),
        log_likelihoods=np.array(result.log_likelihoods),
        interpretation=interpretation,
        recommendation=recommendation,
    )
