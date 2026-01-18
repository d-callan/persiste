"""
Cache Reliability Check for Assembly Plugin.

Detects importance sampling degradation by monitoring ESS
and resimulation counts.
"""

from dataclasses import dataclass


@dataclass
class CacheReliabilityResult:
    """
    Result from cache reliability check.

    Attributes:
        status: Severity level ('ok', 'warning', 'severe')
        inference_stable: Whether inference appears stable
        ess_at_theta_hat: Effective sample size at θ̂
        ess_ratio: ESS as fraction of total paths
        n_resimulations: Number of resimulations triggered
        ess_threshold: Threshold used for classification
        message: Human-readable message
    """

    status: str
    inference_stable: bool
    ess_at_theta_hat: float
    ess_ratio: float
    n_resimulations: int
    ess_threshold: float
    message: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status,
            "inference_stable": self.inference_stable,
            "ess_at_theta_hat": self.ess_at_theta_hat,
            "ess_ratio": self.ess_ratio,
            "n_resimulations": self.n_resimulations,
            "ess_threshold": self.ess_threshold,
            "message": self.message,
        }


def check_cache_reliability(
    cache_stats: dict,
    ess_threshold: float = 0.3,
) -> CacheReliabilityResult:
    """
    Cache reliability check from inference.

    Classifies ESS degradation with soft vs hard failure distinction:
    - ESS > threshold → OK
    - 0.15 < ESS ≤ threshold → warning
    - ESS ≤ 0.15 → severe

    Cost: Zero (reads cache_stats from inference)

    Args:
        cache_stats: Dictionary with cache statistics from inference
        ess_threshold: ESS threshold ratio (default: 0.3)

    Returns:
        CacheReliabilityResult with stability assessment
    """
    ess_at_theta_hat = cache_stats.get("ess_at_theta_hat", 1.0)
    n_resimulations = cache_stats.get("resimulation_count", 0)
    n_paths = cache_stats.get("n_paths", 100)

    # Handle edge case where n_paths is 0
    if n_paths <= 0:
        n_paths = 100

    # Classify with soft vs hard failure distinction
    ess_ratio = ess_at_theta_hat / n_paths

    if ess_ratio <= 0.15:
        status = "severe"
        inference_stable = False
        message = (
            f"Severe ESS degradation at θ̂ ({ess_at_theta_hat:.1f}/{n_paths}, "
            f"{ess_ratio:.1%}); importance sampling unreliable"
        )
    elif ess_ratio <= ess_threshold:
        status = "warning"
        inference_stable = False
        message = (
            f"Low ESS at θ̂ ({ess_at_theta_hat:.1f}/{n_paths}, "
            f"{ess_ratio:.1%}); importance sampling may be unreliable"
        )
    elif n_resimulations > 5:
        status = "warning"
        inference_stable = False
        message = f"Excessive resimulations ({n_resimulations}); inference may be unstable"
    else:
        status = "ok"
        inference_stable = True
        message = (
            f"Cache stable (ESS={ess_at_theta_hat:.1f}/{n_paths}, "
            f"{ess_ratio:.1%}, {n_resimulations} resims)"
        )

    return CacheReliabilityResult(
        status=status,
        inference_stable=inference_stable,
        ess_at_theta_hat=ess_at_theta_hat,
        ess_ratio=ess_ratio,
        n_resimulations=n_resimulations,
        ess_threshold=ess_threshold,
        message=message,
    )
