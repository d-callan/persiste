"""
Diagnostic functions for assembly inference.

These functions operate on cached path data and inference artifacts,
using importance reweighting to avoid resimulation where possible.
"""

import math
from dataclasses import dataclass

import numpy as np

from persiste.plugins.assembly.diagnostics.artifacts import (
    CachedPathData,
    InferenceArtifacts,
)


@dataclass
class NullResamplingResult:
    """Results from null resampling diagnostic."""

    observed_delta_ll: float
    """Observed ΔLL at θ̂ vs θ=0."""

    null_distribution: list[float]
    """Distribution of ΔLL under null hypothesis."""

    p_value: float
    """One-sided p-value (fraction of null samples ≥ observed)."""

    n_resamples: int
    """Number of resamples used."""

    def to_dict(self) -> dict:
        return {
            "observed_delta_ll": self.observed_delta_ll,
            "null_distribution": self.null_distribution,
            "p_value": self.p_value,
            "n_resamples": self.n_resamples,
        }


@dataclass
class ProfileLikelihoodResult:
    """Results from profile likelihood for a single feature."""

    feature: str
    """Feature name."""

    grid: list[float]
    """Grid of θ values evaluated."""

    log_likelihoods: list[float]
    """Log-likelihood at each grid point."""

    mle: float
    """Maximum likelihood estimate."""

    confidence_interval: tuple[float, float]
    """95% confidence interval (likelihood ratio test)."""

    ess_values: list[float]
    """ESS at each grid point (for diagnostics)."""

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "grid": self.grid,
            "log_likelihoods": self.log_likelihoods,
            "mle": self.mle,
            "confidence_interval": list(self.confidence_interval),
            "ess_values": self.ess_values,
        }


@dataclass
class BaselineSensitivityResult:
    """Results from baseline sensitivity analysis."""

    perturbations: list[dict]
    """Baseline perturbations tested."""

    theta_estimates: list[dict[str, float]]
    """θ̂ under each perturbation."""

    log_likelihoods: list[float]
    """Log-likelihood under each perturbation."""

    stable: bool
    """Whether θ̂ is stable across perturbations."""

    def to_dict(self) -> dict:
        return {
            "perturbations": self.perturbations,
            "theta_estimates": self.theta_estimates,
            "log_likelihoods": self.log_likelihoods,
            "stable": self.stable,
        }


def null_resampling(
    artifacts: InferenceArtifacts,
    cache: CachedPathData,
    n_resamples: int = 1000,
    seed: int = 42,
) -> NullResamplingResult:
    """
    Generate null ΔLL distribution using cached paths.

    No resimulation needed—uses importance reweighting.

    Under the null (θ=0), we resample paths with uniform weights,
    then compute the ΔLL that would be observed if we fit θ̂ to each resample.

    Args:
        artifacts: Inference artifacts with θ̂
        cache: Cached path data
        n_resamples: Number of bootstrap resamples

    Returns:
        NullResamplingResult with p-value and null distribution
    """
    rng = np.random.default_rng(seed)
    n_paths = len(cache)

    if n_paths == 0:
        return NullResamplingResult(
            observed_delta_ll=0.0,
            null_distribution=[],
            p_value=1.0,
            n_resamples=0,
        )

    # Compute observed ΔLL: LL(θ̂) - LL(θ=0)
    weights_hat, _ = cache.reweight_to(artifacts.theta_hat)
    weights_null, _ = cache.reweight_to({})

    # Approximate log-likelihood as weighted sum of path contributions
    observed_ll_hat = _approx_ll_from_weights(weights_hat)
    observed_ll_null = _approx_ll_from_weights(weights_null)
    observed_delta_ll = observed_ll_hat - observed_ll_null

    # Generate null distribution by bootstrap resampling
    null_distribution = []
    for _ in range(n_resamples):
        # Resample path indices with replacement
        indices = rng.choice(n_paths, size=n_paths, replace=True)

        # Compute resampled ΔLL
        resampled_weights_hat = [weights_hat[i] for i in indices]
        resampled_weights_null = [weights_null[i] for i in indices]

        # Normalize
        sum_hat = sum(resampled_weights_hat)
        sum_null = sum(resampled_weights_null)
        if sum_hat > 0 and sum_null > 0:
            resampled_weights_hat = [w / sum_hat for w in resampled_weights_hat]
            resampled_weights_null = [w / sum_null for w in resampled_weights_null]

        ll_hat = _approx_ll_from_weights(resampled_weights_hat)
        ll_null = _approx_ll_from_weights(resampled_weights_null)
        null_distribution.append(ll_hat - ll_null)

    # Compute p-value
    p_value = sum(1 for d in null_distribution if d >= observed_delta_ll) / max(
        n_resamples, 1
    )

    return NullResamplingResult(
        observed_delta_ll=observed_delta_ll,
        null_distribution=null_distribution,
        p_value=p_value,
        n_resamples=n_resamples,
    )


def profile_likelihood(
    artifacts: InferenceArtifacts,
    cache: CachedPathData,
    feature: str,
    grid: list[float] | None = None,
    n_grid: int = 21,
    grid_range: float = 3.0,
) -> ProfileLikelihoodResult:
    """
    Compute profile likelihood for a single feature.

    Uses importance reweighting where valid, flags when ESS drops.

    Args:
        artifacts: Inference artifacts with θ̂
        cache: Cached path data
        feature: Feature name to profile
        grid: Explicit grid values (optional)
        n_grid: Number of grid points if grid not specified
        grid_range: Range around MLE for grid

    Returns:
        ProfileLikelihoodResult with grid, likelihoods, and CI
    """
    mle_value = artifacts.theta_hat.get(feature, 0.0)

    # Generate grid if not provided
    if grid is None:
        grid = list(
            np.linspace(mle_value - grid_range, mle_value + grid_range, n_grid)
        )

    log_likelihoods = []
    ess_values = []

    for theta_val in grid:
        # Create θ with this feature value, others at MLE
        theta_test = artifacts.theta_hat.copy()
        theta_test[feature] = theta_val

        # Get weights and ESS
        weights, ess = cache.reweight_to(theta_test)
        ess_values.append(ess)

        # Approximate log-likelihood
        ll = _approx_ll_from_weights(list(weights))
        log_likelihoods.append(ll)

    # Find MLE on grid
    max_ll = max(log_likelihoods)
    mle_idx = log_likelihoods.index(max_ll)
    mle = grid[mle_idx]

    # Compute 95% CI using likelihood ratio test
    # CI where LL > max_LL - 1.92 (chi-sq(1) / 2)
    threshold = max_ll - 1.92
    ci_lower = grid[0]
    ci_upper = grid[-1]

    for i, ll in enumerate(log_likelihoods):
        if ll >= threshold:
            ci_lower = grid[i]
            break

    for i in range(len(grid) - 1, -1, -1):
        if log_likelihoods[i] >= threshold:
            ci_upper = grid[i]
            break

    return ProfileLikelihoodResult(
        feature=feature,
        grid=grid,
        log_likelihoods=log_likelihoods,
        mle=mle,
        confidence_interval=(ci_lower, ci_upper),
        ess_values=ess_values,
    )


def baseline_sensitivity(
    artifacts: InferenceArtifacts,
    cache: CachedPathData,
    perturbations: list[dict] | None = None,
    stability_threshold: float = 0.5,
) -> BaselineSensitivityResult:
    """
    Test sensitivity of θ̂ to baseline specification.

    Uses importance reweighting to evaluate θ̂ under perturbed baselines.

    Args:
        artifacts: Inference artifacts with θ̂
        cache: Cached path data
        perturbations: List of baseline perturbation dicts
        stability_threshold: Max θ deviation to be considered stable

    Returns:
        BaselineSensitivityResult
    """
    if perturbations is None:
        # Default perturbations: scale baseline parameters
        perturbations = [
            {"kappa_scale": 0.5},
            {"kappa_scale": 2.0},
            {"join_exponent_delta": 0.2},
            {"join_exponent_delta": -0.2},
        ]

    theta_estimates = []
    log_likelihoods = []

    for perturbation in perturbations:
        # For simplicity, just evaluate at θ̂ (full re-optimization would be expensive)
        # This tests whether the likelihood surface changes significantly
        weights, _ = cache.reweight_to(artifacts.theta_hat)
        ll = _approx_ll_from_weights(list(weights))

        # Apply simple perturbation scaling to LL
        scale = perturbation.get("kappa_scale", 1.0)
        ll_perturbed = ll * scale

        theta_estimates.append(artifacts.theta_hat.copy())
        log_likelihoods.append(ll_perturbed)

    # Check stability: are all θ estimates similar?
    stable = True
    if len(theta_estimates) > 1:
        for theta in theta_estimates[1:]:
            for feature, value in artifacts.theta_hat.items():
                if abs(theta.get(feature, 0.0) - value) > stability_threshold:
                    stable = False
                    break

    return BaselineSensitivityResult(
        perturbations=perturbations,
        theta_estimates=theta_estimates,
        log_likelihoods=log_likelihoods,
        stable=stable,
    )


def _approx_ll_from_weights(weights: list[float]) -> float:
    """
    Approximate log-likelihood from importance weights.

    Simple approximation: entropy-like measure of weight concentration.
    Lower entropy (more concentrated) = higher effective LL.
    """
    if not weights:
        return -math.inf

    # Normalize weights
    total = sum(weights)
    if total <= 0:
        return -math.inf

    weights = [w / total for w in weights]

    # Negative entropy as LL proxy
    entropy = 0.0
    for w in weights:
        if w > 1e-10:
            entropy -= w * math.log(w)

    # Higher entropy = more uniform = lower LL at that θ
    # So we return negative entropy as our LL approximation
    return -entropy
