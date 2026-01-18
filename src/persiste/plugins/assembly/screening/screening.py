"""
Screening criterion and adaptive grid search for hypothesis triage.

Uses normalized ΔLL to avoid θ with large but noisy effects crowding out stable ones.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass

from persiste.plugins.assembly.likelihood import compute_observation_ll
from persiste.plugins.assembly.screening.steady_state import SteadyStateAssemblyModel
from persiste.plugins.assembly.states.assembly_state import AssemblyState


@dataclass
class ScreeningResult:
    """Result from screening a single hypothesis."""

    theta: dict[str, float]
    """Feature weights for this hypothesis."""

    delta_ll: float
    """ΔLL relative to θ=0 baseline."""

    normalized_delta_ll: float
    """ΔLL / approximate_stderr (for stable ranking)."""

    passed: bool
    """Whether hypothesis passed screening threshold."""

    rank: int = 0
    """Rank among screened hypotheses (1 = best)."""

    absolute_ll: float | None = None
    """Deterministic absolute LL (baseline + ΔLL)."""

    null_ll: float | None = None
    """Deterministic null LL used for ΔLL normalization."""


@dataclass
class AdaptiveScreeningGrid:
    """
    Lightweight adaptive grid search for screening.

    Default strategy:
    1. Start with a coarse symmetric grid around θ=0
    2. Keep top K candidates by normalized ΔLL
    3. Refine locally around those candidates
    4. Stop after fixed budget
    """

    feature_names: list[str]
    """Feature names to screen."""

    coarse_range: float = 2.0
    """Range for coarse grid (±coarse_range around 0)."""

    coarse_steps: int = 5
    """Number of steps in coarse grid per feature."""

    refine_radius: float = 0.5
    """Radius for local refinement around winners."""

    refine_steps: int = 3
    """Number of refinement steps per feature."""

    budget: int = 100
    """Total screening budget (evaluations)."""

    top_k: int = 10
    """Number of top candidates to refine."""

    def coarse_grid(self) -> Iterator[dict[str, float]]:
        """Generate coarse grid points."""
        if not self.feature_names:
            yield {}
            return

        # Generate 1D grid for each feature
        step = 2 * self.coarse_range / max(self.coarse_steps - 1, 1)
        values = [
            -self.coarse_range + i * step for i in range(self.coarse_steps)
        ]

        # For simplicity, screen each feature independently first
        # (full grid would be exponential)
        yield {}  # Null model

        for feature in self.feature_names:
            for val in values:
                if val != 0.0:  # Skip null model (already yielded)
                    yield {feature: val}

    def refine_around(self, center: dict[str, float]) -> Iterator[dict[str, float]]:
        """Generate refinement grid around a center point."""
        if not self.feature_names:
            yield center
            return

        step = 2 * self.refine_radius / max(self.refine_steps - 1, 1)

        for feature in self.feature_names:
            center_val = center.get(feature, 0.0)
            for i in range(self.refine_steps):
                val = center_val - self.refine_radius + i * step
                if val != center_val:
                    refined = center.copy()
                    refined[feature] = val
                    yield refined


def screen_hypotheses(
    hypotheses: list[dict[str, float]],
    model: SteadyStateAssemblyModel,
    observed_compounds: set[str],
    initial_state: AssemblyState,
    threshold: float = 2.0,
    *,
    observation_records: list[dict] | None = None,
    max_depth: int | None = None,
) -> list[ScreeningResult]:
    """
    Screen candidate θ values using deterministic approximation.

    Criterion: normalized ΔLL to avoid θ with large but noisy effects
    crowding out stable ones.

    Args:
        hypotheses: List of θ dicts to screen
        model: Steady-state approximation model
        observed_compounds: Set of observed compound identifiers
        initial_state: Starting state
        threshold: Normalized ΔLL threshold for passing (default: 2.0)

    Returns:
        List of ScreeningResult, sorted by normalized_delta_ll descending
    """
    if not hypotheses:
        return []

    # Compute baseline (θ=0) occupancy and likelihood
    null_latent_states = model.expected_occupancy({}, initial_state)
    if not null_latent_states:
        return []

    effective_max_depth = max_depth or model.config.max_depth

    ll_null = compute_observation_ll(
        null_latent_states,
        observed_compounds,
        model.primitives,
        observation_records=observation_records,
        max_depth=effective_max_depth,
        ess_ratio=1.0,
    )

    results = []
    for theta in hypotheses:
        delta_ll = model.approximate_log_likelihood(
            theta,
            observed_compounds,
            initial_state,
            observation_records=observation_records,
            max_depth=max_depth,
            null_latent_states=null_latent_states,
        )

        # Fisher-ish scaling (cheap variance approximation)
        approx_var = _estimate_screening_variance(theta)
        normalized = delta_ll / (math.sqrt(approx_var) + 1e-6)

        results.append(
            ScreeningResult(
                theta=theta.copy(),
                delta_ll=delta_ll,
                normalized_delta_ll=normalized,
                passed=normalized > threshold,
                absolute_ll=ll_null + delta_ll,
                null_ll=ll_null,
            )
        )

    # Sort by normalized ΔLL descending
    results.sort(key=lambda r: r.normalized_delta_ll, reverse=True)

    # Assign ranks
    for i, result in enumerate(results):
        result.rank = i + 1

    return results


def adaptive_screen(
    model: SteadyStateAssemblyModel,
    observed_compounds: set[str],
    initial_state: AssemblyState,
    grid: AdaptiveScreeningGrid,
    threshold: float = 2.0,
    *,
    observation_records: list[dict] | None = None,
    max_depth: int | None = None,
) -> list[ScreeningResult]:
    """
    Run adaptive screening with coarse-then-refine strategy.

    Args:
        model: Steady-state approximation model
        observed_compounds: Set of observed compound identifiers
        initial_state: Starting state
        grid: Adaptive grid configuration
        threshold: Normalized ΔLL threshold

    Returns:
        List of ScreeningResult from all evaluations, sorted by score
    """
    all_results: list[ScreeningResult] = []
    evaluated: set[str] = set()  # Track evaluated θ (as string key)
    budget_remaining = grid.budget

    # Phase 1: Coarse grid
    coarse_hypotheses = []
    for theta in grid.coarse_grid():
        if budget_remaining <= 0:
            break
        key = _theta_key(theta)
        if key not in evaluated:
            coarse_hypotheses.append(theta)
            evaluated.add(key)
            budget_remaining -= 1

    coarse_results = screen_hypotheses(
        coarse_hypotheses,
        model,
        observed_compounds,
        initial_state,
        threshold,
        observation_records=observation_records,
        max_depth=max_depth,
    )
    all_results.extend(coarse_results)

    # Phase 2: Refine around top K
    top_results = [r for r in coarse_results if r.passed][:grid.top_k]

    for winner in top_results:
        if budget_remaining <= 0:
            break

        refine_hypotheses = []
        for theta in grid.refine_around(winner.theta):
            if budget_remaining <= 0:
                break
            key = _theta_key(theta)
            if key not in evaluated:
                refine_hypotheses.append(theta)
                evaluated.add(key)
                budget_remaining -= 1

        if refine_hypotheses:
            refine_results = screen_hypotheses(
                refine_hypotheses,
                model,
                observed_compounds,
                initial_state,
                threshold,
                observation_records=observation_records,
                max_depth=max_depth,
            )
            all_results.extend(refine_results)

    # Re-sort and re-rank all results
    all_results.sort(key=lambda r: r.normalized_delta_ll, reverse=True)
    for i, result in enumerate(all_results):
        result.rank = i + 1

    return all_results


def _estimate_screening_variance(theta: dict[str, float]) -> float:
    """
    Cheap variance estimate for normalization.

    Simple heuristic: variance scales with sum of squared θ values.
    This prevents large θ from dominating purely due to magnitude.
    """
    if not theta:
        return 1.0

    sum_sq = sum(v * v for v in theta.values())
    return 1.0 + sum_sq


def _theta_key(theta: dict[str, float]) -> str:
    """Create hashable key for θ dict."""
    items = sorted(theta.items())
    return str(items)
