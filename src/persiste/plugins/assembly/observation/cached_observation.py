"""
Cached assembly observation model with importance-sampling support.

This adapter wraps the Rust simulation and cache management, keeping
ConstraintInference pure and unaware of Monte Carlo machinery.
"""

import logging
from dataclasses import dataclass
from typing import Any

import persiste_rust

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.states.assembly_state import AssemblyState

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for importance-sampling cache."""

    trust_radius: float = 1.0
    """L∞ trust region radius. Default: 1.0 (each θ component may move ±1.0)."""

    ess_threshold: float = 0.3
    """ESS threshold ratio. Default: 0.3 (30% of cached trajectories)."""

    max_weight_variance: float = 100.0
    """Maximum weight variance before early warning."""

    topology_sensitive_features: list[str] | None = None
    """Features that can change graph topology (trigger hard resimulation)."""

    topology_change_threshold: float = 2.0
    """Threshold for topology change detection."""


@dataclass
class SimulationSettings:
    """Settings for Gillespie simulation."""

    n_samples: int = 100
    """Number of trajectories to simulate."""

    t_max: float = 50.0
    """Maximum simulation time."""

    burn_in: float = 25.0
    """Burn-in time (discard early states)."""

    max_depth: int = 5
    """Maximum assembly depth for graph exploration."""


@dataclass
class CacheState:
    """Internal cache state."""

    feature_counts: list[dict[str, int]]
    """Feature counts from each cached trajectory."""

    final_state_ids: list[int]
    """Final state IDs from each trajectory."""

    theta_ref: dict[str, float]
    """Reference θ at which trajectories were simulated."""

    n_transitions: list[int]
    """Number of transitions in each trajectory."""


class CachedAssemblyObservationModel:
    """
    Assembly observation model with importance-sampling cache.

    Design: Adapter owns cache validity; ConstraintInference just asks for
    log_likelihood(θ). This keeps ConstraintInference pure and unaware of
    Monte Carlo machinery.

    Features:
    - Parallel trajectory simulation via Rust
    - Importance sampling with ESS monitoring
    - L∞ trust region for cache validity
    - Topology guard for hard resimulation triggers
    """

    def __init__(
        self,
        *,
        primitives: list[str],
        baseline: AssemblyBaseline,
        initial_state: AssemblyState,
        simulation: SimulationSettings | None = None,
        cache_config: CacheConfig | None = None,
        rng_seed: int | None = None,
    ):
        """
        Initialize cached observation model.

        Args:
            primitives: List of primitive building blocks
            baseline: Baseline rate model
            initial_state: Starting state for simulation
            simulation: Simulation settings (default: SimulationSettings())
            cache_config: Cache configuration (default: CacheConfig())
            rng_seed: Optional RNG seed for reproducibility
        """
        self.primitives = primitives
        self.baseline = baseline
        self.initial_state = initial_state
        self.simulation = simulation or SimulationSettings()
        self.cache_config = cache_config or CacheConfig()
        self.rng_seed = rng_seed or 42

        self._cache: CacheState | None = None
        self._resimulation_count = 0
        self._last_ess_ratio: float | None = 1.0

    def get_latent_states(
        self, constraint: AssemblyConstraint
    ) -> dict[int, float]:
        """
        Get latent state distribution for given constraint.

        Transparently handles cache management - either reweights cached
        trajectories or resimulates if cache is invalid.
        Args:
            constraint: Constraint model with feature weights

        Returns:
            Dict mapping state_id -> probability
        """
        theta = constraint.feature_weights

        if self._cache is None:
            self._initialize_cache(theta)
            self._last_ess_ratio = 1.0
            return self._get_current_states()

        # Check topology guard first (hard resimulation trigger)
        if self.cache_config.topology_sensitive_features:
            affected = persiste_rust.check_topology_change(
                theta=theta,
                theta_ref=self._cache.theta_ref,
                sensitive_features=self.cache_config.topology_sensitive_features,
                threshold=self.cache_config.topology_change_threshold,
            )
            if affected:
                logger.info(
                    f"Topology change detected in features {affected}. Resimulating."
                )
                self._resimulate(theta, reason=f"topology_change:{affected}")
                return self._get_current_states()

        # Evaluate cache validity
        status = persiste_rust.evaluate_cache(
            path_feature_counts=self._cache.feature_counts,
            final_state_ids=self._cache.final_state_ids,
            theta=theta,
            theta_ref=self._cache.theta_ref,
            trust_radius=self.cache_config.trust_radius,
            ess_threshold=self.cache_config.ess_threshold,
            max_weight_variance=self.cache_config.max_weight_variance,
        )

        if status["valid"]:
            logger.debug(f"Cache valid. ESS={status['ess']:.1f}")
            self._update_ess_ratio(status.get("ess"))
            return status["latent_states"]
        else:
            logger.info(f"Cache invalid: {status['reason']}. Resimulating.")
            self._resimulate(theta, reason=status["reason"])
            return self._get_current_states()

    def _initialize_cache(self, theta_ref: dict[str, float]) -> None:
        """Run Gillespie simulation and populate cache."""
        logger.info(
            f"Initializing cache with {self.simulation.n_samples} trajectories at θ_ref={theta_ref}"
        )

        results = persiste_rust.simulate_assembly_trajectories(
            primitives=self.primitives,
            initial_parts=self.initial_state.get_parts_list(),
            theta=theta_ref,
            n_samples=self.simulation.n_samples,
            t_max=self.simulation.t_max,
            burn_in=self.simulation.burn_in,
            max_depth=self.simulation.max_depth,
            seed=self.rng_seed,
            kappa=self.baseline.kappa,
            join_exponent=self.baseline.join_exponent,
            split_exponent=self.baseline.split_exponent,
            decay_rate=self.baseline.decay_rate,
        )

        self._cache = CacheState(
            feature_counts=[r["feature_counts"] for r in results],
            final_state_ids=[r["final_state_id"] for r in results],
            theta_ref=theta_ref.copy(),
            n_transitions=[r["n_transitions"] for r in results],
        )
        self._last_ess_ratio = 1.0

    def _resimulate(self, theta_ref: dict[str, float], reason: str) -> None:
        """Resimulate at new reference point."""
        self._resimulation_count += 1
        logger.info(
            f"Resimulation #{self._resimulation_count}: {reason}"
        )
        # Use a different seed for each resimulation
        self.rng_seed = self.rng_seed + self._resimulation_count * 1000
        self._initialize_cache(theta_ref)

    def _get_current_states(self) -> dict[int, float]:
        """Get state distribution from current cache at θ_ref (uniform weights)."""
        if self._cache is None:
            return {}

        state_counts: dict[int, int] = {}
        for state_id in self._cache.final_state_ids:
            state_counts[state_id] = state_counts.get(state_id, 0) + 1

        total = len(self._cache.final_state_ids)
        return {sid: count / total for sid, count in state_counts.items()}

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics for diagnostics."""
        if self._cache is None:
            return {"initialized": False}

        return {
            "initialized": True,
            "n_paths": len(self._cache.feature_counts),
            "theta_ref": self._cache.theta_ref,
            "resimulation_count": self._resimulation_count,
            "mean_transitions": (
                sum(self._cache.n_transitions) / len(self._cache.n_transitions)
                if self._cache.n_transitions
                else 0
            ),
        }

    def invalidate_cache(self) -> None:
        """Force cache invalidation (for testing or after external changes)."""
        self._cache = None

    def _update_ess_ratio(self, ess_value: float | None) -> None:
        """Track latest ESS ratio for downstream weighting."""
        if ess_value is None or self._cache is None:
            self._last_ess_ratio = None
            return
        n_paths = len(self._cache.feature_counts) or 1
        ratio = ess_value / n_paths
        self._last_ess_ratio = max(0.0, min(1.0, ratio))

    @property
    def current_ess_ratio(self) -> float:
        """Return last known ESS ratio (0-1)."""
        if self._last_ess_ratio is None:
            return 1.0
        return self._last_ess_ratio
