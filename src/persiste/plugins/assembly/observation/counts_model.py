"""
Counts-based observation model for Assembly Theory.

Models observed abundances of assembly states directly using count-based likelihoods.
Supports extensible constraint diagnostics (e.g., founder vs derived reuse).
"""

from typing import Any, TYPE_CHECKING

import numpy as np
from scipy.special import gammaln

from persiste.core.observation_models import ObservationModel

if TYPE_CHECKING:
    from persiste.core.data import ObservedTransitions

class AssemblyCountsModel(ObservationModel):
    """
    Counts-based observation model.

    Interpretations:
    - Observed abundances are Poisson-distributed around latent expectations.
    - Expected abundance lambda_s = N * P(state_s | theta)
    - Supports diagnostic extraction for various constraints without hardcoding logic.
    """

    def __init__(
        self,
        graph: Any = None,
        detection_efficiency: float = 1.0,
        background_noise: float = 1e-6,
    ):
        """
        Initialize counts model.

        Args:
            graph: AssemblyGraph for state lookups.
            detection_efficiency: Scaling factor for expected counts.
            background_noise: Minimum expected count (floor).
        """
        self.graph = graph
        self.detection_efficiency = detection_efficiency
        self.background_noise = background_noise

    def rate(self, i: int, j: int) -> float:
        """Required by ObservationModel interface (unused for static counts)."""
        return 0.0

    def log_likelihood(
        self,
        data: "ObservedTransitions",
        baseline: Any,
        graph: Any,
    ) -> float:
        """
        Compute log-likelihood of observations.

        Expected data format:
        - observed_counts: Mapping of state ID (int) or string to observed count.
        - latent_states: Mapping of state ID (int) to probability/occupancy.
        """
        observed_counts = getattr(data, "observed_counts", {})
        latent_states = getattr(data, "latent_states", {})
        total_observed_units = getattr(data, "total_observed_units", None)

        return self.compute_log_likelihood(
            observed_counts,
            latent_states,
            total_observed_units=total_observed_units
        )

    def compute_log_likelihood(
        self,
        observed_counts: dict[int | str, float],
        latent_states: dict[int, float],
        total_observed_units: float | None = None,
    ) -> float:
        """
        Compute Poisson log-likelihood of observed counts.

        Args:
            observed_counts: Mapping of state ID (int) or string to observed count.
            latent_states: Mapping of state ID (int) to probability/occupancy.
            total_observed_units: Total count sum. If None, inferred from observed_counts.
        """
        if not latent_states:
            return -1e10

        if total_observed_units is None:
            total_observed_units = sum(observed_counts.values())

        log_lik = 0.0

        all_latent_ids = set(latent_states.keys())
        all_observed_ids = set(observed_counts.keys())
        relevant_ids = all_latent_ids | all_observed_ids

        for s_id in relevant_ids:
            prob = latent_states.get(s_id, 0.0)
            count = observed_counts.get(s_id, 0.0)

            # Expected count lambda = N * P(s) * efficiency + noise
            lambda_s = (total_observed_units * prob * self.detection_efficiency) \
                + self.background_noise

            term = count * np.log(lambda_s) - lambda_s - gammaln(count + 1)
            log_lik += term

        return log_lik

    def get_constraint_diagnostics(
        self,
        latent_states: dict[int, float],
        founder_threshold: int = 5,
    ) -> dict[str, Any]:
        """
        Generic diagnostic extractor for constraints.
        """
        if not self.graph:
            return {}

        diagnostics = {
            "total_reuse": 0.0,
            "founder_reuse": 0.0,
            "derived_reuse": 0.0,
            "mean_depth": 0.0,
        }

        for s_id, prob in latent_states.items():
            state = self.graph.get_state(s_id)
            if not state:
                continue

            reuse = getattr(state, "reuse_count", 0)
            depth = getattr(state, "assembly_depth", 0)

            diagnostics["total_reuse"] += reuse * prob
            diagnostics["mean_depth"] += depth * prob

            if depth <= founder_threshold:
                diagnostics["founder_reuse"] += reuse * prob
            else:
                diagnostics["derived_reuse"] += reuse * prob

        return diagnostics

    def __str__(self) -> str:
        return f"AssemblyCountsModel(efficiency={self.detection_efficiency}, " \
               f"noise={self.background_noise})"

