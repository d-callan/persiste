"""
Presence-only observation model.

Observes presence/absence of compounds, not full state transitions.
Tolerates massive missingness - only needs to explain what we see.
"""

from typing import Any

import numpy as np
from scipy.special import gammaln

from persiste.core.data import ObservedTransitions
from persiste.core.observation_models import ObservationModel
from persiste.plugins.assembly.states.assembly_state import AssemblyState


class PresenceObservationModel(ObservationModel):
    """
    Presence-only observation model for assembly theory.

    Key principle: You never observe full state transitions.
    You observe presence/absence of compounds.

    P(observed | latent state distribution)

    This model tolerates massive missingness - we only need to explain
    what we observe, not what we don't observe.

    Attributes:
        detection_prob: Probability of detecting a present compound (default: 0.9)
        false_positive_prob: Probability of false detection (default: 0.01)
    """

    def __init__(
        self,
        graph: Any = None,
        detection_prob: float = 0.9,
        false_positive_prob: float = 0.01,
    ):
        """
        Initialize presence observation model.

        Args:
            graph: TransitionGraph (unused, for interface compatibility)
            detection_prob: P(detect | present) (default: 0.9)
            false_positive_prob: P(detect | absent) (default: 0.01)
        """
        self.graph = graph
        self.detection_prob = detection_prob
        self.false_positive_prob = false_positive_prob

    def rate(self, i: int, j: int) -> float:
        """
        Get observation rate (unused for presence model).

        Required by ObservationModel interface.
        """
        return 0.0

    def log_likelihood(
        self,
        data: ObservedTransitions,
        baseline: Any,
        graph: Any,
    ) -> float:
        """
        Compute log-likelihood of observations.

        For presence model, data should contain:
        - observed_compounds: Set of observed compound identifiers
        - latent_states: Dict[AssemblyState, float] - state probabilities

        Args:
            data: ObservedTransitions with observed_compounds and latent_states
            baseline: Baseline model (unused)
            graph: TransitionGraph (unused)

        Returns:
            Log-likelihood of observations given latent state distribution
        """
        # Extract observed compounds and latent states from data
        if not hasattr(data, "observed_compounds") or not hasattr(data, "latent_states"):
            raise ValueError(
                "PresenceObservationModel requires data with "
                "'observed_compounds' and 'latent_states' attributes"
            )

        observed_compounds = getattr(data, "observed_compounds")
        latent_states = getattr(data, "latent_states")

        return self.compute_log_likelihood(observed_compounds, latent_states)

    def compute_log_likelihood(
        self,
        observed_compounds: set[str],
        latent_states: dict[AssemblyState, float],
    ) -> float:
        """
        Compute likelihood of observations given latent state distribution.

        For each observed compound:
            P(observe) = sum over states containing compound of P(state) * P(detect | present)

        Args:
            observed_compounds: Set of observed compound identifiers
            latent_states: Dict mapping states to probabilities

        Returns:
            Log-likelihood
        """
        log_lik = 0.0

        # For each observed compound
        for compound in observed_compounds:
            # Probability that compound is present in latent states
            p_present = sum(
                prob for state, prob in latent_states.items() if state.contains_part(compound)
            )

            # P(observe | present) * P(present) + P(observe | absent) * P(absent)
            p_observe = self.detection_prob * p_present + self.false_positive_prob * (1 - p_present)

            # Add to log-likelihood
            log_lik += np.log(p_observe + 1e-10)  # Avoid log(0)

        return log_lik

    def predict_presence(
        self,
        latent_states: dict[AssemblyState, float],
        compound: str,
    ) -> float:
        """
        Predict probability of observing a compound.

        Args:
            latent_states: Dict mapping states to probabilities
            compound: Compound identifier

        Returns:
            P(observe compound)
        """
        p_present = sum(
            prob for state, prob in latent_states.items() if state.contains_part(compound)
        )

        return self.detection_prob * p_present + self.false_positive_prob * (1 - p_present)

    def __str__(self) -> str:
        return (
            f"PresenceObservationModel("
            f"detect={self.detection_prob:.2f}, "
            f"false_pos={self.false_positive_prob:.3f})"
        )


class FrequencyWeightedPresenceModel(ObservationModel):
    """
    Frequency-weighted presence observation model (Option A - CHEAP, HIGH VALUE).

    Instead of: obs = {A, B, C}
    Use:        obs = {A: 12, B: 3, C: 1}

    Interpretation:
    - Count of detection events
    - Number of samples where present
    - Coarse abundance bins

    Why this helps:
    - Breaks symmetry between θ that affect rates vs reachability
    - Still tolerant to missingness
    - Still realistic for chemistry/metagenomics

    This alone often fixes identifiability.

    Attributes:
        detection_prob: P(detect | present)
        false_positive_rate: Expected false positive count
    """

    def __init__(
        self,
        graph: Any = None,
        detection_prob: float = 0.9,
        false_positive_rate: float = 0.1,
    ):
        """
        Initialize frequency-weighted presence model.

        Args:
            graph: TransitionGraph (unused, for interface compatibility)
            detection_prob: P(detect | present) per sample
            false_positive_rate: Expected false positive count
        """
        self.graph = graph
        self.detection_prob = detection_prob
        self.false_positive_rate = false_positive_rate

    def rate(self, i: int, j: int) -> float:
        """Get observation rate (unused)."""
        return 0.0

    def log_likelihood(
        self,
        data: ObservedTransitions,
        baseline: Any,
        graph: Any,
    ) -> float:
        """
        Compute log-likelihood of frequency-weighted observations.

        Args:
            data: ObservedTransitions with observed_counts and latent_states
            baseline: Baseline model (unused)
            graph: TransitionGraph (unused)

        Returns:
            Log-likelihood
        """
        if not hasattr(data, "observed_counts") or not hasattr(data, "latent_states"):
            raise ValueError(
                "FrequencyWeightedPresenceModel requires data with "
                "'observed_counts' and 'latent_states' attributes"
            )

        observed_counts = getattr(data, "observed_counts")
        latent_states = getattr(data, "latent_states")

        return self.compute_log_likelihood(observed_counts, latent_states)

    def compute_log_likelihood(
        self,
        observed_counts: dict[str, int],
        latent_states: dict[AssemblyState, float],
    ) -> float:
        """
        Compute likelihood of frequency-weighted observations.

        Model: For each compound c:
        - P(c present in state s) = 1 if c in s.parts, else 0
        - P(c present overall) = sum_s P(s) * P(c in s)
        - Count ~ Poisson(λ_c) where λ_c = n_samples * P(c present) * detection_prob

        Args:
            observed_counts: Dict of compound -> count
            latent_states: Dict of state -> probability

        Returns:
            Log-likelihood
        """
        if not latent_states:
            return -np.inf

        # Compute marginal presence probability for each compound
        compound_probs = {}
        all_compounds = set()

        for state, prob in latent_states.items():
            for part in state.get_parts_list():
                all_compounds.add(part)
                compound_probs[part] = compound_probs.get(part, 0.0) + prob

        # Total number of samples (inferred from max count)
        if observed_counts:
            n_samples = sum(observed_counts.values())
        else:
            n_samples = 1

        log_lik = 0.0

        # Likelihood for observed compounds
        for compound, count in observed_counts.items():
            # Expected count = n_samples * P(present) * detection_prob + false_positive_rate
            p_present = compound_probs.get(compound, 0.0)
            lambda_c = n_samples * p_present * self.detection_prob + self.false_positive_rate

            # Poisson likelihood: P(count | lambda) = lambda^count * exp(-lambda) / count!
            # log P = count * log(lambda) - lambda - log(count!)
            # Use scipy.special.gammaln for log(count!) = log(gamma(count+1))
            if lambda_c > 0:
                log_lik += count * np.log(lambda_c) - lambda_c - gammaln(count + 1)
            else:
                # If lambda=0 but we observed it, very unlikely (use small lambda)
                lambda_c = 1e-6
                log_lik += count * np.log(lambda_c) - lambda_c - gammaln(count + 1)

        # Likelihood for unobserved compounds (count=0)
        # Only penalize if they should be present
        for compound in all_compounds:
            if compound not in observed_counts:
                p_present = compound_probs.get(compound, 0.0)
                lambda_c = n_samples * p_present * self.detection_prob + self.false_positive_rate
                # P(count=0 | lambda) = exp(-lambda)
                log_lik += -lambda_c

        return log_lik

    def __str__(self) -> str:
        return (
            "FrequencyWeightedPresenceModel("
            f"detect={self.detection_prob:.2f}, "
            f"false_pos_rate={self.false_positive_rate:.2f})"
        )

    def predict_presence(
        self,
        latent_states: dict[AssemblyState, float],
        compound: str,
    ) -> float:
        """
        Predict probability of observing a compound at least once.
        """
        p_present = sum(
            prob for state, prob in latent_states.items() if state.contains_part(compound)
        )
        return self.detection_prob * p_present + self.false_positive_rate * (1 - p_present)


class FragmentObservationModel(ObservationModel):
    """
    Fragment-based observation model.

    Observes fragment distributions (e.g., from mass spectrometry).

    P(fragments | latent assembly state)

    Each state can produce multiple fragments with different intensities.
    """

    def __init__(
        self,
        graph: Any = None,
        noise_level: float = 0.1,
    ):
        """
        Initialize fragment observation model.

        Args:
            graph: TransitionGraph (unused, for interface compatibility)
            noise_level: Noise standard deviation for fragment intensities
        """
        self.graph = graph
        self.noise_level = noise_level

    def rate(self, i: int, j: int) -> float:
        """Get observation rate (unused)."""
        return 0.0

    def log_likelihood(
        self,
        data: ObservedTransitions,
        baseline: Any,
        graph: Any,
        parameters: dict[str, float] | None = None,
    ) -> float:
        """
        Compute log-likelihood of fragment observations.

        Data should contain:
        - observed_fragments: Dict[str, float] - fragment -> intensity
        - latent_states: Dict[AssemblyState, float] - state probabilities
        """
        if not hasattr(data, "observed_fragments") or not hasattr(data, "latent_states"):
            raise ValueError(
                "FragmentObservationModel requires data with "
                "'observed_fragments' and 'latent_states' attributes"
            )

        observed_fragments = data.observed_fragments
        latent_states = data.latent_states

        return self.compute_log_likelihood(observed_fragments, latent_states)

    def compute_log_likelihood(
        self,
        observed_fragments: dict[str, float],
        latent_states: dict[AssemblyState, float],
    ) -> float:
        """
        Compute likelihood of fragment observations.

        Predicts expected fragment intensities from latent states,
        then compares to observations using Gaussian likelihood.

        Args:
            observed_fragments: Dict of fragment -> observed intensity
            latent_states: Dict of state -> probability

        Returns:
            Log-likelihood
        """
        # Predict fragments from latent states
        expected_fragments = self._predict_fragments(latent_states)

        # Gaussian likelihood for each fragment
        log_lik = 0.0
        for fragment, observed_intensity in observed_fragments.items():
            expected_intensity = expected_fragments.get(fragment, 0.0)

            # Gaussian: -0.5 * ((obs - exp) / noise)^2
            residual = (observed_intensity - expected_intensity) / self.noise_level
            log_lik += -0.5 * (residual**2)

        return log_lik

    def _predict_fragments(
        self,
        latent_states: dict[AssemblyState, float],
    ) -> dict[str, float]:
        """
        Predict fragment intensities from latent states.

        Simple model: each state produces fragments corresponding to its parts.
        Intensity proportional to state probability.
        """
        fragments = {}

        for state, prob in latent_states.items():
            # Each part in the state can be a fragment
            for part, count in state.get_parts_dict().items():
                fragments[part] = fragments.get(part, 0.0) + prob * count

        return fragments

    def __str__(self) -> str:
        return f"FragmentObservationModel(noise={self.noise_level:.3f})"
