from __future__ import annotations

"""
Observation models for copy number data.

Maps observed copy numbers to state likelihoods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.linalg import expm

from persiste.core.observation_models import ObservationModel
from persiste.core.pruning import FelsensteinPruning
from persiste.core.tip_observations import OneHotTipObservation
from persiste.plugins.copynumber.states.cn_states import CopyNumberState
from persiste.core.data import ObservedTransitions
from persiste.core.baseline import Baseline
from persiste.core.transitions import TransitionGraph


def _stationary_distribution(rate_matrix: np.ndarray) -> np.ndarray:
    """Compute stationary distribution for an irreducible CTMC."""
    eigvals, eigvecs = np.linalg.eig(rate_matrix.T)
    idx = int(np.argmin(np.abs(eigvals)))
    stationary = np.real(eigvecs[:, idx])
    stationary = np.clip(stationary, 0.0, None)
    total = stationary.sum()
    if total <= 0:
        return np.full(rate_matrix.shape[0], 1.0 / rate_matrix.shape[0])
    return stationary / total


class MatrixExponentialTransitionProvider:
    """Transition provider that exponentiates a fixed rate matrix."""

    def __init__(self, rate_matrix: np.ndarray):
        if rate_matrix.shape[0] != rate_matrix.shape[1]:
            raise ValueError("rate_matrix must be square")
        self.rate_matrix = rate_matrix
        self._n_states = rate_matrix.shape[0]
        self.equilibrium_frequencies = _stationary_distribution(rate_matrix)
        self._cache: dict[float, np.ndarray] = {}

    @property
    def n_states(self) -> int:
        return self._n_states

    def get_transition_matrix(self, branch_length: float) -> np.ndarray:
        if branch_length <= 0:
            return np.eye(self._n_states)
        cached = self._cache.get(branch_length)
        if cached is not None:
            return cached
        matrix = expm(self.rate_matrix * branch_length)
        self._cache[branch_length] = matrix
        return matrix


class CopyNumberTipConditionalProvider:
    """Tip conditional provider that uses CopyNumberObservation helpers."""

    def __init__(
        self,
        data: np.ndarray,
        taxon_names: list[str],
        obs_model: "CopyNumberObservation",
        n_states: int,
    ):
        if data.shape[0] != len(taxon_names):
            raise ValueError(
                "Observation matrix rows must match number of taxon names "
                f"({data.shape[0]} != {len(taxon_names)})"
            )
        self.data = data
        self.taxon_names = taxon_names
        self.obs_model = obs_model
        self.n_states = n_states
        self.taxon_to_idx = {name: idx for idx, name in enumerate(taxon_names)}

    def get_tip_conditional(self, tip_name: str, site_idx: int) -> np.ndarray:
        tip_idx = self.taxon_to_idx.get(tip_name)
        if tip_idx is None:
            return np.full(self.n_states, 1.0 / self.n_states)

        observed_state = int(self.data[tip_idx, site_idx])
        if observed_state < 0 or observed_state >= self.n_states:
            return np.full(self.n_states, 1.0 / self.n_states)

        likelihood = self.obs_model.get_tip_likelihood(observed_state)
        return np.asarray(likelihood, dtype=float)


@dataclass
class CopyNumberObservation(ABC):
    """
    Abstract base class for copy number observation models.
    
    Observation models define how observed copy numbers map to
    state likelihoods at tree tips.
    """
    
    @abstractmethod
    def get_tip_likelihood(self, observed_state: int) -> np.ndarray:
        """
        Get likelihood vector for observed state at tip.
        
        Args:
            observed_state: Observed binned state (0-3)
        
        Returns:
            (4,) array of likelihoods for each state
        """
        pass


@dataclass
class DeterministicBinObservation(CopyNumberObservation):
    """
    Deterministic bin observation model (v1 default).
    
    Simple, honest observation model:
        - Observed bin = true state (deterministic)
        - No measurement uncertainty
        - No ambiguity
    
    Tip likelihood:
        L[i] = 1.0 if i == observed_state else 0.0
    
    Design rationale:
        - Simple and interpretable
        - No hidden parameters
        - Appropriate for high-quality CN calls
        - Can be extended in v2+ for noisy data
    
    Example:
        >>> obs_model = DeterministicBinObservation()
        >>> # Observed state is SINGLE (1)
        >>> likelihood = obs_model.get_tip_likelihood(1)
        >>> # likelihood = [0, 1, 0, 0]
    """
    n_states: int = field(default_factory=CopyNumberState.n_states)
    _tip_model: OneHotTipObservation = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tip_model = OneHotTipObservation(self.n_states)

    def get_tip_likelihood(self, observed_state: int) -> np.ndarray:
        return self._tip_model.get_tip_likelihood(observed_state)

    def get_tip_likelihoods_matrix(self, observed_states: np.ndarray) -> np.ndarray:
        return self._tip_model.get_tip_likelihoods_matrix(observed_states)


@dataclass
class UncertainBinObservation(CopyNumberObservation):
    """
    Uncertain bin observation model (v2+, future).
    
    Allows for measurement uncertainty:
        - Noisy CN calls
        - Ambiguous bins
        - Probabilistic assignment
    
    Tip likelihood:
        L[i] = P(observed | true_state = i)
    
    Parameters:
        error_rate: Probability of misclassification
        adjacent_only: If True, errors only to adjacent bins
    
    Design note:
        Not implemented in v1. Placeholder for future extension.
    """
    error_rate: float = 0.05
    adjacent_only: bool = True
    
    def get_tip_likelihood(self, observed_state: int) -> np.ndarray:
        """
        Get uncertain tip likelihood.
        
        Allows for observation error to adjacent states.
        """
        if not 0 <= observed_state <= 3:
            raise ValueError(f"observed_state must be 0-3, got {observed_state}")
        
        likelihood = np.zeros(4)
        
        # Correct state gets (1 - error_rate)
        likelihood[observed_state] = 1.0 - self.error_rate
        
        if self.adjacent_only:
            # Distribute error to adjacent states
            adjacent = []
            if observed_state > 0:
                adjacent.append(observed_state - 1)
            if observed_state < 3:
                adjacent.append(observed_state + 1)
            
            if adjacent:
                error_per_adjacent = self.error_rate / len(adjacent)
                for adj in adjacent:
                    likelihood[adj] = error_per_adjacent
        else:
            # Distribute error uniformly to all other states
            error_per_state = self.error_rate / 3
            for i in range(4):
                if i != observed_state:
                    likelihood[i] = error_per_state
        
        return likelihood


class CopyNumberObservationModel(ObservationModel):
    """
    ObservationModel that scores copy-number state data using tree likelihoods.
    
    This adapter reuses the deterministic/uncertain tip observation helpers and plugs
    them into Felsenstein pruning so the core ConstraintInference pipeline can call
    log_likelihood(data, baseline, graph).
    """

    def __init__(
        self,
        graph: Optional[TransitionGraph],
        tree: Any,
        observed_matrix: np.ndarray,
        obs_model: CopyNumberObservation,
        use_jax: bool = False,
    ):
        if tree is None:
            raise ValueError("CopyNumberObservationModel requires a TreeStructure")
        if not hasattr(tree, "tip_names"):
            raise ValueError("tree must expose tip_names for observation alignment")

        self.graph = graph
        self.tree = tree
        self.obs_model = obs_model
        self.use_jax = use_jax
        self.n_states = CopyNumberState.n_states()
        self.taxon_names = list(tree.tip_names)
        self.observed_matrix = self._coerce_matrix(observed_matrix, self.taxon_names)
        self._n_families = self.observed_matrix.shape[1]
        self._pruning = FelsensteinPruning(tree, n_states=self.n_states, use_jax=use_jax)
        self._last_per_family_lls: np.ndarray = np.zeros(self._n_families, dtype=float)

    @staticmethod
    def _coerce_matrix(observed_matrix: np.ndarray, taxon_names: list[str]) -> np.ndarray:
        matrix = np.asarray(observed_matrix, dtype=int)
        if matrix.ndim != 2:
            raise ValueError("observed_matrix must be 2-dimensional (taxa × families)")

        n_taxa = len(taxon_names)
        if matrix.shape[0] == n_taxa:
            return matrix
        if matrix.shape[1] == n_taxa:
            return matrix.T

        raise ValueError(
            "observed_matrix must align with taxon order either along rows or columns"
        )

    def rate(self, i: int, j: int) -> float:
        """Unused; copy-number likelihood is tree-based."""
        return 0.0

    def log_likelihood(
        self,
        data: ObservedTransitions,
        baseline: Baseline,
        graph: TransitionGraph,
    ) -> float:
        """
        Compute log-likelihood of copy-number observations using baseline rates per family.
        
        Args:
            data: ObservedTransitions with metadata storing per-family rate parameters.
            baseline: Baseline providing transition rates (must expose rate matrices).
            graph: TransitionGraph (unused for tree-based evaluation).
        """
        if not hasattr(baseline, "build_rate_matrix"):
            raise TypeError(
                "CopyNumberObservationModel baseline must provide build_rate_matrix()"
            )

        per_family_lls = np.zeros(self._n_families, dtype=float)
        for fam_idx in range(self._n_families):
            rate_matrix = baseline.build_rate_matrix(family_idx=fam_idx)
            transition_provider = MatrixExponentialTransitionProvider(rate_matrix)
            tip_provider = CopyNumberTipConditionalProvider(
                data=self.observed_matrix[:, fam_idx : fam_idx + 1],
                taxon_names=self.taxon_names,
                obs_model=self.obs_model,
                n_states=self.n_states,
            )
            result = self._pruning.compute_likelihood(
                transition_provider=transition_provider,
                tip_provider=tip_provider,
                n_sites=1,
            )
            per_family_lls[fam_idx] = float(result.log_likelihood)

        self._last_per_family_lls = per_family_lls
        return float(per_family_lls.sum())

    @property
    def last_per_family_log_likelihoods(self) -> np.ndarray:
        """Return per-family log-likelihoods from the most recent evaluation."""
        return self._last_per_family_lls.copy()


def create_observation_model(
    model_type: str = 'deterministic',
    **kwargs
) -> CopyNumberObservation:
    """
    Factory function for creating observation models.
    
    Args:
        model_type: Type of observation model
            - 'deterministic' (default, v1)
            - 'uncertain' (v2+)
        **kwargs: Parameters for the observation model
    
    Returns:
        Observation model instance
    
    Example:
        >>> obs_model = create_observation_model('deterministic')
        >>> obs_model = create_observation_model('uncertain', error_rate=0.1)
    """
    if model_type == 'deterministic':
        return DeterministicBinObservation(**kwargs)
    elif model_type == 'uncertain':
        return UncertainBinObservation(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
