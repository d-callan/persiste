"""
Observation models for copy number data.

Maps observed copy numbers to state likelihoods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

from persiste.plugins.copynumber.states.cn_states import CopyNumberState


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
    
    def get_tip_likelihood(self, observed_state: int) -> np.ndarray:
        """
        Get deterministic tip likelihood.
        
        Args:
            observed_state: Observed binned state (0-3)
        
        Returns:
            (4,) one-hot vector
        """
        if not 0 <= observed_state <= 3:
            raise ValueError(f"observed_state must be 0-3, got {observed_state}")
        
        likelihood = np.zeros(4)
        likelihood[observed_state] = 1.0
        
        return likelihood
    
    def get_tip_likelihoods_matrix(self, observed_states: np.ndarray) -> np.ndarray:
        """
        Get tip likelihoods for multiple observations.
        
        Args:
            observed_states: (n_taxa,) array of observed states
        
        Returns:
            (n_taxa, 4) matrix of tip likelihoods
        """
        n_taxa = len(observed_states)
        likelihoods = np.zeros((n_taxa, 4))
        
        for i, state in enumerate(observed_states):
            likelihoods[i] = self.get_tip_likelihood(state)
        
        return likelihoods


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
