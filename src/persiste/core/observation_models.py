"""Observation model abstractions for scoring transitions against baselines."""

from typing import Any, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

if TYPE_CHECKING:
    from persiste.core.transitions import TransitionGraph
    from persiste.core.baseline import Baseline
    from persiste.core.data import ObservedTransitions


class ObservationModel(ABC):
    """
    Abstract base class for observation models.
    
    Defines how to score observed transitions against baseline rates.
    Separate from:
    - TransitionGraph (structure: what's allowed)
    - Baseline (rates: λ_ij if unconstrained)
    
    This is where statistical models live (Poisson, CTMC, Multinomial).
    """
    
    @abstractmethod
    def rate(self, i: int, j: int) -> float:
        """
        Get baseline rate for transition i -> j.
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            Baseline transition rate λ_ij
        """
        pass
    
    @abstractmethod
    def log_likelihood(
        self,
        data: "ObservedTransitions",
        baseline: "Baseline",
        graph: "TransitionGraph",
    ) -> float:
        """
        Compute log-likelihood of observed transitions.
        
        Args:
            data: Observed transition counts/events
            baseline: Baseline rates (λ_ij counterfactual)
            graph: Transition structure (what's allowed)
            
        Returns:
            Log-likelihood value
        """
        pass


@dataclass
class PoissonObservationModel(ObservationModel):
    """
    Poisson baseline for transition events.
    
    Models transitions as independent Poisson processes.
    Good default for:
    - Production events (chemistry, assembly)
    - Memoryless processes
    - Regime-agnostic baselines
    
    Attributes:
        graph: Transition graph structure
        base_rate: Uniform baseline rate (can be overridden per-transition)
        rate_fn: Optional function(i, j) -> rate for non-uniform rates
    """
    
    graph: "TransitionGraph"
    base_rate: float = 1.0
    rate_fn: Optional[Any] = None
    
    def rate(self, i: int, j: int) -> float:
        """
        Get Poisson rate for transition i -> j.
        
        Returns 0 if transition not allowed by graph structure.
        """
        if not self.graph.allows(i, j):
            return 0.0
        
        if self.rate_fn is not None:
            return self.rate_fn(i, j)
        
        return self.base_rate
    
    def log_likelihood(
        self,
        data: "ObservedTransitions",
        baseline: "Baseline",
        graph: "TransitionGraph",
    ) -> float:
        """
        Compute Poisson log-likelihood.
        
        For each allowed transition (i, j):
        - Get baseline rate λ_ij from baseline
        - Optionally scale by exposure
        - Score observed count against Poisson(λ_ij)
        
        Args:
            data: Observed transition counts
            baseline: Baseline rates
            graph: Transition structure
            
        Returns:
            Log-likelihood under Poisson model
        """
        ll = 0.0
        
        for (i, j), count in data.counts.items():
            # Only score allowed transitions
            if not graph.allows(i, j):
                continue
            
            # Get baseline rate
            λ = baseline.get_rate(i, j)
            
            # Scale by exposure if provided
            if data.exposure is not None:
                λ *= data.exposure
            
            # Poisson log-likelihood: k*log(λ) - λ - log(k!)
            if λ <= 0:
                # If baseline rate is 0 but we observed transitions, likelihood is 0
                if count > 0:
                    return float('-inf')
                continue
            
            ll += count * math.log(λ) - λ - math.lgamma(count + 1)
        
        return ll


@dataclass
class CTMCObservationModel(ObservationModel):
    """
    Continuous-time Markov chain baseline.
    
    For processes with:
    - Exponentially distributed waiting times
    - Memoryless transitions
    - Rate matrix parameterization
    
    Attributes:
        graph: Transition graph structure
        rate_matrix: Transition rate matrix (optional)
    """
    
    graph: "TransitionGraph"
    rate_matrix: Optional[Any] = None
    
    def rate(self, i: int, j: int) -> float:
        """Get CTMC rate for transition i -> j."""
        if not self.graph.allows(i, j):
            return 0.0
        
        if self.rate_matrix is not None:
            return float(self.rate_matrix[i, j])
        
        raise ValueError("rate_matrix must be provided for CTMC model")
    
    def log_likelihood(
        self,
        data: "ObservedTransitions",
        baseline: "Baseline",
        graph: "TransitionGraph",
    ) -> float:
        """Compute CTMC log-likelihood."""
        # TODO: Implement CTMC likelihood
        raise NotImplementedError("CTMC likelihood not yet implemented")


@dataclass
class MultinomialObservationModel(ObservationModel):
    """
    Multinomial baseline for competing transitions.
    
    For processes where:
    - Multiple transitions compete
    - Total events fixed
    - Compositional constraints
    
    Attributes:
        graph: Transition graph structure
        probabilities: Transition probability matrix
    """
    
    graph: "TransitionGraph"
    probabilities: Optional[Any] = None
    
    def rate(self, i: int, j: int) -> float:
        """
        Get multinomial probability for transition i -> j.
        
        Note: Returns probability, not rate (different semantics).
        """
        if not self.graph.allows(i, j):
            return 0.0
        
        if self.probabilities is not None:
            return float(self.probabilities[i, j])
        
        raise ValueError("probabilities must be provided for Multinomial model")
    
    def log_likelihood(
        self,
        data: "ObservedTransitions",
        baseline: "Baseline",
        graph: "TransitionGraph",
    ) -> float:
        """Compute multinomial log-likelihood."""
        # TODO: Implement multinomial likelihood
        raise NotImplementedError("Multinomial likelihood not yet implemented")
