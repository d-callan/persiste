"""Observation model abstractions for scoring transitions against baselines."""

from typing import Any, Optional, TYPE_CHECKING, Tuple, Dict
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
    
    def log_likelihood_rate_gradient(
        self,
        data: "ObservedTransitions",
        baseline: "Baseline",
        graph: "TransitionGraph",
    ) -> Optional[dict[tuple[int, int], float]]:
        """
        Optional gradient hook w.r.t. effective rates λ_ij.
        
        Returns mapping {(i, j): dℓ/dλ_ij}. Default implementation
        returns None, signaling no analytic gradient is available.
        """
        return None
    
    def log_likelihood_rate_hessian(
        self,
        data: "ObservedTransitions",
        baseline: "Baseline",
        graph: "TransitionGraph",
    ) -> Optional[dict[tuple[int, int], float]]:
        """
        Optional diagonal Hessian hook w.r.t. effective rates λ_ij.
        
        Returns mapping {(i, j): d²ℓ/dλ²_ij}. Default implementation
        returns None, signaling no analytic Hessian is available.
        """
        return None


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
    
    def log_likelihood_rate_gradient(
        self,
        data: "ObservedTransitions",
        baseline: "Baseline",
        graph: "TransitionGraph",
    ) -> Optional[dict[tuple[int, int], float]]:
        gradients: dict[tuple[int, int], float] = {}
        
        for (i, j), count in data.counts.items():
            if not graph.allows(i, j):
                continue
            
            λ = baseline.get_rate(i, j)
            if data.exposure is not None:
                λ *= data.exposure
            
            if λ <= 0:
                continue
            
            gradients[(i, j)] = count / λ - 1.0
        
        return gradients
    
    def log_likelihood_rate_hessian(
        self,
        data: "ObservedTransitions",
        baseline: "Baseline",
        graph: "TransitionGraph",
    ) -> Optional[dict[tuple[int, int], float]]:
        hessians: dict[tuple[int, int], float] = {}
        
        for (i, j), count in data.counts.items():
            if not graph.allows(i, j):
                continue
            
            λ = baseline.get_rate(i, j)
            if data.exposure is not None:
                λ *= data.exposure
            
            if λ <= 0:
                continue
            
            hessians[(i, j)] = -count / (λ ** 2)
        
        return hessians


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
        """
        Compute CTMC log-likelihood treating per-edge transitions as Poisson counts.
        
        This mirrors the PoissonObservationModel but requires a CTMC rate source
        (either the provided rate_matrix or the constrained baseline).
        """
        ll = 0.0
        for (i, j), count in data.counts.items():
            if not graph.allows(i, j):
                continue
            
            λ = self.rate(i, j) if self.rate_matrix is not None else baseline.get_rate(i, j)
            
            if data.exposure is not None:
                λ *= data.exposure
            
            if λ <= 0:
                if count > 0:
                    return float("-inf")
                continue
            
            ll += count * math.log(λ) - λ - math.lgamma(count + 1)
        
        return ll


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
        """
        Compute multinomial log-likelihood grouped by source state.
        
        For each origin i, counts leaving i are assumed to follow a multinomial
        distribution with probabilities either supplied or derived from baseline rates.
        """
        from collections import defaultdict
        
        # Precompute transition probabilities
        prob_map: dict[tuple[int, int], float] = {}
        row_prob_sums: defaultdict[int, float] = defaultdict(float)
        
        if self.probabilities is not None:
            for (i, j), count in data.counts.items():
                if not graph.allows(i, j):
                    continue
                p = float(self.probabilities[i, j])
                prob_map[(i, j)] = p
        else:
            row_rate_sums: defaultdict[int, float] = defaultdict(float)
            rate_map: dict[tuple[int, int], float] = {}
            for (i, j), _ in data.counts.items():
                if not graph.allows(i, j):
                    continue
                rate = baseline.get_rate(i, j)
                rate_map[(i, j)] = rate
                row_rate_sums[i] += rate
            for (i, j), rate in rate_map.items():
                row_sum = row_rate_sums[i]
                if row_sum <= 0.0:
                    prob_map[(i, j)] = 0.0
                else:
                    prob_map[(i, j)] = rate / row_sum
        
        # Accumulate log-likelihood per source row
        row_counts: defaultdict[int, float] = defaultdict(float)
        ll = 0.0
        
        for (i, j), count in data.counts.items():
            if not graph.allows(i, j):
                continue
            p = prob_map.get((i, j), 0.0)
            row_counts[i] += count
            if p <= 0.0:
                if count > 0:
                    return float("-inf")
                continue
            ll += count * math.log(p)
        
        for i, total in row_counts.items():
            ll += math.lgamma(total + 1)
            for (src, dst), count in data.counts.items():
                if src == i and graph.allows(src, dst):
                    ll -= math.lgamma(count + 1)
        
        return ll
