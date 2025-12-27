"""Baseline rate specifications."""

from typing import Callable, Optional, Union, Any
from dataclasses import dataclass
import numpy as np

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jnp = None  # type: ignore


@dataclass
class Baseline:
    """
    Baseline rate specification.
    
    Defines expected transition rates λ_ij *conditional on transitions being allowed*.
    
    This is a counterfactual: "What would happen if nothing resisted change?"
    
    Does NOT define:
    - Which transitions are allowed (see TransitionGraph)
    - How rates convert to likelihoods (see ObservationModel)
    - Constraint parameters (see ConstraintModel)
    
    Attributes:
        rate_fn: Function(state_i, state_j) -> rate
        rate_matrix: Pre-computed rate matrix (optional)
    """
    
    rate_fn: Optional[Callable[[int, int], float]] = None
    rate_matrix: Optional["Any"] = None
    
    def __post_init__(self):
        """Validate that at least one of rate_fn or rate_matrix is provided."""
        if self.rate_fn is None and self.rate_matrix is None:
            raise ValueError("Must provide either rate_fn or rate_matrix")
    
    @classmethod
    def from_matrix(cls, matrix: Union[np.ndarray, Any]) -> "Baseline":
        """
        Create baseline from pre-computed rate matrix.
        
        Args:
            matrix: (n_states, n_states) rate matrix
            
        Returns:
            Baseline object
        """
        if jnp is None:
            return cls(rate_matrix=np.asarray(matrix))
        return cls(rate_matrix=jnp.asarray(matrix))
    
    @classmethod
    def uniform(cls, rate: float = 1.0) -> "Baseline":
        """
        Create uniform baseline (all transitions have same rate).
        
        Args:
            rate: Uniform transition rate
            
        Returns:
            Baseline object
        """
        return cls(rate_fn=lambda i, j: rate)
    
    @classmethod
    def empirical(cls, data: "Any", graph: "Any" = None) -> "Baseline":
        """
        Estimate baseline from empirical data.
        
        ⚠️  WARNING: This collapses production and observation layers.
        
        Philosophically, this treats observed rates as the baseline,
        which makes constraint detection circular. Useful as a null
        model or for exploratory analysis, but NOT for inferring constraint.
        
        Prefer: define baseline from theory/chemistry/physics, then
        compare against observations to detect constraint.
        
        Args:
            data: ObservedTransitions with counts and exposure
            graph: Optional TransitionGraph to filter allowed transitions
            
        Returns:
            Baseline with empirical rates (count / exposure)
        """
        from persiste.core.data import ObservedTransitions
        
        if not isinstance(data, ObservedTransitions):
            raise TypeError("data must be ObservedTransitions instance")
        
        if data.exposure is None or data.exposure == 0:
            raise ValueError("exposure must be provided and non-zero for empirical baseline")
        
        # Estimate λ_ij = observed_count / exposure
        rates = {}
        for (i, j), count in data.counts.items():
            if graph is None or graph.allows(i, j):
                rates[(i, j)] = count / data.exposure
        
        return cls(rate_fn=lambda i, j: rates.get((i, j), 0.0))
    
    def get_rate(self, i: int, j: int) -> float:
        """
        Get baseline rate for transition i -> j.
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            Baseline rate λ_ij
        """
        if self.rate_matrix is not None:
            return float(self.rate_matrix[i, j])
        elif self.rate_fn is not None:
            return self.rate_fn(i, j)
        else:
            raise RuntimeError("No rate source available")
    
    def to_matrix(self, n_states: int) -> Any:
        """
        Convert to full rate matrix.
        
        Args:
            n_states: Number of states
            
        Returns:
            (n_states, n_states) rate matrix
        """
        if self.rate_matrix is not None:
            return self.rate_matrix
        
        # Build from rate_fn
        if jnp is None:
            matrix = np.zeros((n_states, n_states), dtype=float)
            for i in range(n_states):
                for j in range(n_states):
                    if i != j:
                        matrix[i, j] = self.get_rate(i, j)
            return matrix

        matrix = jnp.zeros((n_states, n_states))
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    matrix = matrix.at[i, j].set(self.get_rate(i, j))
        
        return matrix
