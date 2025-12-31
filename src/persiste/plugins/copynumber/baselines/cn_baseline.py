"""
Baseline rate models for copy number dynamics.

Baseline rates control the "default" behavior:
    - Gain (0→1)
    - Loss (1→0)
    - Amplification (1→2→3)
    - Contraction (3→2→1)

Constraints then modulate these baseline rates.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

from persiste.plugins.copynumber.states.cn_states import (
    CopyNumberState,
    get_sparse_transition_graph,
)


@dataclass
class CopyNumberBaseline(ABC):
    """
    Abstract base class for copy number baseline models.
    
    Baseline models define the "default" transition rates before
    constraints are applied.
    """
    
    @abstractmethod
    def get_baseline_rates(self, family_idx: Optional[int] = None) -> Dict[Tuple[int, int], float]:
        """
        Get baseline rates for transitions.
        
        Args:
            family_idx: Optional family index for hierarchical models
        
        Returns:
            Dictionary mapping (from_state, to_state) → baseline_rate
        """
        pass
    
    @abstractmethod
    def build_rate_matrix(self, family_idx: Optional[int] = None) -> np.ndarray:
        """
        Build baseline rate matrix Q.
        
        Args:
            family_idx: Optional family index for hierarchical models
        
        Returns:
            (4, 4) rate matrix
        """
        pass


@dataclass
class GlobalBaseline(CopyNumberBaseline):
    """
    Global baseline model.
    
    Single set of rates for all families.
    
    Good for:
        - Fast exploratory analysis
        - Homogeneous datasets
        - Initial testing
    
    Parameters:
        gain_rate: Rate of gain (0→1)
        loss_rate: Rate of loss (1→0)
        amplify_rate: Rate of amplification (1→2, 2→3)
        contract_rate: Rate of contraction (3→2, 2→1)
    """
    gain_rate: float = 0.1
    loss_rate: float = 0.1
    amplify_rate: float = 0.05
    contract_rate: float = 0.05
    
    def __post_init__(self):
        """Validate rates are positive."""
        if self.gain_rate <= 0:
            raise ValueError(f"gain_rate must be positive, got {self.gain_rate}")
        if self.loss_rate <= 0:
            raise ValueError(f"loss_rate must be positive, got {self.loss_rate}")
        if self.amplify_rate <= 0:
            raise ValueError(f"amplify_rate must be positive, got {self.amplify_rate}")
        if self.contract_rate <= 0:
            raise ValueError(f"contract_rate must be positive, got {self.contract_rate}")
    
    def get_baseline_rates(self, family_idx: Optional[int] = None) -> Dict[Tuple[int, int], float]:
        """Get baseline rates (same for all families)."""
        return {
            (0, 1): self.gain_rate,
            (1, 0): self.loss_rate,
            (1, 2): self.amplify_rate,
            (2, 1): self.contract_rate,
            (2, 3): self.amplify_rate,
            (3, 2): self.contract_rate,
        }
    
    def build_rate_matrix(self, family_idx: Optional[int] = None) -> np.ndarray:
        """Build baseline rate matrix."""
        Q = np.zeros((4, 4))
        
        rates = self.get_baseline_rates(family_idx)
        
        # Fill in rates
        for (i, j), rate in rates.items():
            Q[i, j] = rate
        
        # Set diagonal to make rows sum to zero
        for i in range(4):
            Q[i, i] = -Q[i, :].sum() + Q[i, i]
        
        return Q


@dataclass
class HierarchicalBaseline(CopyNumberBaseline):
    """
    Hierarchical baseline model (DEFAULT, RECOMMENDED).
    
    Per-family rates drawn from global distribution:
        log(rate_fam) ~ Normal(log(rate_global), σ)
    
    Why this is good:
        - Families differ in dosage tolerance
        - Avoids false positives (learned from GeneContent)
        - More biologically realistic
        - Statistically principled
    
    Parameters:
        global_gain_rate: Global mean gain rate
        global_loss_rate: Global mean loss rate
        global_amplify_rate: Global mean amplification rate
        global_contract_rate: Global mean contraction rate
        sigma: Standard deviation of log-rates across families
        family_rates: Optional pre-computed per-family rates
    """
    global_gain_rate: float = 0.1
    global_loss_rate: float = 0.1
    global_amplify_rate: float = 0.05
    global_contract_rate: float = 0.05
    sigma: float = 0.5
    family_rates: Optional[Dict[int, Dict[Tuple[int, int], float]]] = None
    
    def __post_init__(self):
        """Validate parameters."""
        if self.global_gain_rate <= 0:
            raise ValueError(f"global_gain_rate must be positive, got {self.global_gain_rate}")
        if self.global_loss_rate <= 0:
            raise ValueError(f"global_loss_rate must be positive, got {self.global_loss_rate}")
        if self.global_amplify_rate <= 0:
            raise ValueError(f"global_amplify_rate must be positive, got {self.global_amplify_rate}")
        if self.global_contract_rate <= 0:
            raise ValueError(f"global_contract_rate must be positive, got {self.global_contract_rate}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
    
    def sample_family_rates(self, n_families: int, rng: np.random.Generator) -> None:
        """
        Sample per-family rates from hierarchical model.
        
        Args:
            n_families: Number of gene families
            rng: Random number generator
        """
        self.family_rates = {}
        
        global_rates = {
            'gain': self.global_gain_rate,
            'loss': self.global_loss_rate,
            'amplify': self.global_amplify_rate,
            'contract': self.global_contract_rate,
        }
        
        for fam_idx in range(n_families):
            # Sample log-rates
            log_rates = {
                name: np.log(rate) + rng.normal(0, self.sigma)
                for name, rate in global_rates.items()
            }
            
            # Convert back to rates
            rates = {name: np.exp(lr) for name, lr in log_rates.items()}
            
            # Map to transitions
            self.family_rates[fam_idx] = {
                (0, 1): rates['gain'],
                (1, 0): rates['loss'],
                (1, 2): rates['amplify'],
                (2, 1): rates['contract'],
                (2, 3): rates['amplify'],
                (3, 2): rates['contract'],
            }
    
    def get_baseline_rates(self, family_idx: Optional[int] = None) -> Dict[Tuple[int, int], float]:
        """
        Get baseline rates for a specific family.
        
        Args:
            family_idx: Family index (required for hierarchical model)
        
        Returns:
            Per-family baseline rates
        """
        if family_idx is None:
            raise ValueError("family_idx required for HierarchicalBaseline")
        
        if self.family_rates is None:
            raise ValueError("Must call sample_family_rates() first")
        
        if family_idx not in self.family_rates:
            raise ValueError(f"Family {family_idx} not in family_rates")
        
        return self.family_rates[family_idx]
    
    def build_rate_matrix(self, family_idx: Optional[int] = None) -> np.ndarray:
        """Build per-family baseline rate matrix."""
        Q = np.zeros((4, 4))
        
        rates = self.get_baseline_rates(family_idx)
        
        # Fill in rates
        for (i, j), rate in rates.items():
            Q[i, j] = rate
        
        # Set diagonal to make rows sum to zero
        for i in range(4):
            Q[i, i] = -Q[i, :].sum() + Q[i, i]
        
        return Q
    
    def get_global_rates(self) -> Dict[Tuple[int, int], float]:
        """Get global mean rates (for reporting)."""
        return {
            (0, 1): self.global_gain_rate,
            (1, 0): self.global_loss_rate,
            (1, 2): self.global_amplify_rate,
            (2, 1): self.global_contract_rate,
            (2, 3): self.global_amplify_rate,
            (3, 2): self.global_contract_rate,
        }


def create_baseline(
    baseline_type: str = 'hierarchical',
    **kwargs
) -> CopyNumberBaseline:
    """
    Factory function for creating baseline models.
    
    Args:
        baseline_type: 'hierarchical' (default) or 'global'
        **kwargs: Parameters for the baseline model
    
    Returns:
        Baseline model instance
    
    Example:
        >>> baseline = create_baseline('hierarchical', sigma=0.3)
        >>> baseline = create_baseline('global', gain_rate=0.2)
    """
    if baseline_type == 'hierarchical':
        return HierarchicalBaseline(**kwargs)
    elif baseline_type == 'global':
        return GlobalBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown baseline_type: {baseline_type}")
