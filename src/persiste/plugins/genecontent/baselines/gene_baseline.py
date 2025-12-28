"""
Baseline model for gene content evolution.

Purpose: Absorb intrinsic gene volatility so constraints remain interpretable.

The baseline captures "what would happen without selective constraint":
- Some genes are inherently more volatile (high gain/loss rates)
- Some genes are inherently stable (low gain/loss rates)
- This variation is NOT constraint - it's opportunity

Baseline Options:
1. Hierarchical (recommended): Per-family rates drawn from population distribution
   log λ_gain(g) ~ Normal(μ_gain, σ_gain)
   log λ_loss(g) ~ Normal(μ_loss, σ_loss)

2. Fixed: User-provided per-family rates

3. Global: Single gain/loss rate for all families (simple, for testing)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class RateParameters:
    """
    Gain and loss rates for a gene family.
    
    Attributes:
        gain_rate: Rate of 0→1 transition (gene acquisition)
        loss_rate: Rate of 1→0 transition (gene loss)
        family_id: Gene family identifier
    """
    gain_rate: float
    loss_rate: float
    family_id: str
    
    def __post_init__(self):
        if self.gain_rate < 0:
            raise ValueError(f"gain_rate must be non-negative, got {self.gain_rate}")
        if self.loss_rate < 0:
            raise ValueError(f"loss_rate must be non-negative, got {self.loss_rate}")
    
    @property
    def total_rate(self) -> float:
        """Total transition rate (gain + loss)."""
        return self.gain_rate + self.loss_rate
    
    def rate_matrix(self) -> np.ndarray:
        """
        Return 2x2 rate matrix for this family.
        
        States: 0 = absent, 1 = present
        Q[i,j] = rate from state i to state j (i ≠ j)
        Q[i,i] = -sum of off-diagonal rates
        
        Returns:
            2x2 numpy array
        """
        Q = np.array([
            [-self.gain_rate, self.gain_rate],   # from absent
            [self.loss_rate, -self.loss_rate],   # from present
        ])
        return Q
    
    def transition_probability(self, t: float) -> np.ndarray:
        """
        Compute transition probability matrix P(t) = exp(Qt).
        
        For 2-state system, this has closed form.
        
        Args:
            t: Branch length (time)
            
        Returns:
            2x2 transition probability matrix
        """
        if t < 0:
            raise ValueError(f"Branch length must be non-negative, got {t}")
        
        λ = self.gain_rate
        μ = self.loss_rate
        total = λ + μ
        
        if total < 1e-10:
            # No transitions possible
            return np.eye(2)
        
        # Closed-form solution for 2-state CTMC
        exp_term = np.exp(-total * t)
        
        # P[i,j] = probability of being in state j given starting in state i
        p00 = (μ + λ * exp_term) / total  # absent → absent
        p01 = (λ - λ * exp_term) / total  # absent → present
        p10 = (μ - μ * exp_term) / total  # present → absent
        p11 = (λ + μ * exp_term) / total  # present → present
        
        return np.array([[p00, p01], [p10, p11]])


class GeneContentBaseline(ABC):
    """
    Abstract base class for gene content baseline models.
    
    The baseline defines the "neutral" gain/loss rates for each gene family.
    Constraints then modify these rates to capture selective effects.
    """
    
    @abstractmethod
    def get_rates(self, family_id: str) -> RateParameters:
        """
        Get baseline gain/loss rates for a gene family.
        
        Args:
            family_id: Gene family identifier
            
        Returns:
            RateParameters with gain and loss rates
        """
        pass
    
    @abstractmethod
    def get_all_rates(self, family_ids: List[str]) -> Dict[str, RateParameters]:
        """
        Get baseline rates for all gene families.
        
        Args:
            family_ids: List of gene family identifiers
            
        Returns:
            Dict mapping family_id -> RateParameters
        """
        pass
    
    @abstractmethod
    def n_parameters(self) -> int:
        """Number of free parameters in the baseline model."""
        pass
    
    def log_prior(self) -> float:
        """Log prior probability of baseline parameters (default: 0 = flat prior)."""
        return 0.0


@dataclass
class HierarchicalRates(GeneContentBaseline):
    """
    Hierarchical baseline: per-family rates drawn from population distribution.
    
    log λ_gain(g) ~ Normal(μ_gain, σ_gain)
    log λ_loss(g) ~ Normal(μ_loss, σ_loss)
    
    This is the recommended default because:
    - Reduces false positives by accounting for rate heterogeneity
    - Matches biological reality (genes vary in volatility)
    - Scales well with many gene families
    - Reviewer-safe (principled shrinkage)
    
    Attributes:
        mu_gain: Mean of log gain rate distribution
        sigma_gain: Std dev of log gain rate distribution
        mu_loss: Mean of log loss rate distribution
        sigma_loss: Std dev of log loss rate distribution
        family_rates: Cached per-family rates (fitted or sampled)
        essential_genes: Set of gene families with near-zero loss rate
    """
    mu_gain: float = -2.0  # Default: low gain rate
    sigma_gain: float = 1.0
    mu_loss: float = -1.0  # Default: moderate loss rate
    sigma_loss: float = 1.0
    family_rates: Dict[str, RateParameters] = field(default_factory=dict)
    essential_genes: Optional[set] = None
    
    def get_rates(self, family_id: str) -> RateParameters:
        """Get rates for a family, sampling if not cached."""
        if family_id not in self.family_rates:
            self._sample_rates([family_id])
        return self.family_rates[family_id]
    
    def get_all_rates(self, family_ids: List[str]) -> Dict[str, RateParameters]:
        """Get rates for all families."""
        missing = [f for f in family_ids if f not in self.family_rates]
        if missing:
            self._sample_rates(missing)
        return {f: self.family_rates[f] for f in family_ids}
    
    def _sample_rates(self, family_ids: List[str], rng: Optional[np.random.Generator] = None):
        """Sample rates for families from hierarchical prior."""
        if rng is None:
            rng = np.random.default_rng()
        
        for fam in family_ids:
            # Sample log rates from normal distribution
            log_gain = rng.normal(self.mu_gain, self.sigma_gain)
            log_loss = rng.normal(self.mu_loss, self.sigma_loss)
            
            gain_rate = np.exp(log_gain)
            loss_rate = np.exp(log_loss)
            
            # Handle essential genes
            if self.essential_genes and fam in self.essential_genes:
                loss_rate = 1e-10  # Near-zero loss rate
            
            self.family_rates[fam] = RateParameters(
                gain_rate=gain_rate,
                loss_rate=loss_rate,
                family_id=fam
            )
    
    def set_rates(self, family_id: str, gain_rate: float, loss_rate: float):
        """Manually set rates for a family (e.g., from MLE)."""
        self.family_rates[family_id] = RateParameters(
            gain_rate=gain_rate,
            loss_rate=loss_rate,
            family_id=family_id
        )
    
    def n_parameters(self) -> int:
        """4 hyperparameters + 2 per family."""
        return 4 + 2 * len(self.family_rates)
    
    def log_prior(self) -> float:
        """Log prior: sum of log-normal priors on family rates."""
        log_p = 0.0
        for fam, rates in self.family_rates.items():
            # Log-normal prior on rates
            log_gain = np.log(rates.gain_rate + 1e-10)
            log_loss = np.log(rates.loss_rate + 1e-10)
            
            # Normal log-likelihood
            log_p -= 0.5 * ((log_gain - self.mu_gain) / self.sigma_gain) ** 2
            log_p -= 0.5 * ((log_loss - self.mu_loss) / self.sigma_loss) ** 2
        
        return log_p


@dataclass
class FixedRates(GeneContentBaseline):
    """
    Fixed baseline: user-provided per-family rates.
    
    Use when you have prior knowledge of gene family volatility
    (e.g., from a reference dataset or literature).
    
    Attributes:
        rates: Dict mapping family_id -> (gain_rate, loss_rate)
        default_gain: Default gain rate for unknown families
        default_loss: Default loss rate for unknown families
    """
    rates: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    default_gain: float = 0.1
    default_loss: float = 0.1
    
    def get_rates(self, family_id: str) -> RateParameters:
        """Get rates for a family."""
        if family_id in self.rates:
            gain, loss = self.rates[family_id]
        else:
            gain, loss = self.default_gain, self.default_loss
        
        return RateParameters(
            gain_rate=gain,
            loss_rate=loss,
            family_id=family_id
        )
    
    def get_all_rates(self, family_ids: List[str]) -> Dict[str, RateParameters]:
        """Get rates for all families."""
        return {f: self.get_rates(f) for f in family_ids}
    
    def n_parameters(self) -> int:
        """2 per family with explicit rates + 2 defaults."""
        return 2 * len(self.rates) + 2


@dataclass
class GlobalRates(GeneContentBaseline):
    """
    Global baseline: single gain/loss rate for all families.
    
    Simple model for testing or when rate heterogeneity is not a concern.
    
    Attributes:
        gain_rate: Global gain rate (0→1)
        loss_rate: Global loss rate (1→0)
    """
    gain_rate: float = 0.1
    loss_rate: float = 0.1
    
    def get_rates(self, family_id: str) -> RateParameters:
        """Get global rates (same for all families)."""
        return RateParameters(
            gain_rate=self.gain_rate,
            loss_rate=self.loss_rate,
            family_id=family_id
        )
    
    def get_all_rates(self, family_ids: List[str]) -> Dict[str, RateParameters]:
        """Get rates for all families (all same)."""
        return {f: self.get_rates(f) for f in family_ids}
    
    def n_parameters(self) -> int:
        """2 global parameters."""
        return 2
