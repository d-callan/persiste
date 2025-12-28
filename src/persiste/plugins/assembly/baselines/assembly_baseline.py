"""
Physics-agnostic baseline for assembly transitions.

Baseline knows nothing about chemistry, catalysis, or "life."
Pure combinatorics and size effects.
"""

from enum import Enum
from typing import Optional
import numpy as np

from persiste.core.baseline import Baseline
from persiste.plugins.assembly.states.assembly_state import AssemblyState


class TransitionType(Enum):
    """Primitive assembly transition types."""
    JOIN = "join"      # X + Y → X∘Y
    SPLIT = "split"    # X∘Y → X + Y
    DECAY = "decay"    # X → ∅
    REARRANGE = "rearrange"  # X∘Y → X′∘Y′ (optional, phase 2)


class AssemblyBaseline(Baseline):
    """
    Physics-agnostic baseline for assembly transitions.
    
    Factorized rate formula:
        λ_baseline(i → j) = κ × f(size_i) × g(size_j) × h(type)
    
    Where:
        κ - Global rate constant
        f(size_i) - Source size factor
        g(size_j) - Target size factor
        h(type) - Transition type factor
    
    Key principle: Baseline doesn't know about chemistry.
    No functional groups, no catalysis, no "life."
    
    Attributes:
        kappa: Global rate constant
        join_exponent: Size scaling for join transitions (typically negative)
        split_exponent: Size scaling for split transitions (typically positive)
        decay_rate: Base decay rate
    """
    
    def __init__(
        self,
        kappa: float = 1.0,
        join_exponent: float = -0.5,
        split_exponent: float = 0.3,
        decay_rate: float = 0.01,
    ):
        """
        Initialize assembly baseline.
        
        Args:
            kappa: Global rate constant (default: 1.0)
            join_exponent: Size scaling for joins (default: -0.5, harder with size)
            split_exponent: Size scaling for splits (default: 0.3, easier with size)
            decay_rate: Base decay rate (default: 0.01)
        """
        self.kappa = kappa
        self.join_exponent = join_exponent
        self.split_exponent = split_exponent
        self.decay_rate = decay_rate
    
    def get_rate(
        self,
        i: int,
        j: int,
        source: Optional[AssemblyState] = None,
        target: Optional[AssemblyState] = None,
        transition_type: Optional[TransitionType] = None,
    ) -> float:
        """
        Get baseline transition rate.
        
        For assembly, we need the full states, not just indices.
        This method signature matches PERSISTE's Baseline interface.
        
        Args:
            i: Source state index (unused for assembly)
            j: Target state index (unused for assembly)
            source: Source AssemblyState
            target: Target AssemblyState
            transition_type: Type of transition
            
        Returns:
            Baseline transition rate (no chemistry, pure size effects)
        """
        if source is None or target is None or transition_type is None:
            # Fallback for interface compatibility
            return 0.0
        
        return self.get_assembly_rate(source, target, transition_type)
    
    def get_assembly_rate(
        self,
        source: AssemblyState,
        target: AssemblyState,
        transition_type: TransitionType,
    ) -> float:
        """
        Compute baseline transition rate for assembly.
        
        No chemistry. No functional groups. No catalysis.
        Pure size and type effects.
        
        Args:
            source: Source assembly state
            target: Target assembly state
            transition_type: Type of transition
            
        Returns:
            Baseline rate λ_baseline
        """
        if transition_type == TransitionType.JOIN:
            return self._join_rate(source, target)
        elif transition_type == TransitionType.SPLIT:
            return self._split_rate(source, target)
        elif transition_type == TransitionType.DECAY:
            return self._decay_rate(source)
        elif transition_type == TransitionType.REARRANGE:
            return self._rearrange_rate(source, target)
        else:
            return 0.0
    
    def _join_rate(self, source: AssemblyState, target: AssemblyState) -> float:
        """
        Join rate: X + Y → X∘Y
        
        Harder to join larger assemblies (negative exponent).
        """
        # Use target depth as proxy for combined size
        size_factor = (target.assembly_depth ** self.join_exponent) if target.assembly_depth > 0 else 1.0
        return self.kappa * size_factor
    
    def _split_rate(self, source: AssemblyState, target: AssemblyState) -> float:
        """
        Split rate: X∘Y → X + Y
        
        Easier to split larger assemblies (positive exponent).
        """
        # Use source depth as proxy for assembly size
        size_factor = (source.assembly_depth ** self.split_exponent) if source.assembly_depth > 0 else 1.0
        return self.kappa * size_factor
    
    def _decay_rate(self, source: AssemblyState) -> float:
        """
        Decay rate: X → ∅
        
        Constant decay rate (could be size-dependent in future).
        """
        return self.decay_rate
    
    def _rearrange_rate(self, source: AssemblyState, target: AssemblyState) -> float:
        """
        Rearrange rate: X∘Y → X′∘Y′
        
        Rare under baseline (phase 2 feature).
        """
        # For now, very low rate
        return self.kappa * 0.01
    
    def __str__(self) -> str:
        return (
            f"AssemblyBaseline(κ={self.kappa}, "
            f"join_exp={self.join_exponent}, "
            f"split_exp={self.split_exponent}, "
            f"decay={self.decay_rate})"
        )
