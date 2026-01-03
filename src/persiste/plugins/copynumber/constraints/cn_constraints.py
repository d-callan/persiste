"""
Constraint models for copy number dynamics.

Constraints modulate baseline rates to capture biological phenomena:
    - Dosage stability (buffering vs volatility)
    - Amplification bias (adaptive CNV)
    - Host-conditioned volatility (lineage-specific)

All constraints use multiplicative modifiers:
    Q_ij = baseline_rate_ij × exp(θ_constraint)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Set
import numpy as np

from persiste.core.constraint_utils import (
    MAX_RATE as CORE_MAX_RATE,
    MultiplicativeConstraint,
    apply_multiplicative_constraint,
)
from persiste.plugins.copynumber.states.cn_states import (
    get_sparse_transition_graph,
    validate_transition_matrix,
)

MAX_RATE = CORE_MAX_RATE


@dataclass
class CopyNumberConstraint(MultiplicativeConstraint, ABC):
    """
    Abstract base class for copy number constraints.

    Constraints modify baseline rates via multiplicative factors:
        effective_rate = baseline_rate × exp(θ)
    
    Never additive - learned from GeneContent experience.
    """
    
    @abstractmethod
    def get_rate_multipliers(
        self,
        theta: float,
        family_idx: Optional[int] = None,
        lineage_id: Optional[str] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Get multiplicative rate modifiers.
        
        Args:
            theta: Constraint parameter
            family_idx: Optional family index
            lineage_id: Optional lineage identifier
        
        Returns:
            Dictionary mapping (from_state, to_state) → exp(θ * effect)
        """
        pass
    
    @abstractmethod
    def get_affected_transitions(self) -> Set[Tuple[int, int]]:
        """
        Get set of transitions affected by this constraint.
        
        Returns:
            Set of (from_state, to_state) tuples
        """
        pass


@dataclass
class DosageStabilityConstraint(CopyNumberConstraint):
    """
    Dosage stability constraint (CORE CONSTRAINT).
    
    Biological question:
        "Do some genes resist copy number changes?"
    
    Effect:
        - Suppresses BOTH amplification AND contraction
        - Encourages residence in single or low-multi states
        - Symmetric effect on all CN changes
    
    Parameter interpretation:
        θ < 0  → dosage buffered (stable copy number)
        θ = 0  → neutral (baseline rates)
        θ > 0  → dosage volatile (frequent CN changes)
    
    Affected transitions:
        ALL non-diagonal transitions (0↔1, 1↔2, 2↔3)
    
    Use cases:
        - Essential genes (expect θ < 0)
        - Housekeeping genes (expect θ < 0)
        - Core metabolism (expect θ < 0)
        - Antigen families (expect θ > 0)
    """
    
    def get_rate_multipliers(
        self,
        theta: float,
        family_idx: Optional[int] = None,
        lineage_id: Optional[str] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Get dosage stability multipliers.
        
        All CN-changing transitions get same modifier.
        """
        multiplier = np.exp(theta)
        
        return {
            (0, 1): multiplier,  # gain
            (1, 0): multiplier,  # loss
            (1, 2): multiplier,  # amplify low
            (2, 1): multiplier,  # contract low
            (2, 3): multiplier,  # amplify high
            (3, 2): multiplier,  # contract high
        }
    
    def get_affected_transitions(self) -> Set[Tuple[int, int]]:
        """All non-diagonal transitions."""
        return {
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (2, 3), (3, 2),
        }


@dataclass
class AmplificationBiasConstraint(CopyNumberConstraint):
    """
    Amplification bias constraint.
    
    Biological question:
        "Do pathogenic lineages favor copy number increases?"
    
    Effect:
        - Boosts amplification (1→2, 2→3) by exp(θ)
        - Suppresses contraction (2→1, 3→2) by exp(-θ)
        - Does NOT affect gain/loss (0↔1) - gene birth ≠ amplification
        - Bidirectional for stronger signal
    
    Parameter interpretation:
        θ < 0  → amplification suppressed, contraction favored
        θ = 0  → neutral (baseline rates)
        θ > 0  → amplification favored, contraction suppressed
    
    Affected transitions:
        1→2 (amplify to low-multi) × exp(θ)
        2→3 (amplify to high-multi) × exp(θ)
        2→1 (contract from low-multi) × exp(-θ)
        3→2 (contract from high-multi) × exp(-θ)
    
    NOT affected:
        0→1 (gene birth) - biologically distinct from amplification
        1→0 (gene loss) - not part of amplification dynamics
    
    Use cases:
        - Drug resistance genes (expect θ > 0)
        - Antigen families (expect θ > 0)
        - Efflux pumps (expect θ > 0)
        - Virulence factors (expect θ > 0)
    
    Design rationale:
        1. Bidirectional: amplification up AND contraction down
        2. State-specific: only affects multi-copy transitions (1↔2, 2↔3)
        3. Excludes gene birth (0→1): prevents signal dilution
        4. Biological precision: amplification ≠ gene birth
    """
    
    def get_rate_multipliers(
        self,
        theta: float,
        family_idx: Optional[int] = None,
        lineage_id: Optional[str] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Get amplification bias multipliers.
        
        Bidirectional: amplification up, contraction down.
        """
        amplify_multiplier = np.exp(theta)
        contract_multiplier = np.exp(-theta)
        
        return {
            (1, 2): amplify_multiplier,   # amplify low
            (2, 3): amplify_multiplier,   # amplify high
            (2, 1): contract_multiplier,  # contract low
            (3, 2): contract_multiplier,  # contract high
        }
    
    def get_affected_transitions(self) -> Set[Tuple[int, int]]:
        """Amplification and contraction transitions."""
        return {(1, 2), (2, 3), (2, 1), (3, 2)}


@dataclass
class HostConditionedVolatilityConstraint(CopyNumberConstraint):
    """
    Host-conditioned volatility constraint.
    
    Biological question:
        "Does copy number evolve differently in host-associated lineages?"
    
    Effect:
        - Lineage-conditioned multiplier on ALL CN transitions
        - Can be applied to specific lineages (e.g., host-associated)
        - Pairs naturally with GeneContent's host association
    
    Parameter interpretation:
        θ < 0  → CN more stable in this lineage
        θ = 0  → neutral (baseline rates)
        θ > 0  → CN more volatile in this lineage
    
    Affected transitions:
        ALL non-diagonal transitions (if lineage matches)
    
    Use cases:
        - Host-adapted pathogens
        - Environmental vs clinical isolates
        - Commensal vs pathogenic strains
    
    Design note:
        Requires lineage annotation. If lineage_id not provided
        or doesn't match target lineage, returns neutral multipliers.
    """
    target_lineage: str = 'host_associated'
    
    def get_rate_multipliers(
        self,
        theta: float,
        family_idx: Optional[int] = None,
        lineage_id: Optional[str] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Get host-conditioned multipliers.
        
        Only applies if lineage_id matches target_lineage.
        """
        # Check if this lineage is affected
        if lineage_id is None or lineage_id != self.target_lineage:
            # Neutral multipliers (no effect)
            multiplier = 1.0
        else:
            # Apply constraint
            multiplier = np.exp(theta)
        
        return {
            (0, 1): multiplier,  # gain
            (1, 0): multiplier,  # loss
            (1, 2): multiplier,  # amplify low
            (2, 1): multiplier,  # contract low
            (2, 3): multiplier,  # amplify high
            (3, 2): multiplier,  # contract high
        }
    
    def get_affected_transitions(self) -> Set[Tuple[int, int]]:
        """All non-diagonal transitions (conditionally)."""
        return {
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (2, 3), (3, 2),
        }


def apply_constraint(
    baseline_Q: np.ndarray,
    constraint: CopyNumberConstraint,
    theta: float,
    family_idx: Optional[int] = None,
    lineage_id: Optional[str] = None
) -> np.ndarray:
    """
    Apply constraint to baseline rate matrix.
    Args:
        baseline_Q: (4, 4) baseline rate matrix
        constraint: Constraint to apply
        theta: Constraint parameter
        family_idx: Optional family index
        lineage_id: Optional lineage identifier
    Returns:
        (4, 4) constrained rate matrix
    Example:
        >>> baseline = create_baseline('global')
        >>> Q_base = baseline.build_rate_matrix()
        >>> constraint = DosageStabilityConstraint()
        >>> Q_constrained = apply_constraint(Q_base, constraint, theta=-0.5)
    """
    multipliers = constraint.get_rate_multipliers(theta, family_idx, lineage_id)
    allowed = get_sparse_transition_graph()

    def _validator(q: np.ndarray) -> None:
        validate_transition_matrix(q, allowed)

    return apply_multiplicative_constraint(
        baseline_Q,
        multipliers,
        max_rate=MAX_RATE,
        validator=_validator,
    )


def create_constraint(constraint_type: str, **kwargs) -> CopyNumberConstraint:
    """
    Factory function for creating constraints.
    
    Args:
        constraint_type: Type of constraint
            - 'dosage_stability' (default, recommended)
            - 'amplification_bias'
            - 'host_conditioned'
        **kwargs: Parameters for the constraint
    
    Returns:
        Constraint instance
    
    Example:
        >>> constraint = create_constraint('dosage_stability')
        >>> constraint = create_constraint('amplification_bias')
        >>> constraint = create_constraint('host_conditioned', target_lineage='clinical')
    """
    if constraint_type == 'dosage_stability':
        return DosageStabilityConstraint(**kwargs)
    elif constraint_type == 'amplification_bias':
        return AmplificationBiasConstraint(**kwargs)
    elif constraint_type == 'host_conditioned':
        return HostConditionedVolatilityConstraint(**kwargs)
    else:
        raise ValueError(f"Unknown constraint_type: {constraint_type}")
