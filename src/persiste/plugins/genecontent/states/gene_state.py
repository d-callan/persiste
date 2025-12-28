"""
State model for gene content evolution.

State = gene family presence/absence vector
For taxon t: S_t = { g₁ ∈ {0,1}, g₂ ∈ {0,1}, … }

Key properties:
- Binary (v1)
- Independent dimensions per gene family
- No explicit genome size constraint (yet)
- Compatible with CTMC on trees
- Extensible later to copy number
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import numpy as np


@dataclass(frozen=True)
class GenePresenceState:
    """
    Presence/absence state for a single gene family at a single node.
    
    This is the atomic unit for per-family CTMC.
    
    Attributes:
        present: True if gene family is present, False if absent
        family_id: Identifier for the gene family (e.g., 'OG0001')
    """
    present: bool
    family_id: str
    
    @property
    def value(self) -> int:
        """Return 1 if present, 0 if absent."""
        return 1 if self.present else 0
    
    def __repr__(self) -> str:
        status = "+" if self.present else "-"
        return f"{self.family_id}:{status}"
    
    @classmethod
    def present_state(cls, family_id: str) -> 'GenePresenceState':
        """Create a present state for a gene family."""
        return cls(present=True, family_id=family_id)
    
    @classmethod
    def absent_state(cls, family_id: str) -> 'GenePresenceState':
        """Create an absent state for a gene family."""
        return cls(present=False, family_id=family_id)


@dataclass
class GeneFamilyVector:
    """
    Full gene content state for a taxon: presence/absence across all families.
    
    This represents the complete genome content at a node in the tree.
    
    Attributes:
        presence: Dict mapping family_id -> bool (True=present, False=absent)
        taxon_id: Optional identifier for the taxon
    """
    presence: Dict[str, bool] = field(default_factory=dict)
    taxon_id: Optional[str] = None
    
    @property
    def n_families(self) -> int:
        """Number of gene families tracked."""
        return len(self.presence)
    
    @property
    def n_present(self) -> int:
        """Number of gene families present."""
        return sum(1 for v in self.presence.values() if v)
    
    @property
    def n_absent(self) -> int:
        """Number of gene families absent."""
        return sum(1 for v in self.presence.values() if not v)
    
    @property
    def present_families(self) -> Set[str]:
        """Set of family IDs that are present."""
        return {fam for fam, present in self.presence.items() if present}
    
    @property
    def absent_families(self) -> Set[str]:
        """Set of family IDs that are absent."""
        return {fam for fam, present in self.presence.items() if not present}
    
    def is_present(self, family_id: str) -> bool:
        """Check if a gene family is present."""
        return self.presence.get(family_id, False)
    
    def get_state(self, family_id: str) -> GenePresenceState:
        """Get the GenePresenceState for a specific family."""
        return GenePresenceState(
            present=self.presence.get(family_id, False),
            family_id=family_id
        )
    
    def to_array(self, family_order: List[str]) -> np.ndarray:
        """
        Convert to numpy array with specified family ordering.
        
        Args:
            family_order: List of family IDs defining column order
            
        Returns:
            1D array of 0/1 values
        """
        return np.array([
            1 if self.presence.get(fam, False) else 0
            for fam in family_order
        ], dtype=np.int8)
    
    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        family_order: List[str],
        taxon_id: Optional[str] = None
    ) -> 'GeneFamilyVector':
        """
        Create from numpy array with specified family ordering.
        
        Args:
            array: 1D array of 0/1 values
            family_order: List of family IDs defining column order
            taxon_id: Optional taxon identifier
            
        Returns:
            GeneFamilyVector instance
        """
        presence = {
            fam: bool(val)
            for fam, val in zip(family_order, array)
        }
        return cls(presence=presence, taxon_id=taxon_id)
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, int],
        taxon_id: Optional[str] = None
    ) -> 'GeneFamilyVector':
        """
        Create from dict mapping family_id -> 0/1.
        
        Args:
            data: Dict of family_id -> 0 or 1
            taxon_id: Optional taxon identifier
            
        Returns:
            GeneFamilyVector instance
        """
        presence = {fam: bool(val) for fam, val in data.items()}
        return cls(presence=presence, taxon_id=taxon_id)
    
    def __repr__(self) -> str:
        taxon_str = f"{self.taxon_id}: " if self.taxon_id else ""
        return f"GeneFamilyVector({taxon_str}{self.n_present}/{self.n_families} present)"
    
    def diff(self, other: 'GeneFamilyVector') -> Tuple[Set[str], Set[str]]:
        """
        Compute difference between two states.
        
        Returns:
            (gained, lost): Sets of family IDs gained and lost going from self to other
        """
        self_present = self.present_families
        other_present = other.present_families
        
        gained = other_present - self_present
        lost = self_present - other_present
        
        return gained, lost


def enumerate_transitions(family_id: str) -> List[Tuple[GenePresenceState, GenePresenceState]]:
    """
    Enumerate all possible transitions for a single gene family.
    
    For binary presence/absence, there are exactly 2 transitions:
    - 0 → 1 (gain)
    - 1 → 0 (loss)
    
    Args:
        family_id: Gene family identifier
        
    Returns:
        List of (from_state, to_state) tuples
    """
    absent = GenePresenceState.absent_state(family_id)
    present = GenePresenceState.present_state(family_id)
    
    return [
        (absent, present),  # gain
        (present, absent),  # loss
    ]


def transition_type(from_state: GenePresenceState, to_state: GenePresenceState) -> str:
    """
    Classify a transition as 'gain' or 'loss'.
    
    Args:
        from_state: Starting state
        to_state: Ending state
        
    Returns:
        'gain' if 0→1, 'loss' if 1→0, 'none' if no change
    """
    if not from_state.present and to_state.present:
        return 'gain'
    elif from_state.present and not to_state.present:
        return 'loss'
    else:
        return 'none'
