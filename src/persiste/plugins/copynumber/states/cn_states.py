"""
Copy number state space.

Binned copy number states for tractable CTMC inference.
"""

from enum import IntEnum
from typing import List, Dict, Tuple
import numpy as np


class CopyNumberState(IntEnum):
    """
    Binned copy number states.
    
    Design rationale:
        - Raw copy number (0...N) explodes state space
        - Binning captures biology while keeping CTMC small
        - Works across species with different ploidy
    
    States:
        ABSENT (0): Gene family not present
        SINGLE (1): Single copy (diploid: 2 copies, haploid: 1 copy)
        LOW_MULTI (2): Low multi-copy (2-3 copies beyond baseline)
        HIGH_MULTI (3): High multi-copy (≥4 copies beyond baseline)
    
    Example:
        For diploid organism:
            0 copies → ABSENT
            2 copies → SINGLE
            4-6 copies → LOW_MULTI
            ≥8 copies → HIGH_MULTI
    """
    ABSENT = 0
    SINGLE = 1
    LOW_MULTI = 2
    HIGH_MULTI = 3
    
    @classmethod
    def n_states(cls) -> int:
        """Number of states in the model."""
        return 4
    
    @classmethod
    def state_names(cls) -> List[str]:
        """Human-readable state names."""
        return ['Absent', 'Single', 'Low-Multi', 'High-Multi']
    
    @classmethod
    def from_raw_count(cls, count: int, ploidy: int = 2) -> 'CopyNumberState':
        """
        Map raw copy number to binned state.
        
        Args:
            count: Raw copy number
            ploidy: Organism ploidy (default: diploid = 2)
        
        Returns:
            Binned copy number state
        
        Example:
            >>> CopyNumberState.from_raw_count(0, ploidy=2)
            <CopyNumberState.ABSENT: 0>
            >>> CopyNumberState.from_raw_count(2, ploidy=2)
            <CopyNumberState.SINGLE: 1>
            >>> CopyNumberState.from_raw_count(6, ploidy=2)
            <CopyNumberState.LOW_MULTI: 2>
        """
        if count == 0:
            return cls.ABSENT
        
        baseline = ploidy
        
        if count <= baseline:
            return cls.SINGLE
        
        excess = count - baseline
        
        if excess <= 3:
            return cls.LOW_MULTI
        else:
            return cls.HIGH_MULTI
    
    @classmethod
    def bin_matrix(cls, raw_counts: np.ndarray, ploidy: int = 2) -> np.ndarray:
        """
        Bin an entire copy number matrix.
        
        Args:
            raw_counts: (n_families, n_taxa) matrix of raw copy numbers
            ploidy: Organism ploidy
        
        Returns:
            (n_families, n_taxa) matrix of binned states (0-3)
        """
        binned = np.zeros_like(raw_counts, dtype=int)
        
        for i in range(raw_counts.shape[0]):
            for j in range(raw_counts.shape[1]):
                binned[i, j] = cls.from_raw_count(raw_counts[i, j], ploidy)
        
        return binned


def get_sparse_transition_graph() -> Dict[Tuple[int, int], bool]:
    """
    Get the sparse transition graph for copy number states.
    
    Design:
        Gradual dosage change only:
            0 ↔ 1 ↔ 2 ↔ 3
        
        No direct jumps:
            0 ↔ 3  (forbidden)
            0 ↔ 2  (forbidden)
            1 ↔ 3  (forbidden)
    
    Rationale:
        - Biologically realistic (gradual amplification/contraction)
        - Statistically stabilizing (fewer parameters)
        - Prevents spurious volatility
    
    Returns:
        Dictionary mapping (from_state, to_state) → allowed (bool)
    """
    allowed = {}
    
    # Allowed transitions (bidirectional)
    allowed_pairs = [
        (CopyNumberState.ABSENT, CopyNumberState.SINGLE),
        (CopyNumberState.SINGLE, CopyNumberState.LOW_MULTI),
        (CopyNumberState.LOW_MULTI, CopyNumberState.HIGH_MULTI),
    ]
    
    # Build full transition graph
    for i in range(4):
        for j in range(4):
            if i == j:
                # Self-transitions always allowed (diagonal)
                allowed[(i, j)] = True
            else:
                # Check if transition is in allowed pairs (either direction)
                allowed[(i, j)] = any(
                    (i, j) == pair or (j, i) == pair
                    for pair in allowed_pairs
                )
    
    return allowed


def get_transition_names() -> Dict[Tuple[int, int], str]:
    """
    Get human-readable names for transitions.
    
    Returns:
        Dictionary mapping (from_state, to_state) → name
    """
    names = {
        (0, 1): 'gain',
        (1, 0): 'loss',
        (1, 2): 'amplify_low',
        (2, 1): 'contract_low',
        (2, 3): 'amplify_high',
        (3, 2): 'contract_high',
    }
    
    # Add self-transitions
    for i in range(4):
        names[(i, i)] = f'stay_{CopyNumberState(i).name.lower()}'
    
    return names


def validate_transition_matrix(Q: np.ndarray, allowed: Dict[Tuple[int, int], bool]) -> bool:
    """
    Validate that transition matrix respects sparse graph.
    
    Args:
        Q: (4, 4) rate matrix
        allowed: Allowed transitions from get_sparse_transition_graph()
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    if Q.shape != (4, 4):
        raise ValueError(f"Q must be 4x4, got {Q.shape}")
    
    # Check that forbidden transitions have zero rate
    for i in range(4):
        for j in range(4):
            if i != j and not allowed[(i, j)]:
                if Q[i, j] != 0:
                    raise ValueError(
                        f"Forbidden transition {i}→{j} has non-zero rate {Q[i, j]}"
                    )
    
    # Check row sums (should be zero for valid rate matrix)
    row_sums = Q.sum(axis=1)
    if not np.allclose(row_sums, 0, atol=1e-10):
        raise ValueError(f"Row sums not zero: {row_sums}")
    
    return True
