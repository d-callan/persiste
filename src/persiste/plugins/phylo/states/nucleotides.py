"""Nucleotide state space for phylogenetic analysis."""

from typing import Optional, Dict
import numpy as np

from persiste.core.states import StateSpace


NUCLEOTIDES = ['A', 'C', 'G', 'T']
PURINES = {'A', 'G'}
PYRIMIDINES = {'C', 'T'}


class NucleotideStateSpace(StateSpace):
    """
    State space of nucleotides (A, C, G, T).
    
    Simple 4-state space for nucleotide substitution models.
    Used as building block for codon models and for nucleotide-level analysis.
    
    Attributes:
        nt_to_index: Mapping from nucleotide to state index
        index_to_nt: Mapping from state index to nucleotide
    """
    
    def __init__(self, frequencies: Optional[np.ndarray] = None):
        """
        Initialize nucleotide state space.
        
        Args:
            frequencies: Optional equilibrium frequencies (4-vector, order: A,C,G,T)
        """
        self.nt_to_index: Dict[str, int] = {nt: i for i, nt in enumerate(NUCLEOTIDES)}
        self.index_to_nt: Dict[int, str] = {i: nt for i, nt in enumerate(NUCLEOTIDES)}
        
        self._frequencies = frequencies
        
        super().__init__(
            states=NUCLEOTIDES.copy(),
            dimension=4,
            metadata={"type": "nucleotides"}
        )
    
    @classmethod
    def uniform(cls) -> "NucleotideStateSpace":
        """Create with uniform frequencies (0.25 each)."""
        return cls(np.array([0.25, 0.25, 0.25, 0.25]))
    
    @classmethod
    def from_frequencies(
        cls,
        pi_A: float,
        pi_C: float,
        pi_G: float,
        pi_T: float,
    ) -> "NucleotideStateSpace":
        """
        Create with specified frequencies.
        
        Args:
            pi_A, pi_C, pi_G, pi_T: Nucleotide frequencies
            
        Returns:
            NucleotideStateSpace with given frequencies
        """
        freqs = np.array([pi_A, pi_C, pi_G, pi_T])
        freqs = freqs / freqs.sum()  # Normalize
        return cls(freqs)
    
    def nucleotide(self, index: int) -> str:
        """Get nucleotide for state index."""
        return self.index_to_nt[index]
    
    def index(self, nt: str) -> int:
        """Get state index for nucleotide."""
        return self.nt_to_index[nt.upper()]
    
    def is_purine(self, index: int) -> bool:
        """Check if nucleotide at index is a purine (A or G)."""
        return self.index_to_nt[index] in PURINES
    
    def is_pyrimidine(self, index: int) -> bool:
        """Check if nucleotide at index is a pyrimidine (C or T)."""
        return self.index_to_nt[index] in PYRIMIDINES
    
    def is_transition(self, i: int, j: int) -> bool:
        """
        Check if i→j is a transition (purine↔purine or pyrimidine↔pyrimidine).
        
        Transitions: A↔G, C↔T
        """
        if i == j:
            return False
        
        nt_i, nt_j = self.index_to_nt[i], self.index_to_nt[j]
        return (nt_i in PURINES and nt_j in PURINES) or \
               (nt_i in PYRIMIDINES and nt_j in PYRIMIDINES)
    
    def is_transversion(self, i: int, j: int) -> bool:
        """
        Check if i→j is a transversion (purine↔pyrimidine).
        
        Transversions: A↔C, A↔T, G↔C, G↔T
        """
        if i == j:
            return False
        return not self.is_transition(i, j)
    
    @property
    def frequencies(self) -> np.ndarray:
        """Get nucleotide equilibrium frequencies."""
        if self._frequencies is not None:
            return self._frequencies
        return np.array([0.25, 0.25, 0.25, 0.25])
    
    @frequencies.setter
    def frequencies(self, value: np.ndarray):
        """Set nucleotide frequencies."""
        if len(value) != 4:
            raise ValueError(f"Expected 4 frequencies, got {len(value)}")
        self._frequencies = value / value.sum()
    
    def __repr__(self) -> str:
        return f"NucleotideStateSpace(ACGT)"
