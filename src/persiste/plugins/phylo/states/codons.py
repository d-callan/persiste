"""Codon state space for phylogenetic analysis."""

from typing import List, Optional, Dict, Any
import numpy as np

from persiste.core.states import StateSpace
from persiste.plugins.phylo.states.genetic_code import GeneticCode


class CodonStateSpace(StateSpace):
    """
    State space of sense codons for codon-based selection analysis.
    
    Wraps the generic StateSpace with codon-specific functionality:
    - Genetic code awareness (Universal, Vertebrate-mtDNA, etc.)
    - Codon ↔ index mapping
    - Amino acid translation
    - Synonymous/nonsynonymous classification
    
    In PERSISTE terms:
    - States are the 61 sense codons (for Universal code)
    - Transitions between codons represent substitutions
    - The genetic code defines which transitions are synonymous
    
    Attributes:
        genetic_code: GeneticCode object defining codon→amino acid mapping
        codon_to_index: Mapping from codon string to state index
        index_to_codon: Mapping from state index to codon string
    """
    
    def __init__(
        self,
        genetic_code: GeneticCode,
        codon_frequencies: Optional[np.ndarray] = None,
    ):
        """
        Initialize codon state space.
        
        Args:
            genetic_code: GeneticCode defining sense codons
            codon_frequencies: Optional equilibrium frequencies (61-vector)
        """
        self.genetic_code = genetic_code
        
        # Build codon ↔ index mappings
        self.codon_to_index: Dict[str, int] = {}
        self.index_to_codon: Dict[int, str] = {}
        
        for i, codon in enumerate(sorted(genetic_code.sense_codons)):
            self.codon_to_index[codon] = i
            self.index_to_codon[i] = codon
        
        # Store frequencies
        self._frequencies = codon_frequencies
        
        # Initialize base StateSpace
        super().__init__(
            states=list(self.index_to_codon.values()),
            dimension=len(genetic_code.sense_codons),
            metadata={
                "genetic_code": genetic_code.name,
                "type": "codons",
            }
        )
    
    @classmethod
    def universal(cls, codon_frequencies: Optional[np.ndarray] = None) -> "CodonStateSpace":
        """
        Create codon state space with universal genetic code.
        
        Args:
            codon_frequencies: Optional equilibrium frequencies
            
        Returns:
            CodonStateSpace with 61 sense codons
        """
        return cls(GeneticCode.universal(), codon_frequencies)
    
    @classmethod
    def from_genetic_code(
        cls,
        code_name: str,
        codon_frequencies: Optional[np.ndarray] = None,
    ) -> "CodonStateSpace":
        """
        Create codon state space from named genetic code.
        
        Args:
            code_name: Name of genetic code (e.g., "Universal", "Vertebrate-mtDNA")
            codon_frequencies: Optional equilibrium frequencies
            
        Returns:
            CodonStateSpace for specified genetic code
        """
        return cls(GeneticCode.from_name(code_name), codon_frequencies)
    
    @classmethod
    def from_alignment_frequencies(
        cls,
        alignment: Any,
        genetic_code: Optional[GeneticCode] = None,
    ) -> "CodonStateSpace":
        """
        Create codon state space with frequencies estimated from alignment.
        
        Args:
            alignment: Codon alignment data
            genetic_code: Optional genetic code (default: Universal)
            
        Returns:
            CodonStateSpace with empirical frequencies
        """
        if genetic_code is None:
            genetic_code = GeneticCode.universal()
        
        # TODO: Implement frequency estimation from alignment
        # For now, return uniform frequencies
        n_codons = len(genetic_code.sense_codons)
        frequencies = np.ones(n_codons) / n_codons
        
        return cls(genetic_code, frequencies)
    
    def codon(self, index: int) -> str:
        """Get codon string for state index."""
        return self.index_to_codon[index]
    
    def index(self, codon: str) -> int:
        """Get state index for codon string."""
        return self.codon_to_index[codon.upper()]
    
    def amino_acid(self, index: int) -> str:
        """Get amino acid for state index."""
        codon = self.index_to_codon[index]
        return self.genetic_code.translate(codon)
    
    def is_synonymous(self, i: int, j: int) -> bool:
        """
        Check if transition i→j is synonymous.
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            True if codons encode same amino acid
        """
        codon_i = self.index_to_codon[i]
        codon_j = self.index_to_codon[j]
        return self.genetic_code.is_synonymous(codon_i, codon_j)
    
    def is_single_nucleotide_change(self, i: int, j: int) -> bool:
        """
        Check if transition i→j involves exactly one nucleotide change.
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            True if single nucleotide change
        """
        codon_i = self.index_to_codon[i]
        codon_j = self.index_to_codon[j]
        return self.genetic_code.is_single_nucleotide_change(codon_i, codon_j)
    
    def nucleotide_change(self, i: int, j: int) -> Optional[tuple]:
        """
        Get nucleotide change for transition i→j.
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            (position, from_nt, to_nt) tuple, or None if not single-nt change
        """
        codon_i = self.index_to_codon[i]
        codon_j = self.index_to_codon[j]
        
        diffs = self.genetic_code.nucleotide_differences(codon_i, codon_j)
        
        if len(diffs) == 1:
            pos, from_nt, to_nt = diffs[0]
            return (pos, from_nt, to_nt)
        
        return None
    
    def is_transition(self, i: int, j: int) -> bool:
        """
        Check if transition i→j is a nucleotide transition (vs transversion).
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            True if transition (A↔G or C↔T), False if transversion or multiple changes
        """
        change = self.nucleotide_change(i, j)
        if change is None:
            return False
        
        _, from_nt, to_nt = change
        return self.genetic_code.is_transition(from_nt, to_nt)
    
    @property
    def frequencies(self) -> np.ndarray:
        """
        Get codon equilibrium frequencies.
        
        Returns uniform frequencies if not specified.
        """
        if self._frequencies is not None:
            return self._frequencies
        return np.ones(self.dimension) / self.dimension
    
    @frequencies.setter
    def frequencies(self, value: np.ndarray):
        """Set codon frequencies."""
        if len(value) != self.dimension:
            raise ValueError(f"Expected {self.dimension} frequencies, got {len(value)}")
        self._frequencies = value / value.sum()  # Normalize
    
    def __repr__(self) -> str:
        return f"CodonStateSpace(code={self.genetic_code.name}, n={self.dimension})"
