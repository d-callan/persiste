"""Codon transition graph for phylogenetic analysis."""

from typing import Optional, Iterator, Tuple
from persiste.core.transitions import TransitionGraph
from persiste.plugins.phylo.states.codons import CodonStateSpace
from persiste.plugins.phylo.states.genetic_code import GeneticCode


class CodonTransitionGraph(TransitionGraph):
    """
    Transition graph for codon substitutions.
    
    Only allows single-nucleotide changes between sense codons.
    This is the standard assumption in codon models (MG94, GY94).
    
    The graph structure encodes:
    - Which transitions are possible (single-nt changes only)
    - Which are synonymous vs nonsynonymous
    - Which are transitions vs transversions
    
    In PERSISTE terms:
    - This defines the "opportunity" structure for codon evolution
    - The Baseline (MG94) defines rates for allowed transitions
    - The ConstraintModel (ω/θ) modifies nonsynonymous rates
    
    Attributes:
        codon_space: CodonStateSpace defining states
        _adjacency: Cached adjacency information
    """
    
    def __init__(self, codon_space: CodonStateSpace):
        """
        Initialize codon transition graph.
        
        Args:
            codon_space: CodonStateSpace defining the states
        """
        self.codon_space = codon_space
        self._adjacency = self._build_adjacency()
        
        # Initialize base TransitionGraph
        super().__init__(
            states=codon_space,
            edges=list(self._iter_edges()),
        )
    
    @classmethod
    def universal(cls) -> "CodonTransitionGraph":
        """Create codon graph with universal genetic code."""
        return cls(CodonStateSpace.universal())
    
    @classmethod
    def from_genetic_code(cls, code_name: str) -> "CodonTransitionGraph":
        """Create codon graph from named genetic code."""
        return cls(CodonStateSpace.from_genetic_code(code_name))
    
    def _build_adjacency(self) -> dict:
        """Build adjacency list for efficient lookup."""
        adj = {}
        n = self.codon_space.dimension
        
        for i in range(n):
            adj[i] = set()
            for j in range(n):
                if i != j and self.codon_space.is_single_nucleotide_change(i, j):
                    adj[i].add(j)
        
        return adj
    
    def _iter_edges(self) -> Iterator[Tuple[int, int]]:
        """Iterate over all edges (single-nt codon changes)."""
        for i, neighbors in self._adjacency.items():
            for j in neighbors:
                yield (i, j)
    
    def allows(self, i: int, j: int) -> bool:
        """
        Check if transition i→j is allowed.
        
        Only single-nucleotide changes between sense codons are allowed.
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            True if transition is allowed (single-nt change)
        """
        if i == j:
            return False
        return j in self._adjacency.get(i, set())
    
    def neighbors(self, i: int) -> Iterator[int]:
        """
        Get all states reachable from state i.
        
        Args:
            i: Source state index
            
        Yields:
            Target state indices
        """
        yield from self._adjacency.get(i, set())
    
    def is_synonymous(self, i: int, j: int) -> bool:
        """
        Check if transition i→j is synonymous.
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            True if same amino acid, False otherwise
        """
        return self.codon_space.is_synonymous(i, j)
    
    def is_nonsynonymous(self, i: int, j: int) -> bool:
        """
        Check if transition i→j is nonsynonymous.
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            True if different amino acid, False otherwise
        """
        if not self.allows(i, j):
            return False
        return not self.codon_space.is_synonymous(i, j)
    
    def is_transition(self, i: int, j: int) -> bool:
        """
        Check if transition i→j is a nucleotide transition.
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            True if A↔G or C↔T change
        """
        return self.codon_space.is_transition(i, j)
    
    def is_transversion(self, i: int, j: int) -> bool:
        """
        Check if transition i→j is a nucleotide transversion.
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            True if purine↔pyrimidine change
        """
        if not self.allows(i, j):
            return False
        return not self.codon_space.is_transition(i, j)
    
    def synonymous_edges(self) -> Iterator[Tuple[int, int]]:
        """Iterate over all synonymous transitions."""
        for i, j in self._iter_edges():
            if self.is_synonymous(i, j):
                yield (i, j)
    
    def nonsynonymous_edges(self) -> Iterator[Tuple[int, int]]:
        """Iterate over all nonsynonymous transitions."""
        for i, j in self._iter_edges():
            if self.is_nonsynonymous(i, j):
                yield (i, j)
    
    def count_edges(self) -> dict:
        """
        Count edges by type.
        
        Returns:
            Dict with counts of synonymous, nonsynonymous, transition, transversion
        """
        counts = {
            "synonymous": 0,
            "nonsynonymous": 0,
            "transition": 0,
            "transversion": 0,
            "total": 0,
        }
        
        for i, j in self._iter_edges():
            counts["total"] += 1
            
            if self.is_synonymous(i, j):
                counts["synonymous"] += 1
            else:
                counts["nonsynonymous"] += 1
            
            if self.is_transition(i, j):
                counts["transition"] += 1
            else:
                counts["transversion"] += 1
        
        return counts
    
    def __repr__(self) -> str:
        counts = self.count_edges()
        return (
            f"CodonTransitionGraph("
            f"codons={self.codon_space.dimension}, "
            f"edges={counts['total']}, "
            f"syn={counts['synonymous']}, "
            f"nonsyn={counts['nonsynonymous']})"
        )
