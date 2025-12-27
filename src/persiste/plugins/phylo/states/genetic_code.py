"""Genetic code definitions using BioPython's NCBI tables."""

from typing import List, Tuple, Optional

try:
    from Bio.Data import CodonTable
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    CodonTable = None


# NCBI genetic code ID mapping
# See: https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi
NCBI_CODE_NAMES = {
    "Universal": 1,
    "Standard": 1,
    "Vertebrate-mtDNA": 2,
    "Yeast-mtDNA": 3,
    "Mold-mtDNA": 4,
    "Invertebrate-mtDNA": 5,
    "Ciliate": 6,
    "Echinoderm-mtDNA": 9,
    "Euplotid": 10,
    "Bacterial": 11,
    "Alt-Yeast": 12,
    "Ascidian-mtDNA": 13,
    "Alt-Flatworm-mtDNA": 14,
    "Chlorophycean-mtDNA": 16,
    "Trematode-mtDNA": 21,
    "Scenedesmus-mtDNA": 22,
    "Thraustochytrium-mtDNA": 23,
    "Pterobranchia-mtDNA": 24,
    "Gracilibacteria": 25,
    "Pachysolen": 26,
    "Karyorelict": 27,
    "Condylostoma": 28,
    "Mesodinium": 29,
    "Peritrich": 30,
    "Blastocrithidia": 31,
}


class GeneticCode:
    """
    Genetic code wrapper around BioPython's NCBI codon tables.
    
    Provides a consistent interface to NCBI genetic codes with utilities for:
    - Identifying sense vs stop codons
    - Checking synonymous vs nonsynonymous changes
    - Nucleotide position and type of changes
    
    Uses BioPython's Bio.Data.CodonTable as the authoritative source,
    which is automatically updated from NCBI standards.
    
    Attributes:
        name: Genetic code name (e.g., "Universal", "Vertebrate-mtDNA")
        ncbi_id: NCBI genetic code table ID (1-31+)
        _table: BioPython CodonTable object
    """
    
    def __init__(self, name: str = "Universal", ncbi_id: Optional[int] = None):
        """
        Initialize genetic code.
        
        Args:
            name: Genetic code name (default: "Universal")
            ncbi_id: Optional NCBI ID override (takes precedence over name)
            
        Raises:
            ImportError: If BioPython not installed
            ValueError: If name/ID not recognized
        """
        if not BIOPYTHON_AVAILABLE:
            raise ImportError(
                "BioPython required for GeneticCode. "
                "Install with: pip install biopython"
            )
        
        self.name = name
        
        # Resolve NCBI ID
        if ncbi_id is not None:
            self.ncbi_id = ncbi_id
        elif name in NCBI_CODE_NAMES:
            self.ncbi_id = NCBI_CODE_NAMES[name]
        else:
            raise ValueError(
                f"Unknown genetic code: {name}. "
                f"Available: {list(NCBI_CODE_NAMES.keys())}"
            )
        
        # Load BioPython table
        try:
            self._table = CodonTable.unambiguous_dna_by_id[self.ncbi_id]
        except KeyError:
            raise ValueError(
                f"NCBI genetic code ID {self.ncbi_id} not found. "
                f"Available IDs: {list(CodonTable.unambiguous_dna_by_id.keys())}"
            )
        
        # Cache sense/stop codons
        self._sense_codons = None
        self._stop_codons = None
    
    @classmethod
    def universal(cls) -> "GeneticCode":
        """Create universal (standard) genetic code (NCBI ID 1)."""
        return cls(name="Universal")
    
    @classmethod
    def vertebrate_mtdna(cls) -> "GeneticCode":
        """Create vertebrate mitochondrial genetic code (NCBI ID 2)."""
        return cls(name="Vertebrate-mtDNA")
    
    @classmethod
    def from_name(cls, name: str) -> "GeneticCode":
        """
        Create genetic code by name.
        
        Args:
            name: Genetic code name (see NCBI_CODE_NAMES for options)
            
        Returns:
            GeneticCode instance
        """
        return cls(name=name)
    
    @classmethod
    def from_ncbi_id(cls, ncbi_id: int) -> "GeneticCode":
        """
        Create genetic code by NCBI ID.
        
        Args:
            ncbi_id: NCBI genetic code table ID (1-31+)
            
        Returns:
            GeneticCode instance
        """
        return cls(name=f"NCBI-{ncbi_id}", ncbi_id=ncbi_id)
    
    @property
    def sense_codons(self) -> List[str]:
        """Get list of sense (non-stop) codons."""
        if self._sense_codons is None:
            self._sense_codons = sorted([
                codon for codon in self._table.forward_table.keys()
            ])
        return self._sense_codons
    
    @property
    def stop_codons(self) -> List[str]:
        """Get list of stop codons."""
        if self._stop_codons is None:
            self._stop_codons = sorted(self._table.stop_codons)
        return self._stop_codons
    
    def translate(self, codon: str) -> str:
        """
        Translate codon to amino acid.
        
        Args:
            codon: Three-letter codon string (e.g., "ATG")
            
        Returns:
            Single-letter amino acid code (or '*' for stop)
        """
        codon_upper = codon.upper()
        if codon_upper in self._table.stop_codons:
            return '*'
        return self._table.forward_table.get(codon_upper, 'X')  # X for unknown
    
    def is_sense(self, codon: str) -> bool:
        """Check if codon is a sense codon (not stop)."""
        return codon.upper() not in self._table.stop_codons
    
    def is_stop(self, codon: str) -> bool:
        """Check if codon is a stop codon."""
        return codon.upper() in self._table.stop_codons
    
    def is_synonymous(self, codon1: str, codon2: str) -> bool:
        """
        Check if two codons encode the same amino acid.
        
        Args:
            codon1: First codon
            codon2: Second codon
            
        Returns:
            True if synonymous (same amino acid), False otherwise
        """
        return self.translate(codon1) == self.translate(codon2)
    
    def nucleotide_differences(
        self, codon1: str, codon2: str
    ) -> List[Tuple[int, str, str]]:
        """
        Find nucleotide differences between two codons.
        
        Args:
            codon1: First codon
            codon2: Second codon
            
        Returns:
            List of (position, from_nt, to_nt) tuples
        """
        c1, c2 = codon1.upper(), codon2.upper()
        diffs = []
        
        for i in range(3):
            if c1[i] != c2[i]:
                diffs.append((i, c1[i], c2[i]))
        
        return diffs
    
    def is_single_nucleotide_change(self, codon1: str, codon2: str) -> bool:
        """Check if codons differ by exactly one nucleotide."""
        return len(self.nucleotide_differences(codon1, codon2)) == 1
    
    def is_transition(self, nt1: str, nt2: str) -> bool:
        """
        Check if nucleotide change is a transition (purine↔purine or pyrimidine↔pyrimidine).
        
        Transitions: A↔G, C↔T
        Transversions: A↔C, A↔T, G↔C, G↔T
        """
        transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
        return (nt1.upper(), nt2.upper()) in transitions
    
    def is_transversion(self, nt1: str, nt2: str) -> bool:
        """Check if nucleotide change is a transversion."""
        return not self.is_transition(nt1, nt2) and nt1.upper() != nt2.upper()
    
    def __len__(self) -> int:
        """Return number of sense codons."""
        return len(self.sense_codons)
    
    def __repr__(self) -> str:
        return f"GeneticCode(name={self.name}, ncbi_id={self.ncbi_id}, sense_codons={len(self.sense_codons)})"
