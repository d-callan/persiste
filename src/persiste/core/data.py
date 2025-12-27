"""Data structures for observed transitions."""

from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ObservedTransitions:
    """
    Observed transition data.
    
    Packages transition observations separate from models.
    What counts as a "transition" depends on regime:
    - Genomics: substitutions along branches (latent, inferred)
    - Metagenomics: abundance changes (aggregate, noisy, compositional)
    - Assembly chemistry: assembly steps (partial, censored)
    
    Attributes:
        counts: Transition counts as {(source_idx, target_idx): count}
        exposure: Optional scaling factor (time, opportunity, branch length, etc.)
        metadata: Domain-specific metadata (replicates, conditions, etc.)
    """
    
    counts: Dict[Tuple[int, int], int]
    exposure: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate counts are non-negative."""
        for (i, j), count in self.counts.items():
            if count < 0:
                raise ValueError(f"Negative count {count} for transition ({i}, {j})")
    
    @classmethod
    def from_dict(
        cls,
        counts: Dict[Tuple[int, int], int],
        exposure: Optional[float] = None,
        **metadata
    ) -> "ObservedTransitions":
        """
        Create from transition count dictionary.
        
        Args:
            counts: {(i, j): count} dictionary
            exposure: Optional exposure/time/opportunity
            **metadata: Additional metadata as keyword args
            
        Returns:
            ObservedTransitions instance
        """
        return cls(counts=counts, exposure=exposure, metadata=metadata)
    
    def total_transitions(self) -> int:
        """Total number of observed transitions."""
        return sum(self.counts.values())
    
    def __repr__(self) -> str:
        n_transitions = len(self.counts)
        total = self.total_transitions()
        return f"ObservedTransitions(transitions={n_transitions}, total_count={total})"
