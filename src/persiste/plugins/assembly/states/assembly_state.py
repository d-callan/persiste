"""
Assembly state representation.

States are compositional equivalence classes, not specific molecules.
"""

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class AssemblyState:
    """
    Compositional state in assembly theory.

    Represents an equivalence class of molecular assemblies,
    not a specific molecule. This keeps the state space tractable.

    Key properties:
    - Immutable (frozen dataclass)
    - Hashable (can be dict keys, set members)
    - Compositional (multiset of parts, not molecular graph)
    - Compact (assembly depth instead of full history)

    Attributes:
        parts: Frozen multiset of building blocks (as tuple of (part, count) pairs)
        assembly_depth: Integer assembly index proxy
        motifs: Optional structural motif labels

    Examples:
        >>> # Simple composition
        >>> s1 = AssemblyState.from_parts(['A', 'B', 'C'], depth=2)

        >>> # Multiset (repeated parts)
        >>> s2 = AssemblyState.from_parts(['A', 'A', 'B'], depth=3)

        >>> # With motif labels
        >>> s3 = AssemblyState.from_parts(['peptide']*5, depth=4, motifs={'helix'})
    """

    parts: tuple[tuple[str, int], ...]  # Sorted tuple of (part, count) for hashing
    assembly_depth: int
    motifs: frozenset[str] = frozenset()

    def __post_init__(self):
        """Validate state invariants."""
        if self.assembly_depth < 0:
            raise ValueError(f"Assembly depth must be non-negative, got {self.assembly_depth}")

        if not self.parts and self.assembly_depth > 0:
            raise ValueError("Empty state must have depth 0")

    @classmethod
    def from_parts(
        cls,
        parts: list[str],
        depth: int,
        motifs: set[str] | None = None,
    ) -> 'AssemblyState':
        """
        Create state from list of parts.

        Args:
            parts: List of building block identifiers (can have duplicates)
            depth: Assembly depth
            motifs: Optional set of motif labels

        Returns:
            AssemblyState with parts as sorted frozen multiset
        """
        # Count parts and convert to sorted tuple for hashing
        counter = Counter(parts)
        parts_tuple = tuple(sorted(counter.items()))

        return cls(
            parts=parts_tuple,
            assembly_depth=depth,
            motifs=frozenset(motifs or set()),
        )

    @classmethod
    def empty(cls) -> 'AssemblyState':
        """Create empty state (depth 0, no parts)."""
        return cls(parts=tuple(), assembly_depth=0, motifs=frozenset())

    def get_parts_dict(self) -> dict[str, int]:
        """Get parts as dictionary {part: count}."""
        return dict(self.parts)

    def get_parts_list(self) -> list[str]:
        """Get parts as list (with duplicates)."""
        result = []
        for part, count in self.parts:
            result.extend([part] * count)
        return result

    def total_parts(self) -> int:
        """Total number of parts (with multiplicity)."""
        return sum(count for _, count in self.parts)

    @property
    def size(self) -> int:
        """Alias for total_parts (for compatibility)."""
        return self.total_parts()

    def contains_part(self, part: str) -> bool:
        """Check if state contains a specific part."""
        return any(p == part for p, _ in self.parts)

    def contains_motif(self, motif: str) -> bool:
        """Check if state has a specific motif."""
        return motif in self.motifs

    def is_subassembly_of(self, other: 'AssemblyState') -> bool:
        """
        Check if this state is a subassembly of another.

        True if all parts of this state are contained in other.
        """
        self_dict = self.get_parts_dict()
        other_dict = other.get_parts_dict()

        for part, count in self_dict.items():
            if other_dict.get(part) != count:
                return False

        return True

    def __str__(self) -> str:
        """Human-readable representation."""
        parts_str = ', '.join(f"{p}×{c}" if c > 1 else p for p, c in self.parts)
        motifs_str = f" [{', '.join(self.motifs)}]" if self.motifs else ""
        return f"State(d={self.assembly_depth}: {parts_str}{motifs_str})"

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other: 'AssemblyState') -> bool:
        """
        Ordering for search algorithms.

        Order by: depth, then total parts, then lexicographic on parts.
        """
        if self.assembly_depth != other.assembly_depth:
            return self.assembly_depth < other.assembly_depth

        if self.total_parts() != other.total_parts():
            return self.total_parts() < other.total_parts()

        return self.parts < other.parts
