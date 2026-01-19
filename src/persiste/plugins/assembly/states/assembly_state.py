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

        # Use a private attribute to store the ID for the lazy stable_id property
        object.__setattr__(self, "_stable_id_val", 0)

    @property
    def stable_id(self) -> int:
        """
        Get the 64-bit integer ID for this state, synchronized with Rust.
        
        This property is lazy-loaded because it requires calling into the Rust backend
        to ensure the identifier matches what the simulation uses.
        """
        cached_id = object.__getattribute__(self, "_stable_id_val")
        if cached_id != 0:
            return cached_id

        # Use a minimal simulation to force Rust to compute the deterministic ID
        # This is the "source of truth" strategy.
        import persiste_rust
        
        # We need primitives to resolve the ID, but for ID computation 
        # any list containing the parts will work.
        primitives = list(self.get_parts_dict().keys())
        if not primitives:
            primitives = ["A"] # Fallback for empty state

        results = persiste_rust.simulate_assembly_trajectories(
            primitives,
            self.get_parts_list(),
            {}, # theta
            1,  # n_samples
            0.0, # t_max
            0.0, # burn_in
            self.assembly_depth + 1, # max_depth
            1,  # seed
            1.0, # kappa
            0.0, # join_exponent
            0.0, # split_exponent
            0.0, # decay_rate
        )
        
        rust_id = results["paths"][0]["final_state_id"]
        object.__setattr__(self, "_stable_id_val", rust_id)
        return rust_id

    def __hash__(self) -> int:
        """Use stable_id for hashing to ensure consistency across processes/languages."""
        return hash(self.stable_id)

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
