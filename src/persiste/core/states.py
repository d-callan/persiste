"""State space definitions."""

from typing import Any, List, Optional, Callable, Iterator
from dataclasses import dataclass, field


@dataclass
class StateSpace:
    """
    Abstract state space representation for generalized constraint detection.
    
    Supports multiple representations:
    - Enumerated: small state spaces with explicit list
    - Lazy/generator: large/combinatorial spaces with on-demand enumeration
    - Symbolic: implicit state spaces defined by rules
    
    Attributes:
        states: Pre-enumerated state identifiers (for small spaces)
        dimension: Number of states (if known; None for implicit/infinite spaces)
        metadata: Domain-specific metadata (not interpreted by base class)
        generator: Lazy state enumeration function
    """
    
    states: Optional[List[Any]] = None
    dimension: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    generator: Optional[Callable[[], Iterator[Any]]] = None
    
    def __post_init__(self):
        """Infer dimension from states if not provided."""
        if self.dimension is None and self.states is not None:
            self.dimension = len(self.states)
    
    def iter_states(self) -> Iterator[Any]:
        """
        Iterate over states.
        
        Uses pre-enumerated list if available, otherwise generator.
        
        Yields:
            State identifiers
            
        Raises:
            NotImplementedError: If neither states nor generator defined
        """
        if self.states is not None:
            yield from self.states
        elif self.generator is not None:
            yield from self.generator()
        else:
            raise NotImplementedError("State enumeration not defined")
    
    @classmethod
    def from_list(cls, states: List[Any], metadata: Optional[dict] = None) -> "StateSpace":
        """
        Create state space from explicit list.
        
        Args:
            states: List of state identifiers
            metadata: Optional domain-specific metadata
            
        Returns:
            StateSpace with pre-enumerated states
        """
        return cls(states=states, metadata=metadata or {})
    
    @classmethod
    def from_types(
        cls,
        types: List[Any],
        enumerator: Optional[Callable[[List[Any]], Iterator[Any]]] = None,
        metadata: Optional[dict] = None,
    ) -> "StateSpace":
        """
        Create state space from basic types.
        
        For simple cases (enumerator=None), types become states directly.
        For complex cases, enumerator generates derived states lazily.
        
        Args:
            types: Basic type identifiers
            enumerator: Optional function mapping types to state iterator
            metadata: Optional domain-specific metadata
            
        Returns:
            StateSpace object
            
        Examples:
            # Simple: types are states
            >>> StateSpace.from_types(['A', 'B', 'C'])
            
            # Complex: lazy enumeration
            >>> def assembly_enumerator(types):
            ...     # Generate assemblies on-demand
            ...     for t in types:
            ...         yield t
            ...     for t1, t2 in combinations(types, 2):
            ...         yield f"{t1}+{t2}"
            >>> StateSpace.from_types(['α', 'β'], enumerator=assembly_enumerator)
        """
        metadata = metadata or {}
        
        if enumerator is None:
            # Simple case: types are states
            return cls(states=types, metadata=metadata)
        else:
            # Complex case: lazy enumeration
            return cls(
                generator=lambda: enumerator(types),
                metadata=metadata
            )
    
    @classmethod
    def from_generator(
        cls,
        generator: Callable[[], Iterator[Any]],
        dimension: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> "StateSpace":
        """
        Create state space with lazy enumeration.
        
        Args:
            generator: Function returning state iterator
            dimension: Optional known dimension
            metadata: Optional domain-specific metadata
            
        Returns:
            StateSpace with generator-based enumeration
        """
        return cls(
            generator=generator,
            dimension=dimension,
            metadata=metadata or {}
        )
    
    def __len__(self) -> int:
        """
        Return dimension if known.
        
        Raises:
            TypeError: If dimension is unknown (infinite/implicit space)
        """
        if self.dimension is None:
            raise TypeError("Dimension unknown for implicit/infinite state space")
        return self.dimension
    
    def __getitem__(self, idx: int) -> Any:
        """
        Get state by index.
        
        Only works for pre-enumerated state spaces.
        
        Args:
            idx: State index
            
        Returns:
            State identifier
            
        Raises:
            TypeError: If states not pre-enumerated
        """
        if self.states is None:
            raise TypeError("Indexing not supported for generator-based state space")
        return self.states[idx]
    
    def __repr__(self) -> str:
        if self.dimension is not None:
            return f"StateSpace(dimension={self.dimension})"
        elif self.states is not None:
            return f"StateSpace(states={len(self.states)})"
        else:
            return "StateSpace(generator-based)"

