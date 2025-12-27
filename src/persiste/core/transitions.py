"""Transition graph structures."""

from typing import List, Tuple, Optional, Callable, Iterator, TYPE_CHECKING, Any
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from persiste.core.states import StateSpace


@dataclass
class TransitionGraph:
    """
    Structural definition of allowed transitions between states.
    
    Represents which transitions are possible, not their rates or probabilities.
    Supports both explicit (enumerated edges) and implicit (adjacency function) graphs.
    
    Attributes:
        states: State space
        adjacency_fn: Function(i, j) -> bool indicating if i->j is allowed
        edges: Pre-enumerated list of (from_state_idx, to_state_idx) tuples
        metadata: Domain-specific metadata
    """
    
    states: "StateSpace"
    adjacency_fn: Optional[Callable[[int, int], bool]] = None
    edges: Optional[List[Tuple[int, int]]] = None
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate that at least one representation is provided."""
        if self.adjacency_fn is None and self.edges is None:
            raise ValueError("Must provide either adjacency_fn or edges")
    
    def allows(self, i: int, j: int) -> bool:
        """
        Check if transition i -> j is allowed.
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            True if transition is allowed
        """
        if self.adjacency_fn is not None:
            return self.adjacency_fn(i, j)
        elif self.edges is not None:
            return (i, j) in self.edges
        else:
            raise RuntimeError("No transition structure available")
    
    def iter_edges(self) -> Iterator[Tuple[int, int]]:
        """
        Iterate over allowed transitions.
        
        For explicit graphs, yields pre-enumerated edges.
        For implicit graphs, generates edges on-demand.
        
        Yields:
            (source_idx, target_idx) tuples
        """
        if self.edges is not None:
            yield from self.edges
        elif self.adjacency_fn is not None:
            # Generate edges on-demand from adjacency function
            for i in range(len(self.states)):
                for j in range(len(self.states)):
                    if i != j and self.adjacency_fn(i, j):
                        yield (i, j)
        else:
            raise RuntimeError("No transition structure available")
    
    @classmethod
    def from_edges(
        cls,
        states: "StateSpace",
        edges: List[Tuple[int, int]],
        metadata: Optional[dict] = None,
    ) -> "TransitionGraph":
        """
        Create transition graph from explicit edge list.
        
        Args:
            states: State space
            edges: List of (from_idx, to_idx) tuples
            metadata: Optional domain-specific metadata
            
        Returns:
            TransitionGraph with explicit edges
        """
        return cls(states=states, edges=edges, metadata=metadata or {})
    
    @classmethod
    def from_adjacency(
        cls,
        states: "StateSpace",
        adjacency_fn: Callable[[int, int], bool],
        metadata: Optional[dict] = None,
    ) -> "TransitionGraph":
        """
        Create transition graph from adjacency function (implicit/lazy).
        
        Args:
            states: State space
            adjacency_fn: Function(i, j) -> bool indicating if i->j is allowed
            metadata: Optional domain-specific metadata
            
        Returns:
            TransitionGraph with implicit structure
        """
        return cls(states=states, adjacency_fn=adjacency_fn, metadata=metadata or {})
    
    @classmethod
    def complete(cls, states: "StateSpace") -> "TransitionGraph":
        """
        Create complete graph (all transitions allowed).
        
        Uses adjacency function for efficiency with large state spaces.
        
        Args:
            states: State space
            
        Returns:
            TransitionGraph allowing all i->j where i != j
        """
        return cls(
            states=states,
            adjacency_fn=lambda i, j: i != j,
            metadata={"type": "complete"}
        )
    
    def to_networkx(self) -> Any:
        """
        Convert to NetworkX DiGraph (optional, requires networkx).
        
        Returns:
            NetworkX DiGraph
            
        Raises:
            ImportError: If networkx not installed
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX required for to_networkx(). Install with: pip install networkx"
            )
        
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(self.states)))
        graph.add_edges_from(self.iter_edges())
        return graph
    
    def __repr__(self) -> str:
        if self.edges is not None:
            return f"TransitionGraph(states={len(self.states)}, edges={len(self.edges)})"
        else:
            return f"TransitionGraph(states={len(self.states)}, implicit)"
