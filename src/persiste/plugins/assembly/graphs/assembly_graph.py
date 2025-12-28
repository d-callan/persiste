"""
Lazy assembly graph with truncation and pruning.

Only generates states and transitions as needed.
Scales sublinearly in state space.
"""

from typing import List, Tuple, Dict, Set, Optional
from itertools import combinations
import numpy as np

from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import (
    AssemblyBaseline, TransitionType
)
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint


class AssemblyGraph:
    """
    Lazy assembly graph with truncation and pruning.
    
    Key features:
    - Lazy generation: Only create states when needed
    - Truncation: Max depth limit
    - Pruning: Ignore negligible rates
    - Caching: Reuse computed neighbors
    
    This is critical for scalability - we never enumerate the full state space.
    
    Attributes:
        primitives: List of primitive building blocks
        max_depth: Maximum assembly depth to explore
        min_rate_threshold: Minimum rate to consider (pruning)
    """
    
    def __init__(
        self,
        primitives: List[str],
        max_depth: int = 10,
        min_rate_threshold: float = 1e-6,
    ):
        """
        Initialize lazy assembly graph.
        
        Args:
            primitives: List of primitive building block identifiers
            max_depth: Maximum assembly depth (default: 10)
            min_rate_threshold: Minimum rate for pruning (default: 1e-6)
        """
        self.primitives = primitives
        self.max_depth = max_depth
        self.min_rate = min_rate_threshold
        
        # Caches
        self._state_cache: Dict[Tuple, AssemblyState] = {}
        self._neighbor_cache: Dict[AssemblyState, List[Tuple[AssemblyState, float, TransitionType]]] = {}
        
        # Create primitive states (depth 0)
        self._primitive_states = [
            AssemblyState.from_parts([p], depth=0) for p in primitives
        ]
    
    def get_primitive_states(self) -> List[AssemblyState]:
        """Get primitive building block states."""
        return self._primitive_states
    
    def get_neighbors(
        self,
        state: AssemblyState,
        baseline: AssemblyBaseline,
        constraint: AssemblyConstraint,
    ) -> List[Tuple[AssemblyState, float, TransitionType]]:
        """
        Generate neighbors on demand with effective rates.
        
        Returns list of (neighbor_state, effective_rate, transition_type) tuples.
        Only includes neighbors with rate >= min_rate_threshold.
        
        Args:
            state: Current assembly state
            baseline: Baseline model for rates
            constraint: Constraint model for bonuses
            
        Returns:
            List of (neighbor, rate, type) tuples
        """
        # Check cache
        cache_key = (state, id(baseline), id(constraint))
        if state in self._neighbor_cache:
            return self._neighbor_cache[state]
        
        neighbors = []
        
        # Generate join transitions (if not at max depth)
        if state.assembly_depth < self.max_depth:
            neighbors.extend(self._generate_joins(state, baseline, constraint))
        
        # Generate split transitions (if depth > 0)
        if state.assembly_depth > 0:
            neighbors.extend(self._generate_splits(state, baseline, constraint))
        
        # Generate decay transition
        decay_neighbor = self._generate_decay(state, baseline, constraint)
        if decay_neighbor is not None:
            neighbors.append(decay_neighbor)
        
        # Cache and return
        self._neighbor_cache[state] = neighbors
        return neighbors
    
    def _generate_joins(
        self,
        state: AssemblyState,
        baseline: AssemblyBaseline,
        constraint: AssemblyConstraint,
    ) -> List[Tuple[AssemblyState, float, TransitionType]]:
        """
        Generate join transitions: state + partner → joined_state
        
        For simplicity, we allow joining with any primitive or existing state.
        In practice, you'd limit this based on chemistry/physics.
        """
        joins = []
        
        # Join with primitives
        for primitive in self._primitive_states:
            target = self._join_states(state, primitive)
            if target.assembly_depth <= self.max_depth:
                rate = self._effective_rate(state, target, TransitionType.JOIN, baseline, constraint)
                if rate >= self.min_rate:
                    joins.append((target, rate, TransitionType.JOIN))
        
        # Could also join with other assembled states, but that explodes the space
        # For now, keep it simple: only join with primitives
        
        return joins
    
    def _generate_splits(
        self,
        state: AssemblyState,
        baseline: AssemblyBaseline,
        constraint: AssemblyConstraint,
    ) -> List[Tuple[AssemblyState, float, TransitionType]]:
        """
        Generate split transitions: state → component1 + component2
        
        For simplicity, we split off single primitives.
        Full implementation would enumerate all valid splits.
        """
        splits = []
        
        parts_dict = state.get_parts_dict()
        
        # Try removing each part type
        for part, count in parts_dict.items():
            if count > 0:
                # Create state with one less of this part
                new_parts = parts_dict.copy()
                new_parts[part] = count - 1
                if new_parts[part] == 0:
                    del new_parts[part]
                
                # Convert back to list
                new_parts_list = []
                for p, c in new_parts.items():
                    new_parts_list.extend([p] * c)
                
                # Create target state (depth decreases by 1)
                target = AssemblyState.from_parts(
                    new_parts_list,
                    depth=max(0, state.assembly_depth - 1)
                )
                
                rate = self._effective_rate(state, target, TransitionType.SPLIT, baseline, constraint)
                if rate >= self.min_rate:
                    splits.append((target, rate, TransitionType.SPLIT))
        
        return splits
    
    def _generate_decay(
        self,
        state: AssemblyState,
        baseline: AssemblyBaseline,
        constraint: AssemblyConstraint,
    ) -> Optional[Tuple[AssemblyState, float, TransitionType]]:
        """
        Generate decay transition: state → ∅
        """
        empty = AssemblyState.empty()
        rate = self._effective_rate(state, empty, TransitionType.DECAY, baseline, constraint)
        
        if rate >= self.min_rate:
            return (empty, rate, TransitionType.DECAY)
        return None
    
    def _join_states(self, s1: AssemblyState, s2: AssemblyState) -> AssemblyState:
        """
        Join two states into one.
        
        Parts are combined (multiset union).
        Depth increases by 1.
        """
        # Combine parts
        parts1 = s1.get_parts_dict()
        parts2 = s2.get_parts_dict()
        
        combined = parts1.copy()
        for part, count in parts2.items():
            combined[part] = combined.get(part, 0) + count
        
        # Convert to list
        parts_list = []
        for part, count in combined.items():
            parts_list.extend([part] * count)
        
        # New depth is max of inputs + 1
        new_depth = max(s1.assembly_depth, s2.assembly_depth) + 1
        
        # Combine motifs (if any)
        combined_motifs = s1.motifs | s2.motifs
        
        return AssemblyState.from_parts(parts_list, new_depth, combined_motifs)
    
    def _effective_rate(
        self,
        source: AssemblyState,
        target: AssemblyState,
        transition_type: TransitionType,
        baseline: AssemblyBaseline,
        constraint: AssemblyConstraint,
    ) -> float:
        """
        Compute effective transition rate.
        
        λ_eff = λ_baseline × exp(C)
        """
        base_rate = baseline.get_assembly_rate(source, target, transition_type)
        constraint_contrib = constraint.constraint_contribution(source, target, transition_type)
        
        return base_rate * np.exp(constraint_contrib)
    
    def count_reachable_states(
        self,
        start_state: AssemblyState,
        baseline: AssemblyBaseline,
        constraint: AssemblyConstraint,
        max_states: int = 1000,
    ) -> int:
        """
        Count reachable states from start (BFS).
        
        Useful for understanding graph size.
        """
        visited = {start_state}
        queue = [start_state]
        
        while queue and len(visited) < max_states:
            current = queue.pop(0)
            neighbors = self.get_neighbors(current, baseline, constraint)
            
            for neighbor, _, _ in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited)
    
    def __str__(self) -> str:
        return (
            f"AssemblyGraph("
            f"primitives={self.primitives}, "
            f"max_depth={self.max_depth}, "
            f"min_rate={self.min_rate:.2e}, "
            f"cached_states={len(self._neighbor_cache)})"
        )
