# Assembly Plugin Design

## Guiding Constraints (Non-Negotiable)

1. **States must be enumerable or lazily generatable**
2. **Transitions must be local and factorable**
3. **Baseline must be simple, fast, and physics-agnostic**
4. **Constraint carries chemistry, not baseline**
5. **Observation model must tolerate massive missingness**
6. **Inference must scale sublinearly in state space**

---

## 1. State Representation

### Core Principle
**States are compositional equivalence classes, not specific molecules.**

### State Definition
```python
State = (Parts, AssemblyDepth, OptionalMotifs)
```

### Implementation
```python
@dataclass(frozen=True)
class AssemblyState:
    """
    Compositional state in assembly theory.
    
    Represents an equivalence class of molecular assemblies,
    not a specific molecule.
    """
    parts: FrozenMultiset[str]  # Multiset of building blocks
    assembly_depth: int          # Assembly index proxy
    motifs: FrozenSet[str] = frozenset()  # Optional structural motifs
    
    def __hash__(self):
        return hash((self.parts, self.assembly_depth, self.motifs))
```

### Examples
```python
# Simple composition
State(parts={'A', 'B', 'C'}, depth=2)

# Multiset (repeated parts)
State(parts={'A': 2, 'B': 1}, depth=3)

# With motif labels
State(parts={'peptide': 5}, depth=4, motifs={'helix'})

# Hierarchical
State(parts={'subassembly_X': 2, 'C': 1}, depth=5)
```

### Key Properties
- **Hashable** - Can be used as dict keys, in sets
- **Immutable** - No accidental mutation
- **Compact** - Multiset representation, not full molecular graph
- **Comparable** - Can define ordering for search algorithms

---

## 2. Transition Types

### Primitive Assembly Moves

#### Join
```python
X + Y → X∘Y
```
- Increases assembly depth by 1
- Combines two states into one
- Rate depends on sizes of X and Y

#### Split
```python
X∘Y → X + Y
```
- Decreases assembly depth by 1
- Breaks assembly into components
- Rate increases with assembly depth

#### Decay
```python
X → ∅
```
- Removes state from system
- Rate increases with instability
- Terminal transition

#### Rearrange (Optional, Phase 2)
```python
X∘Y → X′∘Y′
```
- Preserves assembly depth
- Internal reorganization
- Rare under baseline

### Transition Representation
```python
@dataclass
class AssemblyTransition:
    """Primitive assembly move."""
    type: TransitionType  # JOIN, SPLIT, DECAY, REARRANGE
    source: AssemblyState
    target: AssemblyState
    depth_change: int  # ±1 for join/split, 0 for rearrange
    
    def is_valid(self) -> bool:
        """Check conservation laws."""
        if self.type == TransitionType.JOIN:
            # Target parts = union of source parts
            # Target depth = max(source depths) + 1
            pass
        elif self.type == TransitionType.SPLIT:
            # Source parts = union of target parts
            # Source depth = max(target depths) + 1
            pass
```

---

## 3. Baseline Model (Physics-Agnostic)

### Core Principle
**"If chemistry were blind and memoryless, how often would this happen?"**

### Factorized Rate Formula
```python
λ_baseline(i → j) = κ × f(size_i) × g(size_j) × h(type)
```

Where:
- `κ` - Global rate constant
- `f(size_i)` - Source size factor
- `g(size_j)` - Target size factor  
- `h(type)` - Transition type factor

### Implementation
```python
class AssemblyBaseline(Baseline):
    """
    Physics-agnostic baseline for assembly transitions.
    
    Knows nothing about chemistry, catalysis, or "life."
    Pure combinatorics and size effects.
    """
    
    def __init__(
        self,
        kappa: float = 1.0,
        join_exponent: float = -0.5,   # Joins harder with size
        split_exponent: float = 0.3,    # Splits easier with size
        decay_rate: float = 0.01,
    ):
        self.kappa = kappa
        self.join_exp = join_exponent
        self.split_exp = split_exponent
        self.decay_rate = decay_rate
    
    def get_rate(self, source: AssemblyState, target: AssemblyState, 
                 transition_type: TransitionType) -> float:
        """
        Compute baseline transition rate.
        
        No chemistry. No functional groups. No catalysis.
        Pure size and type effects.
        """
        if transition_type == TransitionType.JOIN:
            # Harder to join larger assemblies
            size_factor = (source.assembly_depth ** self.join_exp)
            return self.kappa * size_factor
            
        elif transition_type == TransitionType.SPLIT:
            # Easier to split larger assemblies
            size_factor = (source.assembly_depth ** self.split_exp)
            return self.kappa * size_factor
            
        elif transition_type == TransitionType.DECAY:
            # Constant decay rate
            return self.decay_rate
            
        else:
            return 0.0
```

### Key Properties
- **No chemistry** - Baseline doesn't know about bonds, catalysis, etc.
- **Factorized** - Easy to compute, cache, and reason about
- **Size-dependent** - Larger assemblies behave differently
- **Type-specific** - Join/split/decay have different dynamics

---

## 4. Constraint Model (Assembly Theory Logic)

### Core Principle
**"Which assemblies are preferred or suppressed relative to blind chemistry?"**

### Constraint Parameters θ
```python
θ = {
    'motif_bonus': {'helix': 2.3, 'sheet': 1.8},
    'reuse_bonus': 1.1,           # Recursive assembly advantage
    'depth_penalty': -0.4,         # Complexity cost
    'environmental_fit': 0.5,      # Context-dependent
}
```

### Effective Rate
```python
λ_eff(i → j) = λ_baseline(i → j) × exp(C(i, j; θ))
```

Where `C(i, j; θ)` is the **constraint contribution** (cheap to compute).

### Implementation
```python
class AssemblyConstraint(ConstraintModel):
    """
    Assembly theory constraint model.
    
    Encodes:
    - Motif stability
    - Reusability of subassemblies
    - Environmental compatibility
    - Recursive reuse (core assembly theory)
    """
    
    def __init__(self):
        self.motif_bonuses = {}
        self.reuse_bonus = 0.0
        self.depth_penalty = 0.0
        self.env_fit = 0.0
    
    def constraint_contribution(
        self,
        source: AssemblyState,
        target: AssemblyState,
        transition_type: TransitionType,
    ) -> float:
        """
        Compute C(i → j; θ).
        
        Returns log-scale contribution to rate.
        Positive = favored, negative = suppressed.
        """
        C = 0.0
        
        # Motif bonuses
        for motif in target.motifs:
            C += self.motif_bonuses.get(motif, 0.0)
        
        # Reuse bonus (if target reuses source components)
        if self._is_reused(source, target):
            C += self.reuse_bonus
        
        # Depth penalty (complexity cost)
        C += self.depth_penalty * target.assembly_depth
        
        # Environmental fit
        C += self.env_fit * self._env_score(target)
        
        return C
    
    def _is_reused(self, source: AssemblyState, target: AssemblyState) -> bool:
        """Check if target reuses source subassemblies."""
        # Target contains source as subassembly
        return source.parts.issubset(target.parts)
    
    def _env_score(self, state: AssemblyState) -> float:
        """Environmental compatibility score."""
        # Could be based on:
        # - Solubility proxies
        # - Size compatibility
        # - Functional group exposure
        return 0.0  # Placeholder
```

### Assembly Index Emerges Naturally
```
Assembly Index = Minimum constraint-adjusted path length to state

Low index  = many cheap paths (easy to assemble)
High index = rare under baseline, rescued by constraint (complex)
```

This is **not hard-coded** - it falls out of the model dynamics.

---

## 5. Graph Construction (Lazy + Truncated)

### Strategy
**Don't enumerate the full state space. Generate on demand.**

### Implementation Approach
```python
class AssemblyGraph:
    """
    Lazy assembly graph with truncation.
    
    Only generates states and transitions as needed.
    Prunes low-probability paths.
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        min_rate_threshold: float = 1e-6,
    ):
        self.max_depth = max_depth
        self.min_rate = min_rate_threshold
        
        # Cache explored states
        self._state_cache = {}
        self._neighbor_cache = {}
    
    def get_neighbors(
        self,
        state: AssemblyState,
        baseline: AssemblyBaseline,
        constraint: AssemblyConstraint,
    ) -> List[Tuple[AssemblyState, float]]:
        """
        Generate neighbors on demand.
        
        Returns:
            List of (neighbor_state, effective_rate) tuples
        """
        if state in self._neighbor_cache:
            return self._neighbor_cache[state]
        
        neighbors = []
        
        # Generate join transitions
        for partner in self._get_join_partners(state):
            target = self._join(state, partner)
            if target.assembly_depth <= self.max_depth:
                rate = self._effective_rate(state, target, baseline, constraint)
                if rate >= self.min_rate:
                    neighbors.append((target, rate))
        
        # Generate split transitions
        for components in self._get_split_options(state):
            for target in components:
                rate = self._effective_rate(state, target, baseline, constraint)
                if rate >= self.min_rate:
                    neighbors.append((target, rate))
        
        # Cache and return
        self._neighbor_cache[state] = neighbors
        return neighbors
    
    def _effective_rate(
        self,
        source: AssemblyState,
        target: AssemblyState,
        baseline: AssemblyBaseline,
        constraint: AssemblyConstraint,
    ) -> float:
        """Compute effective transition rate."""
        base_rate = baseline.get_rate(source, target, self._infer_type(source, target))
        constraint_contrib = constraint.constraint_contribution(source, target, ...)
        return base_rate * np.exp(constraint_contrib)
```

### Key Properties
- **Lazy generation** - Only create states when needed
- **Truncation** - Max depth limit
- **Pruning** - Ignore negligible rates
- **Caching** - Reuse computed neighbors
- **Scalable** - Sublinear in full state space

---

## 6. Observation Model (Handles Missingness)

### Core Principle
**You never observe full state transitions. You observe fragments, presence, abundance.**

### Observation Types

#### 1. Presence-Only
```python
class PresenceObservationModel(ObservationModel):
    """
    Observe presence/absence of compounds.
    
    P(observed | state exists at time t)
    """
    
    def log_likelihood(
        self,
        observed_compounds: Set[str],
        latent_states: Dict[AssemblyState, float],  # state -> probability
    ) -> float:
        """
        Compute likelihood of observations given latent state distribution.
        
        Tolerates massive missingness - only need to explain what we see.
        """
        log_lik = 0.0
        
        for compound in observed_compounds:
            # Probability that compound is present
            p_present = sum(
                prob for state, prob in latent_states.items()
                if compound in state.parts
            )
            log_lik += np.log(p_present + 1e-10)  # Avoid log(0)
        
        return log_lik
```

#### 2. Fragment Overlap
```python
class FragmentObservationModel(ObservationModel):
    """
    Observe fragment distributions (e.g., from mass spec).
    
    P(fragments | latent assembly state)
    """
    
    def log_likelihood(
        self,
        observed_fragments: Dict[str, float],  # fragment -> intensity
        latent_states: Dict[AssemblyState, float],
    ) -> float:
        """
        Match observed fragments to expected fragments from states.
        """
        # Predict fragments from each state
        expected_fragments = {}
        for state, prob in latent_states.items():
            for fragment in self._generate_fragments(state):
                expected_fragments[fragment] = expected_fragments.get(fragment, 0.0) + prob
        
        # Compare observed vs expected
        log_lik = 0.0
        for fragment, intensity in observed_fragments.items():
            expected = expected_fragments.get(fragment, 0.0)
            # Poisson or Gaussian likelihood
            log_lik += -0.5 * ((intensity - expected) ** 2)
        
        return log_lik
```

#### 3. Aggregate Statistics
```python
class AggregateObservationModel(ObservationModel):
    """
    Observe size distributions, not individual states.
    
    P(size_distribution | ensemble of states)
    """
    
    def log_likelihood(
        self,
        observed_size_dist: np.ndarray,  # histogram
        latent_states: Dict[AssemblyState, float],
    ) -> float:
        """
        Compare observed vs predicted size distributions.
        """
        # Predict size distribution from states
        predicted_dist = np.zeros_like(observed_size_dist)
        for state, prob in latent_states.items():
            size_bin = self._get_size_bin(state)
            predicted_dist[size_bin] += prob
        
        # Multinomial or KL divergence
        return -np.sum((observed_size_dist - predicted_dist) ** 2)
```

### Key Properties
- **Missingness-tolerant** - Only explain what we observe
- **Flexible** - Multiple observation types
- **Scalable** - Don't need full state enumeration
- **Fits PERSISTE** - Uses `ObservedTransitions` abstraction

---

## 7. Integration with PERSISTE

### State Space
```python
class AssemblyStateSpace(StateSpace):
    """State space for assembly theory."""
    
    def __init__(self, primitives: List[str], max_depth: int = 10):
        self.primitives = primitives
        self.max_depth = max_depth
        self._state_cache = {}
    
    def get_state(self, parts: Multiset, depth: int) -> AssemblyState:
        """Get or create state."""
        key = (frozenset(parts.items()), depth)
        if key not in self._state_cache:
            self._state_cache[key] = AssemblyState(parts, depth)
        return self._state_cache[key]
    
    @property
    def dimension(self) -> int:
        """
        Effective dimension (not full enumeration).
        
        Returns approximate reachable state count.
        """
        # Combinatorial estimate based on primitives and max_depth
        return len(self.primitives) ** self.max_depth  # Upper bound
```

### Transition Graph
```python
class AssemblyTransitionGraph(TransitionGraph):
    """Lazy transition graph for assembly."""
    
    def __init__(self, state_space: AssemblyStateSpace):
        self.state_space = state_space
        self.graph = AssemblyGraph(max_depth=state_space.max_depth)
    
    def neighbors(
        self,
        state: AssemblyState,
        baseline: AssemblyBaseline,
        constraint: AssemblyConstraint,
    ) -> List[Tuple[AssemblyState, float]]:
        """Get neighbors with effective rates."""
        return self.graph.get_neighbors(state, baseline, constraint)
```

---

## 8. Example Usage

```python
# Define primitives
primitives = ['A', 'B', 'C']

# Create state space
state_space = AssemblyStateSpace(primitives, max_depth=5)

# Create baseline (physics-agnostic)
baseline = AssemblyBaseline(
    kappa=1.0,
    join_exponent=-0.5,
    split_exponent=0.3,
)

# Create constraint (assembly theory)
constraint = AssemblyConstraint()
constraint.motif_bonuses = {'stable_dimer': 1.5}
constraint.reuse_bonus = 1.0
constraint.depth_penalty = -0.2

# Create transition graph
graph = AssemblyTransitionGraph(state_space)

# Create observation model
obs_model = PresenceObservationModel(graph, state_space)

# Observed data
observed = {'A∘B', 'B∘C', 'A∘B∘C'}

# Run inference
inference = ConstraintInference(
    baseline=baseline,
    constraint=constraint,
    obs_model=obs_model,
    graph=graph,
)

# Fit constraint parameters
theta_mle = inference.fit(observed_data)

# Predict assembly index for new compound
new_state = state_space.get_state({'A': 2, 'B': 1, 'C': 1}, depth=4)
assembly_index = inference.compute_assembly_index(new_state)
```

---

## 9. Next Steps

### Phase 1: Core Implementation
1. Implement `AssemblyState` (frozen, hashable)
2. Implement `AssemblyBaseline` (factorized rates)
3. Implement `AssemblyConstraint` (motif bonuses, reuse)
4. Implement `AssemblyGraph` (lazy generation)

### Phase 2: Observation Models
1. Implement `PresenceObservationModel`
2. Implement `FragmentObservationModel`
3. Test with synthetic data

### Phase 3: Inference
1. Integrate with `ConstraintInference`
2. Test parameter fitting
3. Validate assembly index emergence

### Phase 4: Real Data
1. Test on experimental assembly data
2. Compare to published assembly indices
3. Refine constraint parameterization

---

## 10. Success Criteria

✅ **States are enumerable** - Lazy generation works
✅ **Transitions are local** - Only primitive moves
✅ **Baseline is simple** - Factorized, fast
✅ **Constraint carries chemistry** - Motifs, reuse, etc.
✅ **Observation tolerates missingness** - Presence-only works
✅ **Inference scales sublinearly** - Pruning + caching effective

If all criteria met → **We don't die** ✨
