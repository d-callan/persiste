# Assembly Plugin Architecture: Three Layers

## Critical Design Decision

**The assembly plugin does NOT bake in constraints. It provides a constraint vocabulary and feature generators.**

This is the difference between a measurement instrument and a theory.

---

## Three-Layer Architecture

### Layer 0: PERSISTE Core
**Mathematical machinery**

- Knows nothing about chemistry or biology
- Provides: `Baseline`, `ConstraintModel`, `ObservationModel`, `Inference`
- Pure math: CTMCs, likelihood, optimization

### Layer 1: Assembly Mechanics (Plugin Core)
**Hypothesis-neutral feature extraction**

**What it defines:**
- State representation (multisets, not molecular graphs)
- Transition grammar (join, split, decay, rearrange)
- Feature extractors (cheap, local, compositional, interpretable)

**What it does NOT define:**
- Which features matter
- What direction they push
- Whether reuse is "good"

**Example features:**
```python
features = {
    'reuse_count': 1.0,        # Observable: how many reuses?
    'depth_change': 1.0,       # Observable: depth increased by 1
    'size_change': 1.0,        # Observable: gained 1 part
    'symmetry_score': 0.33,    # Observable: 1/3 parts are same
    'diversity_score': 1.10,   # Observable: Shannon entropy
    'motif_gained_helix': 1.0, # Observable: helix motif appeared
}
```

**Key principle:** Features are hypotheses-neutral. Weights are hypotheses.

### Layer 2: Assembly Theories (User-Defined)
**Hypothesis-driven constraint models**

**What lives here:**
- Constraint weights θ
- Assumptions about what matters
- Controversy

**Users specify:**
```python
# Null model (no constraints)
θ = {}

# Reuse-only hypothesis
θ = {'reuse_count': 1.0}

# Assembly theory hypothesis
θ = {'reuse_count': 1.0, 'depth_change': -0.3}

# Inferred from data
θ = fit_from_data(observations)
```

**Key principle:** Constraints are hypotheses to test, not assumptions to bake in.

---

## Why This Matters

### ❌ Bad: Baking in Assumptions
```python
# DON'T DO THIS
class AssemblyConstraint:
    def __init__(self):
        self.reuse_bonus = 1.0      # Assumes reuse is good
        self.depth_penalty = -0.3   # Assumes depth is bad
        self.motif_bonuses = {      # Assumes helices are stable
            'helix': 2.0
        }
```

**Problem:** This is philosophy, not science. You're asserting, not testing.

### ✅ Good: Providing Vocabulary
```python
# DO THIS
from persiste.core.constraints import ConstraintModel

class AssemblyConstraint(ConstraintModel):
    def __init__(self, feature_weights=None, **kwargs):
        # standard initialization with PERSISTE core
        super().__init__(parameters={"theta": feature_weights or {}}, **kwargs)
        self.feature_weights = self.parameters["theta"]
        self.feature_extractor = AssemblyFeatureExtractor()
    
    def constraint_contribution(self, source, target, transition_type):
        features = self.feature_extractor.extract_features(source, target, transition_type)
        return sum(self.feature_weights.get(f, 0.0) * v 
                   for f, v in features.to_dict().items())
```

**Benefit:** This is science. You're testing hypotheses.

---

## Scientific Questions Enabled

With this architecture, you can ask:

### 1. Does reuse bias appear in this system?
```python
from persiste.plugins.assembly.cli import fit_assembly_constraints

# Fit multiple models via CLI or interface
result_null = fit_assembly_constraints(data, primitives, feature_names=[])
result_reuse = fit_assembly_constraints(data, primitives, feature_names=['reuse_count'])

LRT = 2 * (result_reuse['stochastic_ll'] - result_null['stochastic_ll'])
```

### 2. Which constraints emerge under inference?
```python
# Use standard analysis recipe
from persiste.plugins.assembly.recipes import run_standard_analysis

result = run_standard_analysis(data, primitives)
theta_hat = result['theta_hat']

# See which weights are non-zero
significant = {f: w for f, w in theta_hat.items() if abs(w) > 0.5}
```

### 3. Do constraints strengthen over time?
```python
θ_early = fit(data_early)
θ_late = fit(data_late)

test: ||θ_late|| > ||θ_early||
```

### 4. Are abiotic systems constraint-free?
```python
θ_abiotic = fit(abiotic_data)

test: θ_abiotic ≈ 0
```

### 5. Are early-life systems weakly constrained?
```python
θ_early_life = fit(early_life_data)
θ_modern = fit(modern_data)

test: 0 < ||θ_early_life|| < ||θ_modern||
```

---

## Implementation

### Layer 1: Feature Extractor
**File:** `src/persiste/plugins/assembly/features/assembly_features.py`

```python
class AssemblyFeatureExtractor:
    """Extract features from assembly transitions."""
    
    def extract_features(self, source, target, transition_type):
        """
        Extract all features from a transition.
        
        Returns:
            TransitionFeatures with observables (not value judgments)
        """
        features = TransitionFeatures()
        
        # Reuse: how many times does source appear in target?
        features.reuse_count = self._compute_reuse(source, target)
        
        # Depth change
        features.depth_change = target.assembly_depth - source.assembly_depth
        
        # Size change
        features.size_change = target.size - source.size
        
        # Motif changes
        features.motif_gained = target.motifs - source.motifs
        features.motif_lost = source.motifs - target.motifs
        
        # Symmetry (of target)
        features.symmetry_score = self._compute_symmetry(target)
        
        # Diversity (of target)
        features.diversity_score = self._compute_diversity(target)
        
        return features
```

### Layer 2: Constraint Model
**File:** `src/persiste/plugins/assembly/constraints/assembly_constraint.py`

```python
class AssemblyConstraint:
    """Constraint model (Layer 2 - theories)."""
    
    def __init__(self, feature_weights=None):
        self.feature_weights = feature_weights or {}
        self.feature_extractor = AssemblyFeatureExtractor()
    
    def constraint_contribution(self, source, target, transition_type):
        """
        Compute C(i → j; θ) = θ · f(i → j)
        
        θ = feature weights (the theory/hypothesis)
        f = feature vector (from feature extractor)
        """
        features = self.feature_extractor.extract_features(source, target, transition_type)
        feature_dict = features.to_dict()
        
        C = 0.0
        for feature_name, feature_value in feature_dict.items():
            weight = self.feature_weights.get(feature_name, 0.0)
            C += weight * feature_value
        
        return C
    
    @classmethod
    def null_model(cls):
        """Null model (no constraints)."""
        return cls(feature_weights={})
    
    @classmethod
    def reuse_only(cls, reuse_weight=1.0):
        """Reuse-only model."""
        return cls(feature_weights={'reuse_count': reuse_weight})
    
    @classmethod
    def assembly_theory(cls, reuse=1.0, depth_penalty=-0.3):
        """Standard assembly theory model."""
        return cls(feature_weights={
            'reuse_count': reuse,
            'depth_change': depth_penalty,
        })
```

---

## Model Comparison Results

From `assembly_model_comparison.py`:

```
Transition                     Model                C        λ_eff      Boost
--------------------------------------------------------------------------------
AB → ABC (simple join)         Null                 0.00    0.7071    1.00x
AB → ABC (simple join)         Reuse-only           1.00    1.9221    2.72x
AB → ABC (simple join)         Assembly Theory      0.70    1.4239    2.01x

AB → helix (motif gain)        Null                 0.00    0.7071    1.00x
AB → helix (motif gain)        Reuse-only           1.00    1.9221    2.72x
AB → helix (motif gain)        Assembly Theory      0.70    1.4239    2.01x

AAB → AABC (reuse + depth)     Null                 0.00    0.5774    1.00x
AAB → AABC (reuse + depth)     Reuse-only           1.00    1.5694    2.72x
AAB → AABC (reuse + depth)     Assembly Theory      0.70    1.1626    2.01x
```

**Key insight:** Different theories give different predictions. Data decides which is right.

---

## Future: Suggested Constraints

Once we have validation from real data, we can offer presets:

```python
# Always available: null model
constraint = AssemblyConstraint.null_model()

# Suggested (based on published studies)
constraint = AssemblyConstraint.suggested(
    regime='early_life',  # or 'abiotic', 'modern'
    confidence='low'      # or 'medium', 'high'
)

# Custom (user-defined)
constraint = AssemblyConstraint({
    'reuse_count': 1.5,
    'depth_change': -0.2,
})
```

**But for now:** Let data decide, not assumptions.

---

## Validation Reality Check

User quote:
> "they are a big assumption w little validation behind them"

**Exactly.** That's why they must be inferred, not assumed.

The assembly plugin should support:

1. **Null constraint:** θ = 0 (blind chemistry baseline)
2. **Weakly regularized inference:** Let data say whether reuse matters
3. **Model comparison:** Null vs reuse-only vs reuse+depth

This mirrors phylogenetics:
- dN/dS vs branch-site vs RELAX
- Without pretending assembly theory is settled

---

## Comparison to Phylogenetics

### HyPhy (phylo plugin)
- **Layer 1:** Codon substitution mechanics (MG94)
- **Layer 2:** Selection models (dN/dS, branch-site, RELAX)
- **Inference:** Fit ω from data, compare models

### Assembly (this plugin)
- **Layer 1:** Assembly mechanics (features)
- **Layer 2:** Constraint models (reuse, depth, etc.)
- **Inference:** Fit θ from data, compare models

**Same pattern:** Mechanics ≠ theories.

---

## Key Takeaways

1. **Layer 1 (mechanics) is hypothesis-neutral**
   - Defines what features exist
   - Does not say which matter

2. **Layer 2 (theories) is user-defined**
   - Users specify which features matter
   - Can be inferred from data

3. **Features are observables, weights are hypotheses**
   - reuse_count = 1.0 is an observable
   - reuse_weight = 1.0 is a hypothesis

4. **Null model is always available**
   - θ = {} (no constraints)
   - Baseline for comparison

5. **Model comparison via likelihood**
   - Fit multiple models
   - Compare AIC/BIC
   - LRT for nested models

6. **Science, not philosophy**
   - Test hypotheses, don't assert them
   - Let data decide

---

## Files Created

```
src/persiste/plugins/assembly/
├── features/
│   ├── __init__.py
│   └── assembly_features.py       # Layer 1 (mechanics)
├── constraints/
│   └── assembly_constraint.py     # Layer 2 (theories) - STANDARDIZED
├── recipes/
│   └── standard_analysis.py       # Official analysis entry point
└── validation/
    └── experiments/
        └── assembly_model_comparison.py   # Architecture demo
```

---

## Next Steps

1. **Implement CTMC dynamics** (Phase 1.5)
   - Simulate assembly to equilibrium
   - θ → rates → dynamics → latent states

2. **Full inference** (Phase 1.7)
   - Fit θ from observed data
   - Compare null vs reuse vs full models

3. **Validation** (Phase 1.8)
   - Simulation study
   - Parameter recovery
   - Real data tests

**Then:** We'll know which constraints actually matter.
