# GeneContent Plugin (v1): Design Document

**Status:** Exploratory / v1 implementation  
**Date:** December 28, 2024

---

## Scientific Question (v1 scope)

> Which gene families are preferentially retained, lost, or gained relative to a neutral gene-content evolutionary process?

Conditioned on:
- Phylogeny
- (Optionally) host / environment labels

---

## 1. State Model

### State Definition

State = gene family presence/absence vector

For taxon t:
```
S_t = { g₁ ∈ {0,1}, g₂ ∈ {0,1}, … }
```

Where:
- g = gene family (orthogroup)
- 1 = present, 0 = absent

### Key Properties

- Binary (v1)
- Independent dimensions per gene family
- No explicit genome size constraint (yet)

### Why this works

- Matches pan-genome data
- Avoids combinatorial state explosion
- Compatible with CTMC on trees
- Extensible later to copy number

### Implementation

- `GenePresenceState`: Single family at single node (0 or 1)
- `GeneFamilyVector`: Full genome content at a node

---

## 2. Transition Model

### Allowed Transitions (per family)

```
0 → 1  (gain)
1 → 0  (loss)
```

No multi-gene transitions in v1.

### Implementation

- `RateParameters`: Gain and loss rates for a family
- `transition_probability(t)`: Closed-form P(t) = exp(Qt) for 2-state CTMC

---

## 3. Baseline Model (CRITICAL)

### Purpose

Absorb intrinsic gene volatility so constraints remain interpretable.

### Baseline Options (User-Configurable)

#### Default (Recommended): Hierarchical per-family rates

```
log λ_gain(g) ~ Normal(μ_gain, σ_gain)
log λ_loss(g) ~ Normal(μ_loss, σ_loss)
```

Why:
- Reduces false positives
- Matches biological reality
- Reviewer-safe
- Scales well

#### Alternative: Fixed per-family rates

User-provided rates from prior knowledge.

#### Simple: Global rates

Single gain/loss rate for all families (testing only).

### Essential Genes Handling

User choice:
- Known essential genes → baseline loss rate ≈ 0
- Unknown → let constraint discover them

### Implementation

- `HierarchicalRates`: Default, recommended
- `FixedRates`: User-provided
- `GlobalRates`: Simple, for testing

---

## 4. Constraint Model

### Purpose

Capture selective structure beyond baseline variability.

### Constraint Form

For each transition:
```
λ*_ij = λ_baseline(g) × exp(C(g, lineage; θ))
```

Where:
- C is cheap to compute
- θ are interpretable parameters

### Constraint Types (v1)

1. **Global retention bias**: Some genes are selectively retained
2. **Host/environment association**: Retained only in specific hosts
3. **Functional group coherence**: Pathway-level retention
4. **Genome reduction bias**: Lineage-specific loss acceleration

### Implementation

- `NullConstraint`: No effect (null hypothesis)
- `PerFamilyConstraint`: Each family has its own effect
- `RetentionBiasConstraint`: Reduced loss for specified families
- `HostAssociationConstraint`: Context-dependent effects
- `GenomeReductionConstraint`: Lineage-specific loss acceleration

---

## 5. Observation Model

### Observed Data

Gene presence/absence at tree tips.

```
Y[taxon, gene] ∈ {0,1}
```

### Assumptions (v1)

- Fully observed presence/absence
- No uncertainty in gene calls (yet)

### Later Extensions

- Missingness
- Copy number
- Assembly bias

### Implementation

- `TipObservations`: Data structure for tip states
- `GeneContentObservationModel`: Core ObservationModel adapter that plugs tip data into tree likelihoods

---

## 6. Inference Pipeline (PERSISTE Standard)

```
θ → constrained rates → CTMC on tree → likelihood
```

Supports:
- MLE (v1)
- Profile likelihoods
- LRTs
- AIC/BIC
- Bootstrap (later)

---

## 7. User Configuration Philosophy

### Rule

> Users choose what varies freely (baseline) and what they want to test (constraints)

### Minimal knobs (v1)

```python
GeneContentModel(
    baseline="hierarchical",
    constraints=["host_association"],
    essential_genes=None,
)
```

Advanced users can override defaults — but safe defaults exist.

---

## 8. Expected Inputs (v1)

### Required

#### Phylogenetic tree
- Newick format
- Branch lengths required

#### Gene family presence/absence matrix
- TSV format
- Rows: taxa
- Columns: gene families
- Values: 0/1

Example:
```
taxon   OG0001  OG0002  OG0003
A       1       0       1
B       1       1       1
C       0       0       1
```

### Optional

#### Metadata
- Host
- Environment
- Lineage labels

Example:
```
taxon   host        niche
A       human       blood
B       mosquito    gut
```

---

## 9. File Structure

```
genecontent/
├── __init__.py
├── states/
│   ├── __init__.py
│   └── gene_state.py          # GenePresenceState, GeneFamilyVector
├── baselines/
│   ├── __init__.py
│   └── gene_baseline.py       # HierarchicalRates, FixedRates, GlobalRates
├── constraints/
│   ├── __init__.py
│   └── gene_constraint.py     # NullConstraint, PerFamilyConstraint, etc.
├── observation/
│   ├── __init__.py
│   └── gene_observation.py    # TipObservations, GeneContentObservation
├── inference/
│   ├── __init__.py
│   └── gene_inference.py      # GeneContentModel, GeneContentInference
├── data/
│   ├── __init__.py
│   └── loaders.py             # load_gene_matrix, load_tree, load_metadata
└── docs/
    └── DESIGN.md              # This file
```

---

## 10. Core Framework Integration

The GeneContent plugin leverages core PERSISTE utilities for inference:

### Core Utilities Used

| Core Module | Purpose | Plugin Usage |
|-------------|---------|--------------|
| `core.trees.TreeStructure` | Generic tree representation | Parse Newick, extract structure |
| `core.pruning.FelsensteinPruning` | Likelihood computation | Per-family likelihood on tree |
| `core.pruning.SimpleBinaryTransitionProvider` | 2-state CTMC | Gain/loss rate matrices |
| `core.tree_inference.TreeMLEOptimizer` | MLE optimization | Fit rate parameters |
| `core.tree_inference.likelihood_ratio_test` | Hypothesis testing | Test constraint significance |

### Benefits of Core Integration

1. **Shared optimization** - Core improvements benefit all plugins
2. **Consistent API** - Same interface across phylo, genecontent, etc.
3. **JAX acceleration** - Optional GPU/TPU support via core
4. **Tested infrastructure** - Core utilities are well-tested

### Plugin-Specific Components

The plugin provides domain-specific:
- **Rate models**: Hierarchical per-family rates
- **Constraints**: Retention bias, host association, genome reduction
- **Data loaders**: Gene matrix, metadata parsing

---

## 11. Inference Pipeline

### High-Level API

```python
from persiste.plugins.genecontent import (
    GeneContentData,
    GeneContentInference,
)
from persiste.plugins.genecontent.constraints.gene_constraint import (
    RetentionBiasConstraint,
)

# Load data
data = GeneContentData(tree, presence_matrix, taxon_names, family_names)

# Create inference engine
inference = GeneContentInference(data)

# Fit and test
constraint = RetentionBiasConstraint(retained_families={'OG0001', 'OG0002'})
null_result, alt_result, lrt = inference.fit_and_test(constraint)

# Interpret
if lrt.significant:
    print("Evidence for selective retention")
```

### Inference Steps

1. **Fit null model**: Global gain/loss rates, no constraints
2. **Fit alternative**: Same rates + constraint parameters
3. **LRT**: Compare models, compute p-value
4. **Model selection**: AIC/BIC for non-nested comparisons

---

## 12. Next Steps

### v1 (COMPLETE)
- [x] State model
- [x] Baseline model
- [x] Constraint model
- [x] Observation model
- [x] Data loaders
- [x] Likelihood computation (Felsenstein pruning via core)
- [x] MLE inference (via core TreeMLEOptimizer)
- [x] LRT hypothesis testing (via core)
- [x] Basic examples

### v2 (future)
- [ ] Copy number extension (states: 0, 1, 2, ...)
- [ ] Detection probability / missingness
- [ ] Pathway-level constraints
- [ ] Parallel computation across families
- [ ] Integration with real pan-genome tools (Roary, Panaroo)
