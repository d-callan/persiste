# CopyNumberDynamics Plugin

Models gene family copy number evolution as a small-state CTMC.

## Biological Question

**How do different lineages regulate gene dosage over evolutionary time?**

In eukaryotic pathogens:
- CNV is adaptive
- Dosage changes are fast, reversible, and lineage-specific
- Selection often acts on variance, not direction

This plugin models **how copy number moves**, not sequence.

## Design Philosophy

### What Makes This Different

**Complements GeneContent (doesn't replace it):**
- GeneContent: presence/absence (0/1)
- CopyNumberDynamics: dosage regulation (0/1/2/3+)

**Statistically honest:**
- Hierarchical baseline (avoids false positives)
- Sparse transition graph (biologically realistic)
- Multiplicative constraints (learned from GeneContent)

**Reuses PersiSTE machinery:**
- Felsenstein pruning
- Per-family likelihoods
- MLE + profile likelihood
- LRT vs null

## State Space

Binned copy number states (not raw counts):

| State | Name | Meaning |
|-------|------|---------|
| 0 | ABSENT | Gene family not present |
| 1 | SINGLE | Single copy (baseline for ploidy) |
| 2 | LOW_MULTI | Low multi-copy (2-3 above baseline) |
| 3 | HIGH_MULTI | High multi-copy (≥4 above baseline) |

**Why binning?**
- Raw copy number (0...N) explodes state space
- Binning captures biology while keeping CTMC tractable
- Works across species with different ploidy

## Transition Graph

**Sparse, gradual transitions only:**

```
0 ↔ 1 ↔ 2 ↔ 3
```

**Forbidden jumps:**
- 0 ↔ 3 (no direct absent → high-multi)
- 0 ↔ 2 (no direct absent → low-multi)
- 1 ↔ 3 (no direct single → high-multi)

**Rationale:**
- Biologically realistic (gradual amplification/contraction)
- Statistically stabilizing (fewer parameters)
- Prevents spurious volatility

## Baseline Models

### Hierarchical (DEFAULT, RECOMMENDED)

Per-family rates drawn from global distribution:
```
log(rate_fam) ~ Normal(log(rate_global), σ)
```

**Why this is good:**
- Families differ in dosage tolerance
- Avoids false positives (learned from GeneContent)
- More biologically realistic

### Global (fast, exploratory)

Single set of rates for all families.

## Constraint Types

### 1. Dosage Stability (CORE)

**Question:** "Do some genes resist copy number changes?"

**Effect:** Suppresses BOTH amplification AND contraction

**Parameter:**
- θ < 0 → dosage buffered (stable)
- θ > 0 → dosage volatile (frequent changes)

**Use cases:**
- Essential genes (expect θ < 0)
- Housekeeping genes (expect θ < 0)
- Antigen families (expect θ > 0)

### 2. Amplification Bias

**Question:** "Do pathogenic lineages favor copy number increases?"

**Effect:** Boosts amplification (1→2, 2→3) only (asymmetric)

**Parameter:**
- θ < 0 → amplification suppressed
- θ > 0 → amplification favored

**Use cases:**
- Drug resistance genes
- Efflux pumps
- Virulence factors

### 3. Host-Conditioned Volatility

**Question:** "Does copy number evolve differently in host-associated lineages?"

**Effect:** Lineage-conditioned multiplier on all CN transitions

**Use cases:**
- Host-adapted pathogens
- Environmental vs clinical isolates

## Quick Start

### Basic Analysis (Null Model)

```python
from persiste.plugins.copynumber import fit

# Fit null model (baseline only)
result = fit(
    cn_matrix=cn_data,  # (n_families, n_taxa) binned states
    family_names=families,
    taxon_names=taxa,
    tree=tree,
    baseline_type='hierarchical',
    verbose=True
)

result.print_summary()
```

### With Dosage Stability Constraint

```python
# Fit alternative model
result_alt = fit(
    cn_matrix=cn_data,
    family_names=families,
    taxon_names=taxa,
    tree=tree,
    constraint_type='dosage_stability',
    theta=-0.5,  # buffered
    verbose=True
)

# Compare to null
from persiste.plugins.copynumber.cn_interface import likelihood_ratio_test

lrt = likelihood_ratio_test(
    alternative=result_alt,
    null=result,
    verbose=True
)
```

### Binning Raw Copy Numbers

```python
from persiste.plugins.copynumber.states.cn_states import CopyNumberState

# Bin raw counts to states
binned = CopyNumberState.bin_matrix(raw_counts, ploidy=2)

# Or individual values
state = CopyNumberState.from_raw_count(6, ploidy=2)
# Returns: LOW_MULTI (2)
```

## Standard Workflow

1. **Prepare data:** Bin raw copy numbers to states (0-3)
2. **Fit null model:** Baseline only (no constraint)
3. **Fit alternative:** With biologically motivated constraint
4. **Compare models:** LRT, AIC, BIC
5. **Interpret θ:** Biological meaning depends on constraint type

## Model Selection

**Always fit null first:**
```python
null = fit(cn_matrix, families, taxa, tree)
```

**Then test specific hypotheses:**
```python
# Hypothesis: Essential genes are dosage-buffered
alt = fit(cn_matrix, families, taxa, tree,
          constraint_type='dosage_stability',
          theta=-0.5)

lrt = likelihood_ratio_test(alt, null)
```

## Integration with GeneContent

Once both plugins exist, you can do joint analyses:

```python
# Pipeline-level integration
from persiste.plugins.genecontent import fit as fit_gc
from persiste.plugins.copynumber import fit as fit_cn

# Presence/absence
gc_result = fit_gc(pam, taxa, genes, tree)

# Copy number dynamics (for retained genes)
cn_result = fit_cn(cn_matrix, families, taxa, tree,
                   constraint_type='dosage_stability')

# Biological interpretation:
# - Genes that are never lost but highly amplified
# - Host-associated retention + CN volatility
```

## What NOT to Do (Yet)

❌ Raw CN states (too big)  
❌ Within-family breakpoint modeling  
❌ Allele-specific CN  
❌ Continuous diffusion models  

These can come later if needed.

## Validation

The plugin includes validation at multiple levels:

1. **Mechanics:** Transition matrix valid, likelihood finite
2. **Statistical honesty:** Null recovery, parameter recovery
3. **Identifiability:** Profile likelihood curvature
4. **Misspecification:** Wrong bin thresholds, missing CN calls

See `examples/` for validation scripts.

## File Organization

```
copynumber/
├── __init__.py                 # Main exports
├── cn_interface.py             # High-level API
├── README.md                   # This file
├── states/
│   └── cn_states.py            # State space (4 states)
├── baselines/
│   └── cn_baseline.py          # Baseline models
├── constraints/
│   └── cn_constraints.py       # Constraint types
├── observation/
│   └── cn_observation.py       # Observation models
├── data/
│   └── cn_data.py              # Data structures
└── examples/
    └── basic_example.py        # Usage examples
```

## References

This plugin implements ideas from:
- Copy number variation in pathogens (adaptive CNV)
- Dosage balance hypothesis (gene dosage constraints)
- Birth-death models (gain/loss dynamics)

But with a **PersiSTE twist:** baseline vs constraint framework applied to dosage regulation.
