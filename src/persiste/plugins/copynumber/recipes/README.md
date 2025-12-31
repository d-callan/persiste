# CopyNumberDynamics Analysis Recipes

Standard workflows for analyzing copy number evolution.

## Overview

These recipes provide self-contained analyses for common CN evolution questions. Each recipe produces interpretable results with clear biological framing.

## Available Recipes

### Recipe 0: Null CN Dynamics (Descriptive Baseline)

**Question:** "What does copy number evolution look like under no hypothesis?"

**Purpose:** Descriptive anchor - always run this first.

**Output:**
- Equilibrium CN distribution π(state)
- Expected copy number
- Dominant transition types
- Per-family summaries

**Usage:**
```python
from persiste.plugins.copynumber.recipes import null_cn_dynamics

report = null_cn_dynamics(
    cn_matrix="data/cn_matrix.tsv",
    tree="data/tree.nwk",
    baseline_type='hierarchical',  # Recommended
)

print(report.interpretation)
```

**Interpretation:**
- Describes baseline dynamics without constraints
- Identifies dominant CN states
- Reveals gain/loss balance
- Guides next steps

---

### Recipe 1: Dosage Stability Scan (Core Constraint Test)

**Question:** "Are copy number changes globally suppressed or enhanced?"

**Purpose:** Test for dosage-specific constraints (CN analogue of "Are genes conserved or labile?")

**Model:**
- Constraint: `DosageStabilityConstraint`
- θ < 0 → dosage buffering (suppressed CN changes)
- θ = 0 → neutral (baseline rates)
- θ > 0 → dosage volatility (enhanced CN changes)

**Usage:**
```python
from persiste.plugins.copynumber.recipes import dosage_stability_scan

report = dosage_stability_scan(
    cn_matrix="data/cn_matrix.tsv",
    tree="data/tree.nwk",
)

if report.p_value < 0.05:
    if report.theta < -0.3:
        print("Strong dosage buffering detected")
    elif report.theta > 0.3:
        print("Copy number volatility detected")
```

**Interpretation:**
- **θ < -0.5:** Strong buffering → Essential genes, housekeeping
- **θ > 0.5:** Volatility → Antigen families, stress response
- **|θ| < 0.1:** Weak effect or no constraint

**Biological Context:**
- Buffering: Dosage-sensitive genes, stable expression
- Volatility: Adaptive variation, immune evasion

---

### Recipe 2: Amplification Bias Test

**Question:** "Are increases favored over decreases beyond baseline?"

**Purpose:** Test for asymmetry between amplification and contraction (dosage-specific, NOT redundant with gain/loss)

**Model:**
- Constraint: `AmplificationBiasConstraint` (bidirectional)
- Amplification (1→2, 2→3) × exp(θ)
- Contraction (2→1, 3→2) × exp(-θ)
- **Does NOT affect gene birth (0→1)** - biologically distinct

**Usage:**
```python
from persiste.plugins.copynumber.recipes import amplification_bias_test

report = amplification_bias_test(
    cn_matrix="data/cn_matrix.tsv",
    tree="data/tree.nwk",
)

if report.p_value < 0.05 and report.theta > 0.3:
    print("Amplification bias detected - check for drug resistance")
```

**Interpretation:**
- **θ > 0.5:** Strong amplification bias → Drug resistance, adaptive CNV
- **θ < -0.5:** Contraction bias → Dosage constraint on multi-copy genes
- **|θ| < 0.1:** Balanced amplification/contraction

**Biological Context:**
- Amplification bias: Drug resistance, stress response, virulence factors
- Contraction bias: Cost of maintaining high copy number

**Key Design Feature:**
- Excludes 0→1 (gene birth ≠ amplification)
- Prevents signal dilution
- Biological precision

---

### Recipe 3: Lineage-Conditioned CN Volatility

**Question:** "Do some clades experience elevated dosage turnover?"

**Purpose:** Test for lineage-specific CN dynamics (descriptive, not causal)

**Model:**
- Constraint: `HostConditionedVolatilityConstraint`
- θ < 0 → suppressed volatility in target lineage
- θ = 0 → no lineage effect
- θ > 0 → elevated volatility in target lineage

**Usage:**
```python
from persiste.plugins.copynumber.recipes import lineage_volatility_test

report = lineage_volatility_test(
    cn_matrix="data/cn_matrix.tsv",
    tree="data/tree.nwk",
    target_lineage="host_associated",
)

if report.p_value < 0.05:
    print(f"Lineage {report.target_lineage} shows elevated CN volatility")
```

**Important Framing:**
- **Say:** "This lineage exhibits elevated CN volatility"
- **NOT:** "Host association causes CN volatility"
- This is DESCRIPTIVE, not causal

**Interpretation:**
- Elevated volatility → Environmental adaptation, niche-specific genes
- Suppressed volatility → Stable environment, core functions

---

### Recipe 4: Joint Presence × Dosage (FUTURE)

**Question:** "How do presence/absence and dosage dynamics interact?"

**Status:** Interface defined, not implemented in v1

**Purpose:** Joint GeneContent + CopyNumberDynamics analysis

**Questions this will answer:**
1. Are genes retained but dosage-unstable?
2. Are genes with stable presence also dosage-stable?
3. Does amplification bias differ between core vs accessory?
4. Are recently gained genes more likely to amplify?

**Implementation:** Planned for v2 (requires multi-plugin infrastructure)

See `recipe_4_joint_presence_dosage.py` for interface definition.

---

## Diagnostic Utilities

### Expected vs Observed CN

**Critical diagnostic:** Check if null model fits data.

```python
from persiste.plugins.copynumber.diagnostics import (
    expected_vs_observed_cn,
    interpret_diagnostic,
)

# Generate diagnostic plot
fig = expected_vs_observed_cn(
    cn_matrix="data/cn_matrix.tsv",
    tree="data/tree.nwk",
    save_path="diagnostics/expected_vs_observed.png",
)

# Get interpretation
interpretation = interpret_diagnostic(
    cn_matrix="data/cn_matrix.tsv",
    tree="data/tree.nwk",
)
print(interpretation)
```

**If expected and observed diverge wildly:**
- Binning may be wrong (check state definitions)
- Baseline may be mis-specified (try hierarchical)
- Data quality may be poor (check for errors)

This is the CN equivalent of "tree-sequence mismatch check".

---

## Recommended Workflow

```python
from persiste.plugins.copynumber.recipes import (
    null_cn_dynamics,
    dosage_stability_scan,
    amplification_bias_test,
)
from persiste.plugins.copynumber.diagnostics import expected_vs_observed_cn

# 1. Descriptive baseline
null_report = null_cn_dynamics(cn_matrix, tree=tree)

# 2. Diagnostic check
fig = expected_vs_observed_cn(cn_matrix, tree=tree)

# 3. Test for dosage stability
dosage_report = dosage_stability_scan(cn_matrix, tree=tree)

# 4. Test for amplification bias (if multi-copy genes present)
if null_report.stationary_distribution[2] + null_report.stationary_distribution[3] > 0.05:
    amp_report = amplification_bias_test(cn_matrix, tree=tree)
```

See `examples/recipe_demo.py` for complete workflow.

---

## Interpretation Guidelines

### Phrasing (Important!)

**Do say:**
- "This gene family shows evidence of suppressed copy number transitions"
- "Amplification and contraction are balanced"
- "This lineage exhibits elevated CN volatility"

**Don't say:**
- "This gene is dosage regulated" (too causal)
- "Host association causes CN volatility" (not supported)
- "Amplification is adaptive" (requires additional evidence)

### Effect Size Interpretation

**Detectable effects:**
- |θ| ≥ 0.7 (2× rate change) - Strong biological signal

**Marginal effects:**
- 0.3 ≤ |θ| < 0.7 (1.3-2× rate change) - Moderate signal

**Weak effects:**
- |θ| < 0.3 (<1.3× rate change) - Interpret cautiously

### Sample Size Requirements

**Minimum recommended:**
- 200+ gene families
- 20+ taxa
- Branch lengths ≥ 1.0 (or total tree depth ≥ 2.0)

**For stronger signals:**
- 500+ families (better θ estimation)
- 50+ taxa (more phylogenetic information)

---

## Technical Details

### Constraints

All constraints modify rates multiplicatively:
```
effective_rate = baseline_rate × exp(θ × effect)
```

Never additive - learned from GeneContent experience.

### CN States

Coarse bins (not raw counts):
- State 0: Absent (CN = 0)
- State 1: Single-copy (CN = 1)
- State 2: Low-multi (CN = 2-5)
- State 3: High-multi (CN ≥ 6)

### Baseline Models

**Global baseline:**
- Single rate for all families
- Fast, but may underfit heterogeneity

**Hierarchical baseline (recommended):**
- Family-specific rates drawn from distribution
- Captures regime heterogeneity
- Better fit for diverse gene families

---

## Examples

See `examples/` directory:
- `recipe_demo.py` - Complete workflow demonstration
- `basic_example.py` - Simple usage examples

---

## See Also

- `../validation/VALIDATION_RESULTS_v2.md` - Validation details
- `../README.md` - Plugin documentation
- `../examples/` - Usage examples
- `/recipes/README.md` - Multi-plugin recipes (future)
