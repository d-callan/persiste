# Assembly Plugin: User Guide

**Version:** 1.0  
**Date:** December 28, 2024

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Safe-by-Default API](#safe-by-default-api)
5. [Interpreting Results](#interpreting-results)
6. [Best Practices](#best-practices)
7. [Common Pitfalls](#common-pitfalls)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## Introduction

### What is the Assembly Plugin?

The Assembly Plugin infers **constraint parameters** that govern chemical assembly processes. It answers questions like:

- **Do molecules prefer to reuse existing components?** (reuse constraint)
- **Do molecules avoid deep nesting?** (depth constraint)
- **How strong are these preferences?** (parameter magnitudes)

### What Makes It Different?

**Safe-by-default design:**
- Automatic null testing (can't forget to compare)
- Joint baseline + constraint inference (prevents false positives)
- Built-in diagnostics (identifiability, robustness)
- Conservative thresholds (no overclaiming)

**Philosophical stance:**
- Constraints are only meaningful **relative to a generative null**
- This is **comparative model selection**, not "constraint detection"
- Aligns with phylogenetic mixed models and HyPhy philosophy

---

## Quick Start

### Installation

```python
# Already installed if you have PERSISTE
import sys
sys.path.insert(0, 'src')

from persiste.plugins.assembly.cli import fit_assembly_constraints, InferenceMode
from persiste.plugins.assembly.recipes.standard_analysis import run_standard_analysis
```

### Minimal Example (Using Recipe)

```python
# 1. Setup
primitives = ['A', 'B', 'C', 'D', 'E']
observed_compounds = {'A', 'B', 'AB', 'ABC'}

# 2. Run official analysis recipe
result = run_standard_analysis(
    observed_compounds=observed_compounds,
    primitives=primitives
)

# The recipe handles fitting, safety checks, and printing a summary.
```

### Advanced Usage (Low-level API)

```python
# Use the CLI interface directly for more control
result = fit_assembly_constraints(
    observed_compounds=observed_compounds,
    primitives=primitives,
    mode=InferenceMode.FULL_STOCHASTIC,
    feature_names=['reuse_count', 'depth_change'],
    n_samples=500,
    seed=123
)
```

**That's it!** The API handles:
- Null model comparison
- Profile diagnostics
- Evidence classification
- Warning generation

---

## Core Concepts

### 1. Baseline vs Constraints

**Baseline:** The "null model" - how assembly works **without** selection.
- Example: Join rate ∝ n^(-0.5), split rate ∝ n^(0.3)
- Captures basic thermodynamics/kinetics
- Defined in `AssemblyBaseline`

**Constraints:** Deviations from baseline due to **selection pressures**.
- Example: Reuse is favored → rate multiplied by exp(θ_reuse × reuse_count)
- Captures evolutionary/functional preferences
- Defined in `AssemblyConstraint`

**Key insight:** Constraints are only meaningful **relative to the baseline**.

### 2. Evidence Classification

**Δ LL = stochastic_ll(constrained) - stochastic_null_ll**

| Δ LL | Evidence | Interpretation |
|------|----------|----------------|
| < 2  | None     | No evidence for constraints |
| 2-5  | Weak     | Suggestive but not conclusive |
| 5-10 | Moderate | Likely real, validate |
| > 10 | Strong   | Well-supported |

**Conservative threshold:** Use Δ LL > 10 for publication claims.

---

## Safe-by-Default API

### fit_assembly_constraints()

**The recommended entry point.** Located in `cli.py`. Automatically:

1. Fits null model (θ = 0)
2. Fits constrained model (jointly adjusting baseline parameters if needed)
3. Computes Δ LL
4. Runs Tier 1 safety checks
5. Returns structured results for Tier 2 diagnostics

**Usage:**
```python
from persiste.plugins.assembly.cli import fit_assembly_constraints, InferenceMode

result = fit_assembly_constraints(
    observed_compounds=observed,
    primitives=primitives,
    mode=InferenceMode.FULL_STOCHASTIC,
    feature_names=['reuse_count', 'depth_change'],
)
```

**Returns:** A dictionary with:
- `theta_hat`: Fitted weights
- `stochastic_delta_ll`: Evidence strength
- `safety`: Tier 1 safety report
- `artifacts`: Structured data for Tier 2 recipes

---

## Deep Diagnostics (Tier 2 Recipes)

For publication-quality results, use the opt-in diagnostic recipes in `persiste.plugins.assembly.recipes`.

### 1. Significance Testing
```python
from persiste.plugins.assembly.recipes import null_resampling_diagnostic
report = null_resampling_diagnostic(result)
report.print_summary()
```

### 2. Uncertainty Quantification
```python
from persiste.plugins.assembly.recipes import profile_likelihood_sweep
report = profile_likelihood_sweep(result, 'reuse_count')
report.print_summary()
```

### 3. Baseline Sensitivity
```python
from persiste.plugins.assembly.recipes import baseline_perturbation_sensitivity
report = baseline_perturbation_sensitivity(result)
report.print_summary()
```

---

## Best Practices

### 1. Check Tier 1 Safety First
Always inspect `result['safety']['overall_safe']`. If it's `False`, the baseline might be misspecified or the data too sparse to identify the parameters.

### 2. Use Conservative Thresholds
Don't over-interpret Δ LL < 5. Small improvements can arise from stochastic noise or minor baseline mismatch.

### 3. Validate with Tier 2 Recipes
Before claiming a discovery, run `null_resampling_diagnostic`. If the p-value is > 0.05, the observed Δ LL could be explained by null fluctuations.

---

## API Reference

### fit_assembly_constraints
Main inference function. Supports deterministic screening and stochastic refinement.

### AssemblyBaseline
Baseline rate specification.
- `kappa`: global rate constant
- `join_exponent`: size-scaling for joins
- `split_exponent`: size-scaling for splits

### AssemblyConstraint
Standard constraint model.
- `feature_weights`: dict of weights
- `depth_gate_threshold`: for Symmetry Break A
- `primitive_classes`: for Symmetry Break B
- `founder_rank_threshold`: for Symmetry Break C

---

## Summary

**Key takeaways:**

1. **Use `fit_assembly_constraints()`** - Standard entry point with built-in Tier 1 safety.
2. **Joint Inference** - Baseline parameters are adjusted automatically to prevent false positives.
3. **Conservative Evidence** - Δ LL > 10 is the recommended threshold for publication.
4. **Tier 2 Recipes** - Use opt-in diagnostics for uncertainty and sensitivity analysis.

**This framework makes it hard to do the wrong thing and easy to do the right thing.**

For official analysis workflows, see: `recipes/standard_analysis.py`

For technical details, see: `docs/ASSEMBLY_TECHNICAL.md`

---

**Questions?** Open an issue or contact the PERSISTE team.
