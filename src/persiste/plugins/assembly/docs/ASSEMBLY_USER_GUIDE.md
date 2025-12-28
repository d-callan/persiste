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

from persiste.plugins.assembly.inference.robust_inference import RobustConstraintInference
from persiste.plugins.assembly.baselines.baseline_family import SimpleBaselineFamily
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.observation.presence_model import FrequencyWeightedPresenceModel
```

### Minimal Example

```python
# 1. Setup
primitives = ['A', 'B', 'C', 'D', 'E']
graph = AssemblyGraph(primitives, max_depth=5)
baseline_family = SimpleBaselineFamily(parameter='join_exponent')
obs_model = FrequencyWeightedPresenceModel()

# 2. Create inference object
inference = RobustConstraintInference(
    graph=graph,
    baseline_family=baseline_family,
    obs_model=obs_model,
    regularization=0.1,  # L2 penalty
)

# 3. Fit with automatic diagnostics
result = inference.fit_with_null(
    observed_counts={'A': 50, 'B': 45, 'AB': 30, ...},
    constraint_features=['reuse_count', 'depth_change'],
)

# 4. Interpret
print(result)
print(result.get_recommendation())
```

**That's it!** The API handles:
- Null model comparison
- Profile diagnostics
- Evidence classification
- Warning generation

---

## Core Concepts

### 1. Baseline vs Constraints

**Baseline:** The "null model" - how assembly works **without** constraints.
- Example: Join rate ∝ n^(-0.5), split rate ∝ n^(0.3)
- Captures basic thermodynamics/kinetics

**Constraints:** Deviations from baseline due to **selection pressures**.
- Example: Reuse is favored → rate multiplied by exp(θ_reuse × reuse_count)
- Captures evolutionary/functional preferences

**Key insight:** Constraints are only meaningful **relative to the baseline**.

### 2. Joint Inference

**Problem:** If baseline is wrong, constraints compensate → false positives.

**Solution:** Infer baseline **and** constraints jointly:

```
θ, φ = argmax P(data | baseline(φ), constraints(θ))
```

**Implementation:**
```python
# Infer join_exponent (baseline) + constraints (θ)
baseline_family = SimpleBaselineFamily(
    parameter='join_exponent',  # Which baseline param to infer
    initial_value=-0.5,         # Starting guess
    prior_std=0.2,              # Regularization strength
)
```

**Why it works:**
- Baseline absorbs baseline errors
- Constraints only explain **constraint-specific** patterns
- False positives drop dramatically

### 3. Regularization

**Problem:** With finite data, spurious constraints can improve fit by chance.

**Solution:** L2 penalty on constraint parameters:

```
Objective = log P(data | θ, φ) - λ ||θ||²
```

**Implementation:**
```python
inference = RobustConstraintInference(
    ...,
    regularization=0.1,  # λ = 0.1 (recommended)
)
```

**Effect:**
- Pushes small/borderline constraints toward zero
- Doesn't eliminate strong constraints
- Standard practice in statistics

### 4. Evidence Classification

**Δ LL = log L(constrained) - log L(null)**

| Δ LL | Evidence | Interpretation |
|------|----------|----------------|
| < 2  | None     | No evidence for constraints |
| 2-5  | Weak     | Suggestive but not conclusive |
| 5-10 | Moderate | Likely real, validate |
| > 10 | Strong   | Well-supported |

**Conservative threshold:** Use Δ LL > 10 for publication claims.

### 5. Identifiability

**Question:** Can we uniquely determine θ from data?

**Test:** Profile likelihood - vary θ, measure Δ LL.

**Interpretation:**
- **Flat profile** (range < 2): Not identifiable
- **Broad profile** (range 2-10): Weakly identifiable
- **Sharp profile** (range > 10): Identifiable

**Automatic:** `result.identifiable` and `result.profile_diagnostics`

---

## Safe-by-Default API

### fit_with_null()

**The recommended API.** Automatically:

1. Fits null model (θ = 0)
2. Fits constrained model
3. Computes Δ LL
4. Runs profile diagnostics
5. Classifies evidence
6. Generates warnings

**Usage:**
```python
result = inference.fit_with_null(
    observed_counts,
    constraint_features=['reuse_count', 'depth_change'],
    profile_diagnostics=True,  # Default: True
    verbose=True,              # Default: True
)
```

**Returns:** `ConstraintResult` with:
- `estimate`: Fitted θ
- `baseline_params`: Fitted φ
- `delta_ll`: Improvement over null
- `evidence`: "none" | "weak" | "moderate" | "strong"
- `identifiable`: True/False
- `profile_diagnostics`: Per-parameter diagnostics
- `warnings`: List of issues
- `robustness_score`: 0-1 score

### baseline_sensitivity_analysis()

**Test robustness to baseline specification.**

```python
baseline_variations = [
    SimpleBaselineFamily(parameter='join_exponent', initial_value=-0.4),
    SimpleBaselineFamily(parameter='join_exponent', initial_value=-0.5),
    SimpleBaselineFamily(parameter='join_exponent', initial_value=-0.6),
]

stability = inference.baseline_sensitivity_analysis(
    observed_counts,
    baseline_variations,
)

# stability = {'reuse_count': 0.15, 'depth_change': 0.08}
```

**Interpretation:**
- **< 0.2:** Stable (good!)
- **0.2-0.5:** Moderate sensitivity
- **> 0.5:** Unstable (baseline matters!)

### cross_validate()

**Test generalization to held-out data.**

```python
cv_score = inference.cross_validate(
    observed_counts,
    k=5,  # 5-fold CV
)
```

**Interpretation:**
- **> -100:** Good generalization
- **< -100:** Possible overfitting

---

## Interpreting Results

### Example Output

```
============================================================
Constraint Inference Result
============================================================

Evidence: STRONG
  Δ LL: 45.23
  Identifiable: True
  Robustness: 0.85

Constraint Parameters:
  reuse_count: 1.234 (strong)
  depth_change: -0.567 (moderate)

Baseline Parameters:
  join_exponent: -0.523

⚠ Warnings:
  • None

============================================================
```

### What to Report

**Minimal:**
- Δ LL
- Evidence class
- Parameter estimates
- Warnings

**Recommended:**
- All of the above
- Profile diagnostics (identifiability)
- Baseline sensitivity (if tested)
- Cross-validation score (if tested)

**Example:**
> "We found strong evidence for a reuse constraint (θ_reuse = 1.23, Δ LL = 45.2, profile range = 38.5). The constraint was robust to baseline variations (±0.15) and cross-validated successfully (CV score = -85.3)."

### Decision Tree

```
Is Δ LL > 10?
├─ No → Use null model (no constraints)
└─ Yes → Check warnings
    ├─ Warnings present? → Validate carefully
    │   ├─ Baseline sensitivity → Test variations
    │   ├─ Not identifiable → Get more data
    │   └─ Other → Address specific issue
    └─ No warnings → Constraints well-supported
        └─ Optional: Cross-validate for extra confidence
```

---

## Best Practices

### 1. Always Use Joint Inference

**DO:**
```python
baseline_family = SimpleBaselineFamily(parameter='join_exponent')
```

**DON'T:**
```python
baseline_family = FixedBaseline(baseline)  # Only if well-validated!
```

**Why:** Prevents false positives from baseline misspecification.

### 2. Use Regularization

**DO:**
```python
inference = RobustConstraintInference(..., regularization=0.1)
```

**DON'T:**
```python
inference = RobustConstraintInference(..., regularization=0.0)
```

**Why:** Suppresses spurious constraints from finite-sample noise.

### 3. Check Diagnostics

**DO:**
```python
result = inference.fit_with_null(..., profile_diagnostics=True)
if not result.identifiable:
    print("Warning: Parameters not identifiable!")
```

**DON'T:**
```python
# Just trust the point estimates
theta = result.estimate
```

**Why:** Point estimates can be misleading if not identifiable.

### 4. Validate on Independent Data

**DO:**
```python
# Split data
train_counts, test_counts = split_data(observed_counts)

# Fit on train
result = inference.fit_with_null(train_counts)

# Validate on test
test_ll = evaluate_on_test(result, test_counts)
```

**DON'T:**
```python
# Use all data for fitting, no validation
result = inference.fit_with_null(all_data)
```

**Why:** Prevents overfitting, ensures generalization.

### 5. Report Uncertainty

**DO:**
```python
print(f"θ_reuse = {result.estimate['reuse_count']:.2f}")
print(f"Profile range: {result.profile_diagnostics['reuse_count'].ll_range:.1f}")
print(f"Evidence: {result.evidence}")
```

**DON'T:**
```python
print(f"θ_reuse = {result.estimate['reuse_count']}")  # No context!
```

**Why:** Science requires uncertainty quantification.

---

## Common Pitfalls

### Pitfall 1: Trusting Δ LL < 5

**Problem:** Weak evidence (Δ LL = 3) can occur by chance.

**Solution:** Use conservative threshold (Δ LL > 10).

**Example:**
```python
if result.delta_ll < 10:
    print("Insufficient evidence. Use null model.")
```

### Pitfall 2: Ignoring Warnings

**Problem:** Warnings indicate real issues.

**Solution:** Address each warning before publishing.

**Example:**
```python
if result.warnings:
    print("⚠ Issues detected:")
    for w in result.warnings:
        print(f"  • {w}")
    print("→ Validate before claiming constraints")
```

### Pitfall 3: Fixed Baseline Without Validation

**Problem:** If baseline is wrong, constraints compensate.

**Solution:** Use joint inference or validate baseline first.

**Example:**
```python
# Validate baseline on independent data
baseline_ll = validate_baseline(baseline, validation_data)
if baseline_ll < threshold:
    print("Baseline doesn't fit! Use joint inference.")
```

### Pitfall 4: Small Sample Size

**Problem:** Need ≥80 samples for reliable inference.

**Solution:** Check scaling curves (see ASSEMBLY_SCALING_CURVES.md).

**Example:**
```python
if len(observed_counts) < 20:
    print("⚠ Too few samples. Need ≥80 for robust inference.")
```

### Pitfall 5: Not Testing Baseline Sensitivity

**Problem:** Results may be fragile to baseline choice.

**Solution:** Run baseline_sensitivity_analysis().

**Example:**
```python
stability = inference.baseline_sensitivity_analysis(...)
if max(stability.values()) > 0.5:
    print("⚠ Unstable across baselines!")
```

---

## Advanced Usage

### Custom Baseline Families

```python
from persiste.plugins.assembly.baselines.baseline_family import BaselineFamily, BaselinePrior

# Infer multiple baseline parameters
baseline_family = BaselineFamily(
    parameters={'kappa': 1.0, 'join_exponent': -0.5},
    priors={
        'kappa': BaselinePrior(mean=1.0, std=0.3),
        'join_exponent': BaselinePrior(mean=-0.5, std=0.2),
    },
    fixed_parameters={'split_exponent': 0.3},
)
```

### Custom Constraints

```python
# Add new feature
class CustomConstraint(AssemblyConstraint):
    def compute_features(self, state):
        features = super().compute_features(state)
        features['my_feature'] = ...  # Custom logic
        return features

# Infer custom feature
result = inference.fit_with_null(
    observed_counts,
    constraint_features=['reuse_count', 'my_feature'],
)
```

### Bootstrap Confidence Intervals

```python
# Bootstrap
n_boot = 100
estimates = []

for i in range(n_boot):
    # Resample data
    boot_counts = resample(observed_counts)
    
    # Fit
    theta, _, _ = inference.fit(boot_counts)
    estimates.append(theta['reuse_count'])

# 95% CI
ci_95 = np.percentile(estimates, [2.5, 97.5])
print(f"95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]")
```

---

## Troubleshooting

### "Not identifiable" Warning

**Cause:** Flat profile likelihood (range < 10).

**Solutions:**
1. Get more data (increase sample size)
2. Increase system complexity (more primitives, deeper)
3. Use stronger constraints (larger effect sizes)
4. Check if feature is actually present in data

### "Δ LL < 2" (No Evidence)

**Cause:** Data doesn't support constraints.

**Solutions:**
1. Use null model (no constraints)
2. Check if data is actually from constrained process
3. Increase sample size
4. Try different features

### Optimization Fails

**Cause:** Bad initial guess or numerical issues.

**Solutions:**
1. Try different initial values
2. Increase `maxiter` in optimizer
3. Check for NaN/Inf in data
4. Simplify model (fewer features)

### False Positives

**Cause:** Baseline misspecification or finite-sample noise.

**Solutions:**
1. Use joint inference (SimpleBaselineFamily)
2. Increase regularization (0.1 → 0.2)
3. Use conservative threshold (Δ LL > 10)
4. Validate with baseline_sensitivity_analysis()

---

## API Reference

### RobustConstraintInference

**Constructor:**
```python
RobustConstraintInference(
    graph: AssemblyGraph,
    baseline_family: BaselineFamily,
    obs_model: FrequencyWeightedPresenceModel,
    initial_state: Optional[AssemblyState] = None,
    regularization: float = 0.1,
    n_latent_samples: int = 100,
    t_max: float = 50.0,
    burn_in: float = 25.0,
)
```

**Methods:**

- `fit_with_null()` - Recommended API with automatic diagnostics
- `fit()` - Low-level fitting (use fit_with_null() instead)
- `baseline_sensitivity_analysis()` - Test robustness to baseline
- `cross_validate()` - K-fold cross-validation

### ConstraintResult

**Attributes:**

- `estimate: Dict[str, float]` - Fitted constraint parameters
- `baseline_params: Dict[str, float]` - Fitted baseline parameters
- `delta_ll: float` - Improvement over null
- `evidence: str` - Evidence class
- `identifiable: bool` - Whether parameters are identifiable
- `profile_diagnostics: Dict` - Per-parameter diagnostics
- `warnings: List[str]` - Warning messages
- `robustness_score: float` - Overall robustness (0-1)

**Methods:**

- `is_significant(threshold=10.0)` - Check if Δ LL > threshold
- `get_recommendation()` - Human-readable recommendation

### BaselineFamily

**Subclasses:**

- `FixedBaseline` - No inference (fast but risky)
- `SimpleBaselineFamily` - Infer 1 parameter (recommended)
- `BaselineFamily` - Infer multiple parameters (advanced)

---

## Summary

**Key takeaways:**

1. **Use `fit_with_null()`** - Safe by default
2. **Use `SimpleBaselineFamily`** - Prevents false positives
3. **Use `regularization=0.1`** - Suppresses noise
4. **Check `result.evidence`** - Don't overclaim
5. **Validate with sensitivity analysis** - Ensure robustness

**This framework makes it hard to do the wrong thing and easy to do the right thing.**

For examples, see: `examples/assembly_robust_example.py`

For technical details, see: `docs/ASSEMBLY_TECHNICAL.md`

---

**Questions?** Open an issue or contact the PERSISTE team.
