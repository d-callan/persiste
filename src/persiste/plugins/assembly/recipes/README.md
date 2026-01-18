# Assembly Constraint Analysis Recipes

Standard workflows for analyzing assembly constraint inference robustness.

## Overview

These recipes provide self-contained diagnostic analyses for assembly constraint inference. Each recipe produces interpretable results with clear recommendations.

**Important**: These are Tier 2 diagnostics—opt-in, expensive, and runnable post-hoc. For automatic safety checks (Tier 1), see the `safety/` module.

## Available Recipes

### Recipe 0: Null Resampling Diagnostic

**Question:** "Is θ̂ significantly different from θ=0?"

**Purpose:** Tests whether inferred constraints are real or noise.

**Output:**
- Observed ΔLL at θ̂
- Null distribution via bootstrap resampling
- P-value
- Severity and recommendation

**Usage:**
```python
from persiste.plugins.assembly.recipes import null_resampling_diagnostic

report = null_resampling_diagnostic(artifacts, cache, n_resamples=1000)
report.print_summary()

if report.p_value < 0.05:
    print("Constraints are significant!")
```

**Interpretation:**
- p < 0.01: STRONG evidence for constraints
- p < 0.05: MODERATE evidence for constraints
- p ≥ 0.05: WEAK evidence—constraints may be noise

---

### Recipe 1: Profile Likelihood Sweep

**Question:** "How well-constrained is each parameter?"

**Purpose:** Characterizes uncertainty in θ̂ via profile likelihood.

**Output:**
- MLE for the feature
- 95% confidence interval
- Profile likelihood curve
- Severity and recommendation

**Usage:**
```python
from persiste.plugins.assembly.recipes import profile_likelihood_sweep

report = profile_likelihood_sweep(artifacts, cache, 'reuse_count')
report.print_summary()
report.plot(save_path='profile_reuse_count.png')
```

**Interpretation:**
- CI width < 0.5: WELL-CONSTRAINED parameter
- CI width < 1.5: MODERATELY constrained
- CI width ≥ 1.5: POORLY constrained—interpret cautiously

---

### Recipe 2: Baseline Perturbation Sensitivity

**Question:** "How sensitive is θ̂ to baseline misspecification?"

**Purpose:** Stress tests inference under baseline perturbations.

**Output:**
- θ̂ under each perturbation
- Log-likelihoods
- Stability assessment
- Severity and recommendation

**Usage:**
```python
from persiste.plugins.assembly.recipes import baseline_perturbation_sensitivity

report = baseline_perturbation_sensitivity(artifacts, cache)
report.print_summary()

if not report.stable:
    print("WARNING: Inference sensitive to baseline!")
```

**Interpretation:**
- Stable: Constraints robust under perturbations
- Unstable: Validate baseline before trusting constraints

---

## Recommended Workflow

```python
from persiste.plugins.assembly.recipes import (
    null_resampling_diagnostic,
    profile_likelihood_sweep,
    baseline_perturbation_sensitivity,
)

# After inference, run diagnostics
artifacts = result['artifacts']
cache = result['cache']

# 1. Test significance
null_report = null_resampling_diagnostic(artifacts, cache)
null_report.print_summary()

# 2. Check parameter uncertainty
for feature in artifacts.theta_hat:
    profile_report = profile_likelihood_sweep(artifacts, cache, feature)
    profile_report.print_summary()

# 3. Test baseline sensitivity
pert_report = baseline_perturbation_sensitivity(artifacts, cache)
pert_report.print_summary()
```

---

## Interpretation Guidelines

### Phrasing (Important!)

**Do say:**
- "This constraint shows evidence of elevated reuse preference"
- "The depth penalty parameter is well-constrained"
- "Inference appears robust to baseline perturbations"

**Don't say:**
- "This proves assemblies prefer reuse" (too causal)
- "The constraint is correct" (not testable)
- "Baseline is wrong" (too strong)

### Effect Size Interpretation

**Strong effects:**
- |θ| ≥ 0.7 (2× rate change) - Clear biological signal

**Moderate effects:**
- 0.3 ≤ |θ| < 0.7 (1.3-2× rate change) - Interpret with caution

**Weak effects:**
- |θ| < 0.3 (<1.3× rate change) - May be noise

### Sample Size Requirements

**Minimum recommended:**
- 100+ simulation samples
- 10+ observed compounds
- Diverse assembly depths

**For stronger signals:**
- 500+ samples (better θ estimation)
- 50+ compounds (more statistical power)

---

## Technical Details

### DiagnosticReport Base Class

All recipes return dataclasses inheriting from `DiagnosticReport`:

```python
class DiagnosticReport(ABC):
    severity: Literal["ok", "warning", "fail"]
    recommendation: str
    
    def print_summary(self) -> None: ...
    def to_dict(self) -> dict: ...
```

This ensures consistent interface across recipes.

### Importance Reweighting

Recipes use cached path data with importance reweighting to avoid
expensive resimulation. ESS (effective sample size) is monitored
to ensure weight validity.

### Confidence Intervals

Profile likelihood CIs use the likelihood ratio test:
- CI where LL > max_LL - 1.92 (χ²(1) / 2)

---

## See Also

- `../safety/` - Tier 1 automatic safety checks
- `../diagnostics/` - Diagnostic artifacts and suite
- `../validation/` - Internal validation and benchmarking scripts
- `../../docs/ASSEMBLY_USER_GUIDE.md` - Full user guide and API reference
