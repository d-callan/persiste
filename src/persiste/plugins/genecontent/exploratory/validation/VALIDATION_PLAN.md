# GeneContent Plugin Validation Plan

## Overview

This document outlines the comprehensive validation strategy for the GeneContent plugin.
Following the assembly plugin playbook, we validate the system before touching real data.

**Principle**: The system must be trustworthy before publication.

## Validation Ladder

### Level 1: Mechanical Correctness

**Goal**: Verify the likelihood computation is mathematically correct.

#### Tests:
1. **Likelihood matches known cases**
   - Single branch, single family: analytical solution
   - Two-state system: compare against hand calculation
   - Symmetric tree: verify symmetry properties

2. **Gain-only / loss-only limits**
   - When loss_rate → 0: only gains occur
   - When gain_rate → 0: only losses occur
   - Verify likelihood behavior at limits

3. **Branch length scaling**
   - Double all branch lengths → predictable likelihood change
   - Zero branch lengths → identity transition
   - Very long branches → equilibrium frequencies

**Output**: `validation/outputs/level1_mechanical.txt`

**Status**: ⬜ Not started

---

### Level 2: Statistical Honesty

**Goal**: Verify the inference procedure is statistically valid.

#### Tests:
1. **Null recovery**
   - Simulate data with θ = 0 (no constraint)
   - Fit model and recover θ̂
   - **Success criterion**: θ̂ ≈ 0 (within sampling error)

2. **Type I error rate**
   - Simulate 1000 datasets under null (θ = 0)
   - Test H₀: θ = 0 vs H₁: θ ≠ 0
   - **Success criterion**: Rejection rate ≈ 0.05 at α = 0.05

3. **No spurious constraints**
   - Simulate with correct baseline
   - Fit with same baseline
   - **Success criterion**: No false positives

**Output**: `validation/outputs/level2_statistical.txt`

**Status**: ⬜ Not started

---

### Level 3: Identifiability

**Goal**: Verify parameters are identifiable and don't trade off catastrophically.

#### Tests:
1. **Profile likelihoods show curvature**
   - Fix θ, optimize other parameters
   - Plot profile likelihood
   - **Success criterion**: Clear minimum, not flat

2. **Parameter recovery**
   - Simulate with known θ ≠ 0
   - Recover θ̂
   - **Success criterion**: θ̂ ≈ θ (correct sign and magnitude)

3. **Parameters don't trade off**
   - Check gain_rate vs loss_rate correlation
   - Check θ vs baseline_rate correlation
   - **Success criterion**: No perfect correlation

4. **Host association ≠ global retention**
   - Simulate host-associated constraint
   - Verify it's distinguishable from global retention
   - **Success criterion**: Different likelihood profiles

**Output**: `validation/outputs/level3_identifiability.txt`

**Status**: ⬜ Not started

---

### Level 4: Baseline Sensitivity

**Goal**: Verify robustness to baseline misspecification (critical given assembly results).

#### Tests:
1. **Slight baseline misspecification**
   - Simulate with baseline A
   - Fit with baseline B (slightly different)
   - **Success criterion**: θ̂ still recovers correctly

2. **Different hierarchical priors**
   - Test uniform baseline
   - Test birth-death baseline
   - Test empirical frequency baseline
   - **Success criterion**: Consistent results across baselines

3. **Global vs per-family baselines**
   - Simulate with per-family rates
   - Fit with global rate
   - **Success criterion**: Document bias, if any

4. **False positive inflation**
   - Misspecify baseline deliberately
   - Measure Type I error rate
   - **Success criterion**: Document which baselines inflate FP

**Output**: `validation/outputs/level4_baseline_sensitivity.txt`

**Status**: ⬜ Not started

---

## Validation Checklist

Before touching real data, all levels must pass:

- [ ] Level 1: Mechanical Correctness
  - [ ] Likelihood matches known cases
  - [ ] Gain-only / loss-only limits work
  - [ ] Branch length scaling correct

- [ ] Level 2: Statistical Honesty
  - [ ] Null recovery: θ = 0 → θ̂ ≈ 0
  - [ ] Type I error ≈ 0.05
  - [ ] No spurious constraints

- [ ] Level 3: Identifiability
  - [ ] Profile likelihoods show curvature
  - [ ] Parameter recovery: θ ≠ 0 → θ̂ ≈ θ
  - [ ] No catastrophic parameter trade-offs
  - [ ] Host association distinguishable

- [ ] Level 4: Baseline Sensitivity
  - [ ] Robust to slight misspecification
  - [ ] Consistent across hierarchical priors
  - [ ] Global vs per-family documented
  - [ ] False positive inflation documented

## Running Validation

```bash
# Run all validation tests
cd src/persiste/plugins/genecontent/validation
python run_validation.py --all

# Run specific level
python run_validation.py --level 1
python run_validation.py --level 2
python run_validation.py --level 3
python run_validation.py --level 4

# Generate validation report
python run_validation.py --report
```

## Validation Outputs

All outputs are saved to `validation/outputs/` with timestamps:
- `level1_mechanical_YYYYMMDD_HHMMSS.txt`
- `level2_statistical_YYYYMMDD_HHMMSS.txt`
- `level3_identifiability_YYYYMMDD_HHMMSS.txt`
- `level4_baseline_sensitivity_YYYYMMDD_HHMMSS.txt`
- `validation_report_YYYYMMDD_HHMMSS.pdf`

## Success Criteria Summary

| Level | Test | Success Criterion |
|-------|------|-------------------|
| 1 | Known cases | Likelihood matches analytical |
| 1 | Limits | Gain-only/loss-only behave |
| 1 | Scaling | Branch length effects correct |
| 2 | Null recovery | \|θ̂\| < 0.1 when θ = 0 |
| 2 | Type I error | 0.03 < rejection rate < 0.07 |
| 2 | Spurious | No false positives |
| 3 | Curvature | Profile LL has clear minimum |
| 3 | Recovery | \|θ̂ - θ\| / θ < 0.2 |
| 3 | Trade-offs | \|correlation\| < 0.9 |
| 3 | Distinguishable | ΔLL > 10 between models |
| 4 | Misspecification | \|bias\| < 0.15 |
| 4 | Priors | CV(θ̂) < 0.3 across baselines |
| 4 | Global vs family | Document bias direction |
| 4 | FP inflation | Type I < 0.10 |

## References

- Assembly plugin validation: `src/persiste/plugins/assembly/validation/`
- Simulation playbook: `examples/assembly/simulation_validation.py`
- Profile likelihood methods: `src/persiste/core/tree_inference.py`

## Notes

This validation framework is **required before publication**. Reviewers will ask about:
1. Parameter recovery
2. Type I error control
3. Baseline sensitivity
4. Comparison to existing methods

All validation outputs must be version-controlled and readily available.
