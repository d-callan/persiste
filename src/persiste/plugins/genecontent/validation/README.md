# GeneContent Plugin Validation Framework

## Overview

This validation framework ensures the GeneContent plugin is **trustworthy before publication**.
Following the assembly plugin playbook, we validate mechanical correctness, statistical honesty,
identifiability, and baseline sensitivity before touching real data.

## Quick Start

```bash
# Navigate to validation directory
cd src/persiste/plugins/genecontent/validation

# Run all validation tests (recommended)
python run_validation.py --all

# Quick validation (fewer replicates, faster)
python run_validation.py --quick --all

# Run specific level
python run_validation.py --level 1
python run_validation.py --level 2 --replicates 200
```

## Validation Levels

### Level 1: Mechanical Correctness ⚙️

**Goal**: Verify likelihood computation is mathematically correct.

**Tests**:
- Likelihood matches analytical solutions
- Gain-only / loss-only limits behave correctly
- Branch length scaling is correct
- Zero branch lengths enforce identity

**Runtime**: ~1 minute

```bash
python run_validation.py --level 1
```

### Level 2: Statistical Honesty 📊

**Goal**: Verify inference is statistically valid.

**Tests**:
- Null recovery: θ = 0 → θ̂ ≈ 0
- Type I error rate ≈ 0.05 at α = 0.05
- No spurious constraints under correct baseline

**Runtime**: ~10 minutes (100 replicates)

```bash
python run_validation.py --level 2
python run_validation.py --level 2 --replicates 200  # More thorough
```

### Level 3: Identifiability 🎯

**Goal**: Verify parameters are identifiable.

**Tests**:
- Profile likelihoods show curvature
- Parameter recovery: θ ≠ 0 → θ̂ ≈ θ
- No catastrophic parameter trade-offs
- Host association distinguishable from global retention

**Runtime**: ~5 minutes

**Outputs**: Profile likelihood plots saved to `outputs/`

```bash
python run_validation.py --level 3
```

### Level 4: Baseline Sensitivity ⚠️

**Goal**: Test robustness to baseline misspecification (critical given assembly results).

**Tests**:
- Robustness to slight misspecification
- Consistency across hierarchical priors
- Global vs per-family baseline comparison
- **False positive inflation** under misspecification

**Runtime**: ~15 minutes (50 replicates)

```bash
python run_validation.py --level 4
python run_validation.py --level 4 --replicates 100  # More thorough
```

## Outputs

All validation outputs are saved to `outputs/` with timestamps:

```
outputs/
├── level1_mechanical_YYYYMMDD_HHMMSS.txt
├── level2_statistical_YYYYMMDD_HHMMSS.txt
├── level3_identifiability_YYYYMMDD_HHMMSS.txt
├── level4_baseline_sensitivity_YYYYMMDD_HHMMSS.txt
├── profile_likelihood_YYYYMMDD_HHMMSS.png
├── validation_report_YYYYMMDD_HHMMSS.txt
└── validation_summary_YYYYMMDD_HHMMSS.json
```

**These outputs are version-controlled and must be readily available for reviewers.**

## Success Criteria

| Level | Test | Criterion |
|-------|------|-----------|
| 1 | Known cases | Likelihood matches analytical (< 1e-6 error) |
| 1 | Limits | Gain-only/loss-only behave correctly |
| 1 | Scaling | Branch length effects are correct |
| 2 | Null recovery | \|θ̂\| < 0.1 when θ = 0 |
| 2 | Type I error | 0.03 < rejection rate < 0.07 |
| 2 | Spurious | ≤ 1 false positive in 10 tests |
| 3 | Curvature | Profile LL has clear minimum |
| 3 | Recovery | \|θ̂ - θ\| / \|θ\| < 0.2 |
| 3 | Trade-offs | \|correlation\| < 0.9 |
| 3 | Distinguishable | ΔLL > 10 between models |
| 4 | Misspecification | \|bias\| < 0.15 |
| 4 | Priors | CV(θ̂) < 0.3 across baselines |
| 4 | Global vs family | Bias direction documented |
| 4 | FP inflation | Type I < 0.10 under misspecification |

## Validation Checklist

Before using real data, **all levels must pass**:

- [ ] **Level 1: Mechanical Correctness**
  - [ ] Likelihood matches known cases
  - [ ] Gain-only / loss-only limits work
  - [ ] Branch length scaling correct

- [ ] **Level 2: Statistical Honesty**
  - [ ] Null recovery works
  - [ ] Type I error ≈ nominal
  - [ ] No spurious constraints

- [ ] **Level 3: Identifiability**
  - [ ] Profile likelihoods show curvature
  - [ ] Parameters recover correctly
  - [ ] No catastrophic trade-offs
  - [ ] Host association distinguishable

- [ ] **Level 4: Baseline Sensitivity**
  - [ ] Robust to slight misspecification
  - [ ] Consistent across priors
  - [ ] Global vs per-family documented
  - [ ] FP inflation documented

## Interpreting Results

### ✓ All Tests Pass

The system is **ready for real data**. Proceed to:
1. Document validation in publication
2. Compare against existing methods (Count, GLOOME, BadiRate)
3. Apply to real datasets

### ✗ Some Tests Fail

**DO NOT use real data** until issues are resolved:

1. **Level 1 failures**: Likelihood computation bug
2. **Level 2 failures**: Statistical procedure invalid
3. **Level 3 failures**: Parameters not identifiable
4. **Level 4 failures**: Baseline sensitivity issues

Review failed tests, fix issues, and re-run validation.

## Baseline Sensitivity (Critical)

Given assembly plugin results showing false positive inflation:

**You MUST document**:
- Which baselines are robust
- Which baselines inflate false positives
- Recommendations for practitioners

This is **not optional** - reviewers will ask.

## Comparison to Existing Methods

After passing all validation levels, compare against:
- **Count**: Dollo parsimony
- **GLOOME**: Gain-loss mapping
- **BadiRate**: Birth-death-innovation

On both simulated and small real datasets.

## Development Workflow

1. **Implement feature**
2. **Run quick validation**: `python run_validation.py --quick --all`
3. **Fix issues**
4. **Run full validation**: `python run_validation.py --all`
5. **Commit outputs** to version control
6. **Document in publication**

## Continuous Integration

Add to CI pipeline:

```yaml
- name: Run GeneContent Validation
  run: |
    cd src/persiste/plugins/genecontent/validation
    python run_validation.py --quick --all
```

## References

- **Assembly plugin validation**: `src/persiste/plugins/assembly/validation/`
- **Simulation playbook**: `examples/assembly/simulation_validation.py`
- **Validation plan**: `VALIDATION_PLAN.md`

## Questions?

See `VALIDATION_PLAN.md` for detailed methodology and rationale.

---

**Remember**: Validation makes the system trustworthy. Don't skip it.
