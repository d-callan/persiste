# CopyNumberDynamics Validation Framework

This document describes the three-tier validation framework for the CopyNumberDynamics plugin.

## Philosophy

Validation follows the principle: **"Think in three tiers: unit sanity → simulation recovery → empirical plausibility"**

This mirrors how HyPhy validates new methods - reviewers will immediately recognize this approach.

## Tier 1: Structural / Sanity Validation

**Goal:** Verify basic mathematical correctness ("this is not broken")

**Cost:** Cheap, must-have

### A. Rate Matrix Integrity

Tests that rate matrices are valid CTMCs:

- ✓ All off-diagonal rates ≥ 0
- ✓ Row sums = 0
- ✓ Forbidden transitions (e.g., 0→2) strictly zero
- ✓ Allowed transitions positive
- ✓ Constraints preserve validity
- ✓ Matrix exponential produces valid probabilities

**Implementation:** `tests/plugins/copynumber/test_structural_validation.py::TestRateMatrixIntegrity`

### B. Likelihood Monotonicity

Tests that constraints behave correctly:

- ✓ Constraining θ=0 recovers baseline LL exactly
- ✓ Turning constraints on cannot improve LL when θ=0
- ✓ Multipliers follow exp(θ) correctly

**Implementation:** `tests/plugins/copynumber/test_structural_validation.py::TestLikelihoodMonotonicity`

### C. Identifiability Smoke Test

Tests that parameters are distinguishable:

- ✓ Gain vs amplification produce distinct likelihood surfaces
- ✓ Loss vs contraction are not confounded
- ✓ Different constraint types produce different patterns

**Implementation:** `tests/plugins/copynumber/test_structural_validation.py::TestIdentifiabilitySmokeTest`

**Acceptance:** All tests must pass. If any fail, STOP.

## Tier 2: Simulation-Based Recovery (CORE)

**Goal:** "Simulate → Recover" - the plugin earns its keep here

**Cost:** Moderate, essential

### Minimal Simulation Grid (v1)

| Scenario | What you simulate | What should happen |
|----------|-------------------|-------------------|
| **Null** | Baseline only | No constraint significant |
| **Dosage buffering** | θ < 0 on all transitions | DosageStability detected |
| **Amplification bias** | θ > 0 amplify only | AmplificationBias detected |
| **High volatility lineage** | Lineage-specific θ > 0 | HostConditionedVolatility detected |
| **Mis-specified** | Amplify but test dosage | No significance |

### Key Acceptance Criteria

- ✓ **≥80% power** at reasonable branch lengths (0.3-1.0)
- ✓ **≤5% false positives** under null (with tolerance to 15% for small samples)
- ✓ **θ recovered within ~20-30%** of true value
- ✓ **Wrong constraint rejected** (Δ AIC > -2)

### Test Suite

**1. Null Scenario (False Positive Rate)**
- Simulate baseline only
- Test for spurious constraint detection
- Target: FPR ≤ 5% (allow 15% with small samples)

**2. Dosage Buffering Detection**
- Simulate θ = -0.5 on all transitions
- Fit with DosageStabilityConstraint
- Should detect (p < 0.05)

**3. Amplification Bias Detection**
- Simulate θ = 0.5 on amplify only
- Fit with AmplificationBiasConstraint
- Should detect (p < 0.05)

**4. Parameter Recovery**
- Simulate with known θ
- Profile likelihood over θ values
- Best θ within 30% of true value

**5. Wrong Constraint Rejection**
- Simulate dosage buffering
- Test with amplification bias constraint
- Should not be preferred (Δ AIC > -2)

**6. Statistical Power**
- Test at different branch lengths (0.1, 0.3, 0.5, 1.0)
- Test at different sample sizes (20, 50, 100, 200)
- Should detect at reasonable settings

**Implementation:** `tests/plugins/copynumber/test_simulation_recovery.py`

## Tier 3: Empirical Plausibility (Future)

**Goal:** Does this make biological sense on real data?

**Cost:** Expensive, publication-grade

### Planned Tests (v2+)

1. **Known biology recovery**
   - Essential genes → dosage buffered (θ < 0)
   - Drug resistance → amplification bias (θ > 0)
   - Antigen families → high volatility (θ > 0)

2. **Cross-validation with literature**
   - Compare to published CNV studies
   - Validate on well-characterized systems

3. **Robustness to misspecification**
   - Wrong bin thresholds
   - Missing CN calls
   - Tree uncertainty

**Status:** Not implemented in v1

## Running Validation

### Quick Test (Structural Only)

```bash
pytest tests/plugins/copynumber/test_structural_validation.py -v
```

### Full Simulation Suite

```bash
pytest tests/plugins/copynumber/test_simulation_recovery.py -v
```

### Complete Validation

```bash
python src/persiste/plugins/copynumber/validation/validation_runner.py
```

This runs all tiers and produces a comprehensive report.

## Validation Results

### Expected Output

```
======================================================================
COPY NUMBER DYNAMICS - VALIDATION SUITE
======================================================================

TIER 1: STRUCTURAL / SANITY VALIDATION
======================================================================

A. Rate Matrix Integrity
  ✓ Off-diagonal non-negative
  ✓ Row sums zero
  ✓ Forbidden transitions zero
  ✓ Allowed transitions positive
  ✓ Hierarchical baseline valid
  ✓ Constraints preserve validity

B. Likelihood Monotonicity
  ✓ θ=0 recovers baseline
  ✓ Constraints neutral at θ=0

C. Identifiability Smoke Test
  ✓ Gain vs amplify distinct
  ✓ Loss vs contract distinct
  ✓ Constraint types distinct

TIER 2: SIMULATION-BASED RECOVERY
======================================================================

1. Null Scenario (False Positive Rate)
  False positive rate: 10.0% (threshold: ≤15%)
  ✓ PASS

2. Dosage Buffering Detection
  p-value: 1.23e-05
  ✓ DETECTED

3. Amplification Bias Detection
  p-value: 3.45e-04
  ✓ DETECTED

4. Parameter Recovery
  True θ: -0.50
  Best θ: -0.50
  Relative error: 0.0%
  ✓ PASS (threshold: <30%)

5. Wrong Constraint Rejection
  Δ AIC: 1.23
  ✓ REJECTED

6. Statistical Power
  Detection at branch length 0.5: YES
  ✓ PASS (≥80% power expected)

======================================================================
VALIDATION SUMMARY
======================================================================

Tier 1 (Structural): 11/11 passed
Tier 2 (Simulation): 6/6 passed

Overall: 17/17 tests passed

✓ ALL VALIDATION TESTS PASSED

The CopyNumberDynamics plugin is validated and ready for use.
======================================================================
```

## Interpretation

### What Each Tier Tells You

**Tier 1 (Structural):**
- Plugin is mathematically sound
- No implementation bugs in core machinery
- Constraints work as designed

**Tier 2 (Simulation):**
- Plugin can recover known signals
- Statistical properties are correct
- Power is adequate for real applications
- False positive rate is controlled

**Tier 3 (Empirical):**
- Plugin makes biological sense
- Results align with known biology
- Robust to real-world complications

### When to Stop

**STOP if Tier 1 fails:** Fix implementation bugs first

**STOP if Tier 2 fails:** Fundamental statistical issues

**Proceed with caution if Tier 3 fails:** May need model refinement

## Continuous Validation

### When to Re-validate

- After any changes to core inference
- After adding new constraint types
- After modifying baseline models
- Before publication
- Periodically (e.g., quarterly)

### Regression Testing

All Tier 1 and key Tier 2 tests should be part of CI/CD:

```bash
# In CI pipeline
pytest tests/plugins/copynumber/ -v --tb=short
```

## Comparison to Other Tools

This validation framework follows best practices from:

- **HyPhy:** Simulation-based validation, power analysis
- **PAML:** Parameter recovery tests
- **IQ-TREE:** Likelihood surface profiling
- **MrBayes:** Null recovery, false positive control

Reviewers familiar with these tools will recognize the approach.

## Future Enhancements

### v1.1
- Add more simulation scenarios
- Test hierarchical baseline more thoroughly
- Add tree shape sensitivity tests

### v2.0
- Implement Tier 3 (empirical validation)
- Add cross-validation framework
- Benchmark against published CNV studies

### v3.0
- Bayesian validation (posterior coverage)
- Model adequacy tests
- Goodness-of-fit diagnostics

## References

This validation framework is inspired by:

1. Kosakovsky Pond et al. (2005) - HyPhy validation
2. Yang (2007) - PAML validation methodology
3. Huelsenbeck & Ronquist (2001) - Bayesian validation
4. Goldman & Yang (1994) - Simulation-based testing

## Contact

For questions about validation:
- See plugin README
- Check test implementations
- Review simulation framework code
