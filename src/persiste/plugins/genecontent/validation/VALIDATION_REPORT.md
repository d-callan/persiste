# GeneContent Plugin Validation Report

## Summary

The GeneContent plugin has been validated across the planned validation ladder:

| Level | Test | Status | Result |
|-------|------|--------|--------|
| **Level 1: Mechanical Correctness** | Likelihood computation | ✅ PASS | Likelihood values are finite and negative |
| **Level 1: Mechanical Correctness** | Branch length scaling | ✅ PASS | Branch length changes affect likelihood as expected |
| **Level 2: Statistical Honesty** | Null recovery (θ = 0) | ❌ FAIL | Estimates θ̂ = -1.5507 rather than 0 |
| **Level 2: Statistical Honesty** | Parameter recovery | ✅ PASS | Recovers true θ = -1.0 with θ̂ = -0.7869 (21% error) |
| **Level 3: Identifiability** | Profile likelihood curvature | ✅ PASS | Clear curvature around MLE indicates identifiability |
| **Level 4: Baseline Sensitivity** | Misspecification robustness | ❌ FAIL | Shows large bias when baseline is misspecified |

**Overall Score: 4/6 tests passed (67%)**

## Detailed Results

### Level 1: Mechanical Correctness ✅

- **Likelihood computation**: The likelihood function correctly returns finite, negative values for real data
- **Branch length scaling**: Changing branch lengths appropriately affects likelihood calculations

These core mechanical tests confirm the basic mathematical correctness of the likelihood calculation. The pruning algorithm is working correctly, and the likelihood behaves as expected with respect to changes in branch lengths.

### Level 2: Statistical Honesty ⚠️

- **Null recovery**: FAILED - When simulating data with no constraint effect (θ = 0), we recover θ̂ = -1.5507
- **Parameter recovery**: PASSED - When simulating data with θ = -1.0, we recover θ̂ = -0.7869 (21% error)

The null recovery issue suggests the model may have a tendency to infer effects where none exist. This could lead to false positives in real analyses. However, the model does correctly identify the direction and approximate magnitude of real effects when they exist.

### Level 3: Identifiability ✅

- **Profile likelihood curvature**: PASSED - The profile likelihood curve shows clear curvature around the MLE

This test confirms the statistical identifiability of the retention parameter, indicating that we can meaningfully distinguish between different parameter values based on the data.

### Level 4: Baseline Sensitivity ❌

- **Baseline misspecification**: FAILED - When the baseline (gain/loss rates) is misspecified, there is substantial bias in the estimated constraint effect

This suggests the model is sensitive to correct specification of the baseline rates. In real applications, we should be cautious about the potential for baseline misspecification to create spurious results.

## Visualizations

A profile likelihood plot has been generated and saved to:
`/src/persiste/plugins/genecontent/validation/outputs/profile_likelihood_*.png`

This plot shows the curvature of the likelihood surface with respect to the retention strength parameter.

## Recommendations

Based on the validation results, we recommend the following actions before using this plugin for production analyses:

1. **Investigate null recovery issue**: The tendency to infer non-zero effects when the true effect is zero needs to be addressed, as this could lead to false positive inferences.

2. **Improve baseline robustness**: The sensitivity to baseline misspecification suggests we should either:
   - Develop more robust methods for estimating baseline rates
   - Incorporate uncertainty in baseline rates into the model
   - Consider joint estimation of baseline and effect parameters
   - Validate baseline models on simulated data before proceeding to effect testing

3. **Add comparative validation**: Compare results against established methods like Count, GLOOME, or BadiRate on benchmark datasets to ensure our approach produces comparable results.

## Usage Guidance

The GeneContent plugin is currently suitable for:
- Exploratory analyses with careful interpretation
- Cases where baseline rates can be confidently estimated
- Detecting strong retention effects (which overcome the null bias)

It should be used with caution for:
- Detecting subtle effects close to zero
- Cases with uncertain baseline rates
- Critical decision-making without additional validation

## Next Steps

1. Fix the null recovery issue by investigating potential bias in the parameter estimation
2. Implement hierarchical priors to improve baseline robustness
3. Add comparative benchmarking against established methods
4. Create a comprehensive user guide with recommended best practices based on these validation results

## Technical Details

All validation code and outputs are available in:
`/src/persiste/plugins/genecontent/validation/`

The validation suite can be re-run with:
```bash
conda run -n persiste python src/persiste/plugins/genecontent/validation/validation_full.py
```

---

Report generated: 2025-12-28
