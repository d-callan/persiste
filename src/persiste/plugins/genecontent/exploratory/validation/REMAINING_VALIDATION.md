# GeneContent Validation: Remaining Tasks

Based on the validation requirements and our current progress, the following validation tasks remain to fully complete the validation ladder:

## Level 2: Statistical Honesty (Additional Tests)

1. **Type I Error Rate Test**
   - Run 100+ replicates of simulated data under the null hypothesis
   - For each replicate, perform LRT test at α = 0.05
   - Verify that proportion of rejections is ≈ 0.05
   - Success criterion: Rejection rate in range [0.03, 0.07]

2. **No Spurious Constraints Test**
   - Simulate data without constraint effects 
   - Test multiple random family sets for retention effect
   - Verify false positive rate matches nominal α level
   - Success criterion: False positive rate ≤ 1.5 × α

## Level 3: Identifiability (Additional Tests)

1. **Parameter Trade-Off Test**
   - Simulate data with known retention bias
   - Create grid of gain/loss rates and retention effects
   - Compute likelihood surface over this grid
   - Check for parameter trade-offs (ridge patterns in likelihood surface)
   - Success criterion: Clear peak in likelihood surface (not a ridge)

2. **Host Association Test**
   - Simulate data with host-specific retention effects
   - Test whether model can distinguish host association from global retention
   - Compare AIC of host-specific vs global models
   - Success criterion: Correct model has lowest AIC >80% of time

## Level 4: Baseline Sensitivity (Additional Tests)

1. **Hierarchical Prior Test**
   - Implement hierarchical Bayesian priors for rate variation
   - Compare inference with vs without hierarchical priors
   - Assess impact on parameter estimates and false positive rates
   - Success criterion: Hierarchical priors reduce bias under misspecification

2. **Per-Family vs Global Baseline Test**
   - Compare inferences with family-specific vs global gain/loss rates
   - Assess impact on constraint effect estimates
   - Evaluate computational cost vs statistical benefit
   - Success criterion: Per-family rates don't qualitatively change conclusions

3. **Baseline Sensitivity Analysis Framework**
   - Create standardized method to assess effect of baseline choices
   - Compare estimates across range of plausible baseline models
   - Report range of constraint effects consistent with data
   - Success criterion: Transparent reporting of sensitivity

## Comparative Validation

1. **Count Comparison**
   - Run Count on simulated datasets with known parameters
   - Compare parameter estimates with GeneContent
   - Assess relative accuracy and computational performance
   - Success criterion: Similar or better accuracy than Count

2. **GLOOME Comparison**
   - Run GLOOME on simulated datasets with known parameters
   - Compare inference results with GeneContent
   - Assess ability to detect similar patterns of gain/loss rate variation
   - Success criterion: Qualitatively similar conclusions on key test cases

3. **BadiRate Comparison**
   - Run BadiRate on simulated datasets with known parameters
   - Compare branch-specific rate estimates with GeneContent
   - Assess differences in model assumptions and impact
   - Success criterion: Understand and document key differences

## Implementation Plan

For each remaining test:

1. Implement test script extending validation_full.py
2. Add test function to ValidationSuite class
3. Run test and document results
4. Update VALIDATION_REPORT.md with findings

Priority order:
1. Level 2 additional tests (most critical for trustworthiness)
2. Level 3 parameter trade-off test (key for understanding limitations)
3. Level 4 baseline sensitivity tests (critical given assembly results)
4. Comparative validation (important for adoption and credibility)

## Timeline

| Task | Estimated Time | Priority |
|------|----------------|----------|
| Level 2 additional tests | 1-2 days | High |
| Level 3 additional tests | 2-3 days | Medium |
| Level 4 additional tests | 3-4 days | High |
| Comparative validation | 4-5 days | Medium |
| Final reporting | 1 day | Medium |

Total estimated time: 11-15 days
