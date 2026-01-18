# Assembly Plugin Code Review & Recommendations

**Date**: January 11, 2026  
**Reviewer**: Cascade AI  
**Scope**: Comprehensive review of assembly constraint inference plugin

---

## Executive Summary

The assembly plugin is **well-architected with clean separation of concerns**, but has a **critical mathematical bug** in the likelihood model that prevents theta recovery. The codebase is maintainable and mostly complete, but the inference pipeline cannot distinguish between different constraint parameter values due to a flawed likelihood formulation.

**Overall Scores**:
- **Robustness**: 6/10 - Good error handling and caching, but likelihood model is fundamentally flawed
- **Completeness**: 7/10 - Most components implemented, but key inference functionality broken
- **Maintainability**: 8/10 - Clean code and good structure, but needs better documentation of assumptions

---

## 🔴 CRITICAL BUG: Likelihood Model Flaw

### Location
`src/persiste/plugins/assembly/likelihood.py:81-110`

### The Bug

The likelihood function uses a **squared residual** that is always positive, removing directionality:

```python
# Lines 90-95 (CURRENT - WRONG)
baseline_reuse = 10.0  # Empirically observed null mean
reuse_residual = mean_reuse - baseline_reuse

# Likelihood contribution: reward observed reuse relative to baseline
# Positive theta increases reuse, so positive residual should increase LL
reuse_contrib = 0.5 * (reuse_residual**2) / variance_reuse
```

**Why this is wrong**:
- `reuse_contrib = 0.5 * residual²` is **always positive** regardless of residual sign
- If `mean_reuse = 20` (high), residual = +10, contrib = +50
- If `mean_reuse = 5` (low), residual = -5, contrib = +12.5
- **Both increase the likelihood!** The squared term removes directionality.

### Impact

When computing ΔLL between different theta values:
1. Each theta produces a different `mean_reuse` (verified working)
2. But `reuse_contrib = 0.5 * residual²` is a **parabola centered at baseline_reuse**
3. All theta values equidistant from baseline get the **same ΔLL**
4. The optimizer sees no gradient—all directions look equally good

**Validation Evidence**:
- ΔLL ≈ 4.1-4.3 across all theta ∈ {0.5, 1.0, 1.5, 2.0} (constant)
- Recovery rate = 0% (optimizer can't distinguish theta values)
- Signal detection = 100% (ΔLL > 0), but no discrimination

### Root Cause: Architectural Mismatch

The current model treats `mean_reuse` as an **observation** and fits a Gaussian to it. But `mean_reuse` is computed from **latent states**, not observed data.

**What we have**:
```python
LL(theta) = -0.5 * (mean_reuse(theta) - baseline)² / variance
# This is symmetric around baseline, so theta=+1 and theta=-1 look the same!
```

**What we need**:
```python
LL(theta) = log P(observed_data | latent_states(theta))
# Where latent_states(theta) changes with theta
```

The likelihood should compare the **latent state distribution** under different theta values, not just summary statistics.

### The Fix (Immediate)

**Option 1: Remove the squared term** (quick fix, but still wrong model)
```python
# Change from:
reuse_contrib = 0.5 * (reuse_residual**2) / variance_reuse

# To:
reuse_contrib = reuse_residual / math.sqrt(variance_reuse)
# This gives a linear relationship: higher reuse → higher LL
```

**Option 2: Use proper Gaussian log-likelihood** (better, but still not ideal)
```python
# Gaussian log-likelihood: -0.5 * (observed - predicted)² / variance
# But we need to model what "predicted" should be under each theta
reuse_contrib = -0.5 * ((mean_reuse - predicted_reuse(theta))**2) / variance_reuse
```

**Option 3: KL divergence between state distributions** (correct, but more work)
```python
# Compare latent state distributions directly
kl_div = sum(p_theta(s) * log(p_theta(s) / p_null(s)) for s in states)
LL_contribution = -kl_div  # Penalize divergence from null
```

---

## 🟡 ARCHITECTURAL ISSUES

### 1. Likelihood Model Doesn't Match Inference Goal

**Problem**: The observation model computes likelihood of **compound presence**, but constraints affect **transition rates**, not final presence/absence.

**Location**: `src/persiste/plugins/assembly/likelihood.py:106-109`

```python
# Current fallback (theta-independent):
if compound in primitives:
    record_contribs.append(math.log(0.9))  # Fixed probability
else:
    record_contribs.append(math.log(0.5))  # Fixed probability
```

**Impact**: 
- Constraints that change *how* assemblies form (reuse patterns) don't affect *which* assemblies are observed
- ΔLL stays constant because the set of observed compounds doesn't change with theta
- This is why even with constraint-specific features, discrimination fails

**Recommendation**: 
- Use **trajectory-level likelihoods** that incorporate transition statistics
- Or use **Poisson model** for transition counts (how many times each transition occurs)
- Or compute likelihood based on **state visitation frequencies** weighted by constraint

### 2. Feature Statistics Computed from Wrong Distribution

**Location**: `src/persiste/plugins/assembly/cli.py:80-141`

**Problem**: The `_compute_constraint_features` function weights features by **latent state probabilities**, but these probabilities are computed via **importance sampling reweighting** of cached trajectories.

The cache was simulated at `theta_ref` (usually null), so the feature counts in the cache reflect null dynamics. Reweighting changes state probabilities but **not the feature counts themselves**.

**Example**:
```python
# Cache simulated at theta=0 (null):
# - Trajectory 1: state_A (reuse_count=10)
# - Trajectory 2: state_B (reuse_count=15)

# Evaluate at theta=1.0:
# - Reweight: state_A gets prob=0.3, state_B gets prob=0.7
# - Weighted mean: 0.3*10 + 0.7*15 = 13.5

# But the reuse_count values (10, 15) were generated under theta=0!
# They don't reflect what would happen if we actually simulated at theta=1.0
```

**Impact**: Constraint-specific features are **approximations** based on reweighting, not true statistics from theta-specific simulations.

**Recommendation**:
- For critical evaluations, **resimulate** at the target theta instead of reweighting
- Or use the Rust simulator's ability to compute expected feature counts analytically
- Or acknowledge this as an approximation and validate its accuracy

### 3. Grid Search Optimizer is Too Simplistic

**Location**: `src/persiste/plugins/assembly/cli.py:465-502`

**Problem**: The optimizer uses a **fixed grid search** with aggressive regularization:

```python
# Lines 469-472
if best_delta_ll < regularization_threshold:  # threshold = 5.0
    theta_norm = sum(abs(v) for v in test_theta.values())
    if theta_norm > 2.0:  # L1 norm threshold
        continue  # Skip this theta
```

If initial ΔLL < 5.0 (which it always is with current likelihood), the optimizer refuses to explore theta values with L1 norm > 2.0.

**Impact**: Optimizer stays near zero even when true theta is large.

**Recommendation**:
- Remove or significantly relax regularization threshold (e.g., threshold = 0.5)
- Or make regularization adaptive based on signal strength
- Or use gradient-based optimization (scipy.optimize.minimize)

---

## 🟢 CODE QUALITY ASSESSMENT

### Strengths

1. **Well-Structured Architecture**
   - Clear separation: baseline → constraint → features → inference
   - Modular design with distinct responsibilities
   - Good use of dataclasses and type hints
   - Example: `AssemblyConstraint` cleanly separates feature weights from feature extraction

2. **Comprehensive Documentation**
   - Docstrings explain design principles
   - Examples in constraint class show usage patterns
   - Comments explain non-obvious logic
   - Example: `compute_observation_ll` has detailed design principles in docstring

3. **Robust Caching System**
   - Importance sampling with ESS monitoring
   - Trust region validation
   - Automatic resimulation when cache invalid
   - Location: `src/persiste/plugins/assembly/observation/cached_observation.py`

4. **Good Test Coverage** (after recent additions)
   - Null calibration tests
   - Positive-signal tests
   - Feature extraction tests
   - Validation scripts for power analysis
   - Location: `tests/plugins/assembly/test_positive_signal.py`

5. **Safety Checks**
   - Tier 1 safety checks catch common issues
   - ΔLL threshold adjustment when unsafe
   - Logging for diagnostics
   - Location: `src/persiste/plugins/assembly/safety/`

### Weaknesses

1. **Likelihood Model Mismatch** (critical)
   - Wrong mathematical formulation (squared residual)
   - Doesn't capture constraint effects on dynamics
   - No gradient for optimizer to follow

2. **Limited Optimizer**
   - Grid search only (no gradient-based methods)
   - Aggressive regularization prevents exploration
   - No adaptive step sizing
   - Fixed grid: only tests {-1.0, -0.5, +0.5, +1.0} per feature

3. **Feature Engineering Assumptions**
   - Hardcoded `baseline_reuse = 10.0` (magic number)
   - Assumes Gaussian distribution for features
   - No validation that assumptions hold
   - Location: `likelihood.py:90`

4. **Incomplete Error Handling**
   - Many functions return empty dicts on failure without clear error messages
   - Silent fallbacks (e.g., `observation_summary or {}`)
   - No validation of input constraints
   - Example: `_extract_feature_summary` returns `{}` on exception

5. **Performance Concerns**
   - Recomputes features for every theta evaluation
   - No memoization of expensive operations
   - Grid search scales poorly with feature dimensionality
   - Example: 4 features × 4 values = 16 evaluations minimum

### Maintainability

**Good**:
- Consistent code style
- Clear naming conventions
- Modular structure makes changes localized
- Type hints aid understanding
- Separation of concerns (baseline, constraint, features, inference)

**Needs Improvement**:
- Magic numbers scattered throughout (0.9, 0.5, 10.0, 5.0, 2.0)
- Complex likelihood function (191 lines) should be split
- Limited unit test coverage for individual components
- No integration tests for end-to-end workflows
- Insufficient documentation of mathematical assumptions

---

## 📊 COMPLETENESS ASSESSMENT

### Implemented & Working ✅

- Baseline rate models (join, split, decay)
- Feature extraction (core + symmetry breaks)
- Constraint contribution computation
- Rust simulator integration
- Importance sampling cache
- Deterministic screening
- Safety checks
- Null calibration tests

### Implemented But Broken ⚠️

- Stochastic inference (likelihood doesn't discriminate)
- Theta recovery (optimizer can't find true values)
- Constraint-specific feature extraction (reweighting approximation)
- Power analysis (no recovery at any tested parameter values)

### Missing ❌

- Gradient-based optimization
- Confidence intervals / uncertainty quantification
- Model selection (AIC/BIC computed but not used)
- Cross-validation
- Observation model that captures transition-level statistics
- Trajectory-level likelihood computation
- Analytical feature count computation

---

## 🎯 RECOMMENDATIONS

### Immediate Fixes (High Priority) 🔴

#### 1. Fix Likelihood Model

**File**: `src/persiste/plugins/assembly/likelihood.py:81-110`

**Current code**:
```python
reuse_contrib = 0.5 * (reuse_residual**2) / variance_reuse
```

**Quick fix** (linear relationship):
```python
# Option A: Linear scoring (simple, directional)
reuse_contrib = reuse_residual / math.sqrt(variance_reuse)
```

**Better fix** (proper likelihood):
```python
# Option B: Model predicted reuse as function of theta
# predicted_reuse = baseline_reuse + theta_effect
# Then use Gaussian log-likelihood
predicted_reuse = baseline_reuse + sum(theta[f] * feature_sensitivity[f] for f in features)
reuse_contrib = -0.5 * ((mean_reuse - predicted_reuse)**2) / variance_reuse
```

**Best fix** (KL divergence):
```python
# Option C: Compare state distributions directly
# This requires more refactoring but is mathematically correct
kl_contrib = sum(
    p_theta[state] * math.log(p_theta[state] / p_null[state])
    for state in latent_states
)
```

**Action**: Start with Option A (linear), validate it works, then consider Option B or C.

#### 2. Relax Optimizer Regularization

**File**: `src/persiste/plugins/assembly/cli.py:465-477`

**Current code**:
```python
regularization_threshold = 5.0
if best_delta_ll < regularization_threshold:
    theta_norm = sum(abs(v) for v in test_theta.values())
    if theta_norm > 2.0:  # L1 norm threshold
        continue
```

**Fix**:
```python
# Option A: Remove regularization entirely (for testing)
# (Just comment out the if block)

# Option B: Make it much less aggressive
regularization_threshold = 0.5  # Down from 5.0
if best_delta_ll < regularization_threshold:
    theta_norm = sum(abs(v) for v in test_theta.values())
    if theta_norm > 10.0:  # Up from 2.0
        continue

# Option C: Make it adaptive
if best_delta_ll < 0.1:  # Only regularize if no signal at all
    theta_norm = sum(abs(v) for v in test_theta.values())
    if theta_norm > 5.0:
        continue
```

**Action**: Start with Option A (remove), then add back adaptive regularization if needed.

#### 3. Add Validation Tests

**File**: `tests/plugins/assembly/test_positive_signal.py`

**Add test**:
```python
def test_likelihood_increases_with_theta():
    """ΔLL should increase monotonically with theta (for positive effects)."""
    results = []
    for theta_val in [0.0, 0.5, 1.0, 1.5]:
        result = fit_assembly_constraints(
            observed_compounds={"A", "B"},
            primitives=["A", "B"],
            mode=InferenceMode.FULL_STOCHASTIC,
            feature_names=["reuse_count"],
            n_samples=100,
            seed=42,
        )
        results.append((theta_val, result.get("stochastic_delta_ll")))
    
    # ΔLL should increase with theta
    delta_lls = [dll for _, dll in results]
    assert delta_lls == sorted(delta_lls), (
        f"ΔLL should increase with theta: {results}"
    )
```

**Action**: Add this test, verify it fails with current code, passes after fix.

### Medium-Term Improvements (Next Sprint) 🟡

#### 4. Better Observation Model

**Goal**: Incorporate transition-level statistics, not just final state presence.

**Options**:
- **Trajectory likelihood**: Compute P(trajectory | theta) for each observed trajectory
- **Transition counts**: Use Poisson model for how many times each transition occurs
- **State visitation**: Weight by how long trajectories spend in each state

**File to modify**: `src/persiste/plugins/assembly/likelihood.py`

**Estimated effort**: 2-3 days

#### 5. Performance Optimization

**Goals**:
- Memoize feature computations
- Parallelize grid search
- Cache likelihood evaluations

**Files to modify**:
- `src/persiste/plugins/assembly/cli.py` (add caching)
- `src/persiste/plugins/assembly/likelihood.py` (memoization)

**Estimated effort**: 1-2 days

#### 6. Code Quality Improvements

**Goals**:
- Extract magic numbers to named constants
- Split large functions (especially `compute_observation_ll`)
- Add more unit tests for individual components
- Document mathematical assumptions

**Files to modify**: Multiple

**Estimated effort**: 2-3 days

### Long-Term Architecture (Future) 🔵

#### 7. Gradient-Based Optimization

**Goal**: Replace grid search with scipy.optimize.minimize

**Benefits**:
- Faster convergence
- Can handle higher-dimensional theta
- Automatic step sizing

**Requirements**:
- Implement gradient computation (or use numerical gradients)
- Ensure likelihood is smooth and differentiable

**Estimated effort**: 3-5 days

#### 8. Bayesian Inference

**Goal**: Implement MCMC or variational inference for posterior sampling

**Benefits**:
- Uncertainty quantification
- Model averaging
- Better handling of multimodal posteriors

**Requirements**:
- Prior specification
- MCMC sampler (e.g., emcee, PyMC)
- Convergence diagnostics

**Estimated effort**: 1-2 weeks

#### 9. Analytical Feature Computation

**Goal**: Use Rust simulator to compute expected feature counts analytically

**Benefits**:
- No need for importance sampling approximation
- Exact feature statistics under each theta
- Faster evaluation

**Requirements**:
- Extend Rust simulator API
- Implement analytical computation in Rust
- Wire into Python inference

**Estimated effort**: 1 week

---

## 📋 IMPLEMENTATION PLAN

### Phase 1: Critical Fixes (This Session)

1. **Fix likelihood model** (30 min)
   - Change squared residual to linear relationship
   - Test that ΔLL now varies with theta

2. **Relax regularization** (10 min)
   - Remove or significantly increase thresholds
   - Test that optimizer explores larger theta values

3. **Add validation test** (20 min)
   - Test that ΔLL increases monotonically with theta
   - Verify test fails before fix, passes after

4. **Run validation scripts** (10 min)
   - Rerun reuse-only overdrive
   - Rerun power grid
   - Verify recovery rate > 0%

**Total estimated time**: 70 minutes

### Phase 2: Validation & Testing (Next Session)

5. **Comprehensive testing**
   - Test with multiple seeds
   - Test with different effect sizes
   - Test with different sample sizes

6. **Document findings**
   - Update validation plan
   - Document what works and what doesn't
   - Identify remaining issues

### Phase 3: Medium-Term Improvements (Future)

7. **Better observation model**
8. **Performance optimization**
9. **Code quality improvements**

---

## 🔍 DETAILED FILE-BY-FILE ANALYSIS

### `src/persiste/plugins/assembly/likelihood.py`

**Purpose**: Compute observation log-likelihood given latent states

**Issues**:
- **Line 95**: Squared residual bug (critical)
- **Line 90**: Magic number `baseline_reuse = 10.0`
- **Lines 106-109**: Theta-independent fallback
- **Function length**: 191 lines (too long, should split)

**Strengths**:
- Good documentation
- Handles multiple observation models
- ESS ratio weighting

**Recommendations**:
- Fix squared residual (immediate)
- Extract constants to module-level
- Split into smaller functions
- Add unit tests for each branch

### `src/persiste/plugins/assembly/cli.py`

**Purpose**: Main inference API

**Issues**:
- **Lines 469-472**: Aggressive regularization (critical)
- **Lines 80-141**: Feature extraction from reweighted cache (architectural)
- **Lines 465-502**: Simple grid search (limitation)
- **Function length**: `fit_assembly_constraints` is 400+ lines

**Strengths**:
- Well-documented
- Modular structure
- Good error handling
- Multiple inference modes

**Recommendations**:
- Relax regularization (immediate)
- Consider gradient-based optimization (medium-term)
- Split into smaller functions (medium-term)
- Add more logging for debugging

### `src/persiste/plugins/assembly/constraints/assembly_constraint.py`

**Purpose**: Constraint model (theta → rate multipliers)

**Issues**:
- None critical
- Some unused methods (e.g., `get_constrained_baseline`)

**Strengths**:
- Clean interface
- Good documentation
- Factory methods for common cases
- Type hints throughout

**Recommendations**:
- No immediate changes needed
- Consider removing unused methods (low priority)

### `src/persiste/plugins/assembly/observation/cached_observation.py`

**Purpose**: Importance sampling cache for stochastic inference

**Issues**:
- Feature counts reflect theta_ref, not target theta (architectural)
- No analytical feature computation

**Strengths**:
- Robust caching logic
- ESS monitoring
- Trust region validation
- Automatic resimulation

**Recommendations**:
- Document the reweighting approximation limitation
- Consider adding analytical feature computation (long-term)
- Add tests for cache invalidation logic

### `src/persiste/plugins/assembly/screening/screening.py`

**Purpose**: Deterministic screening for hypothesis triage

**Issues**:
- Variance estimation is heuristic (`1.0 + sum_sq`)
- No constraint-specific features in screening

**Strengths**:
- Adaptive grid search
- Normalized ΔLL criterion
- Budget management

**Recommendations**:
- Integrate constraint-specific features (medium-term)
- Improve variance estimation (medium-term)
- Add tests for grid generation

---

## 📚 REFERENCES

### Key Concepts

- **ΔLL (Delta Log-Likelihood)**: Difference in log-likelihood between a hypothesis and the null model
- **Importance Sampling**: Reweighting trajectories from one distribution to approximate another
- **ESS (Effective Sample Size)**: Measure of importance sampling quality
- **Trust Region**: Region around theta_ref where importance sampling is valid
- **Screening**: Fast deterministic evaluation to filter hypotheses before expensive stochastic inference

### Related Documentation

- `docs/ASSEMBLY_VALIDATION_PLAN.md`: Validation strategy
- `docs/development/PHASE_1_2_COMPLETE.md`: Implementation history
- `docs/ASSEMBLY_DIAGNOSTICS_PLAN.md`: Diagnostic procedures

---

## 📝 CHANGELOG

### 2026-01-11: Initial Code Review
- Identified critical likelihood model bug
- Documented architectural issues
- Provided detailed recommendations
- Created implementation plan

---

## ✅ NEXT STEPS

1. **Immediate** (this session):
   - Fix likelihood model squared residual bug
   - Relax optimizer regularization
   - Add validation test
   - Run validation scripts

2. **Short-term** (next session):
   - Comprehensive testing with multiple configurations
   - Document what works and remaining issues
   - Prioritize medium-term improvements

3. **Medium-term** (next sprint):
   - Implement better observation model
   - Optimize performance
   - Improve code quality

4. **Long-term** (future):
   - Gradient-based optimization
   - Bayesian inference
   - Analytical feature computation
