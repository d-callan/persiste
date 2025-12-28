# Assembly Plugin: Robustness Work Summary

**Date:** December 28, 2024  
**Status:** Experimental work - NOT production ready  
**Outcome:** Documented learnings, shelved for future iteration

---

## Original Problem

After initial validation showed assembly constraints were identifiable, we wanted to address robustness concerns:
- Prevent false positives under model misspecification
- Make baseline a first-class inferable object
- Add built-in diagnostics and safeguards
- Create "safe-by-default" inference API

---

## What We Tried

### Phase 1: Robustness Features (V1)

**Files created:**
- `src/persiste/plugins/assembly/baselines/baseline_family.py`
- `src/persiste/plugins/assembly/inference/constraint_result.py`
- `src/persiste/plugins/assembly/inference/robust_inference.py`
- `examples/assembly_robust_example.py`
- `docs/ASSEMBLY_USER_GUIDE.md`
- `docs/ASSEMBLY_TECHNICAL.md`
- `docs/ASSEMBLY_README.md`

**Features implemented:**
1. **BaselineFamily** - Inferable baseline parameters with priors
2. **ConstraintResult** - Structured results with diagnostics
3. **RobustConstraintInference** - Safe-by-default API with:
   - Joint baseline + constraint inference
   - Automatic null testing
   - L2 regularization
   - Profile likelihood diagnostics
   - Baseline sensitivity analysis
   - Cross-validation support

**What worked:**
- ✓ API design was clean and intuitive
- ✓ Diagnostics were comprehensive
- ✓ Evidence classification was helpful
- ✓ Null testing prevented overclaiming

**What didn't work:**
- ✗ **Performance:** 5-10 minutes per inference (vs 3 seconds for simple approach)
- ✗ **Bottleneck:** Repeated Gillespie simulation inside optimization loop
- ✗ **Root cause:** Every likelihood evaluation required fresh stochastic simulation

**Profiling results:**
```
Null fit:     ~120 seconds (1200 likelihood evaluations × 0.1s each)
Constrained:  ~120 seconds
Profile:      ~120 seconds (if run)
Total:        5-10 minutes
```

---

### Phase 2: Performance Optimization (V2)

**Files created:**
- `src/persiste/plugins/assembly/inference/state_cache.py`
- `src/persiste/plugins/assembly/inference/robust_inference_v2.py`
- `examples/test_fast_inference.py`
- `examples/profile_inference_bottleneck.py`
- `docs/ASSEMBLY_ARCHITECTURE_V2.md`
- `docs/ASSEMBLY_V2_SUMMARY.md`

**Architecture:**
1. **StateDistributionCache** - Pre-simulate states, reuse for likelihood
2. **Staged inference** - Fast/standard/thorough modes
3. **Conditional diagnostics** - Only run when ΔLL uncertain

**What worked:**
- ✓ Fast mode: 0.5s (720× speedup for null datasets)
- ✓ Caching concept sound for screening
- ✓ Conditional diagnostics saved computation

**What didn't work:**
- ✗ **Fundamental flaw:** Can't cache states at θ=0 and optimize θ away from 0
- ✗ **Invalid approximation:** Cached states only valid near reference point
- ✗ **Optimizer failure:** Returns θ=0 because cache doesn't reflect parameter changes
- ✗ **Without caching:** V2 is just as slow as V1 (15-20 min)

**Validation results:**
```
Test 1 (strong constraint): FAIL - Δ LL = -0.19 (should be >10)
Test 2 (null model):        PASS - Δ LL = 0.00 ✓
Test 3 (moderate):          PASS - Δ LL = 11.35 ✓ (but took 22 min)
```

**Why it failed:**
- Cached states represent p(state | θ=0)
- Optimizer needs p(state | θ) for arbitrary θ
- Without importance sampling, cache is useless for optimization
- With fresh simulation, no speed benefit

---

### Phase 3: Validation Against Original Baseline

**Files created:**
- `examples/validate_v2_quick.py`
- `examples/validate_v2_simple.py`
- `examples/validate_v2_vs_v1.py`
- `docs/ASSEMBLY_V2_VALIDATION.md`

**Key finding:**
The **original simple validation approach works perfectly:**
```python
# Direct likelihood evaluation (no complex framework)
ll_true = obs_model.compute_log_likelihood(observed_counts, latent_states_true)
ll_null = obs_model.compute_log_likelihood(observed_counts, latent_states_null)
delta_ll = ll_true - ll_null  # 9.38 (strong signal)
# Total time: 3 seconds
```

**Conclusion:**
- Original validation was correct
- Simple approach is fast enough (3s per evaluation)
- Complex robustness framework added problems without solving the core issue

---

## Core Learnings

### 1. The Caching Problem

**Attempted solution:** Cache states at reference parameters, reuse for optimization.

**Why it doesn't work:**
- Cached states: `{s₁, s₂, ..., sₙ} ~ p(state | θ_ref)`
- Likelihood needs: `p(data | θ) = Σ p(data | state) × p(state | θ)`
- Cache gives: `p(state | θ_ref)` not `p(state | θ)`
- **Invalid approximation** when θ ≠ θ_ref

**What would work:**
- **Importance sampling:** Reweight cached states using `p(state | θ) / p(state | θ_ref)`
- **Adaptive caching:** Refresh cache when parameters move too far
- **Multiple reference points:** Cache at grid of θ values
- **Deterministic approximation:** Mean-field ODE for screening, stochastic for validation

### 2. The Speed-Accuracy Tradeoff

**Fast but wrong:** Cached states at θ=0 → optimizer stuck at θ=0  
**Accurate but slow:** Fresh simulation → 15-20 min per inference  
**Need:** Importance sampling or other advanced technique

### 3. The Simplicity Lesson

**Complex framework (V1):**
- 1000+ lines of code
- 5-10 minutes runtime
- Many moving parts
- Hard to debug

**Simple approach (original):**
- 50 lines of code
- 3 seconds runtime
- Direct likelihood evaluation
- Easy to understand

**Lesson:** Don't add complexity without clear benefit.

---

## What Actually Works

### Validated Baseline (Original Approach)

**From:** `examples/assembly_validation_fixed.py`

```python
# 1. Generate data with known parameters
constraint_true = AssemblyConstraint(feature_weights=theta_true)
simulator = GillespieSimulator(graph, baseline, constraint_true)
observed_counts = simulate_observations(simulator, n_samples=100)

# 2. Evaluate likelihood under true parameters
latent_states_true = simulator.sample_final_states(n_samples=100)
ll_true = obs_model.compute_log_likelihood(observed_counts, latent_states_true)

# 3. Evaluate likelihood under null
simulator_null = GillespieSimulator(graph, baseline, null_constraint)
latent_states_null = simulator_null.sample_final_states(n_samples=100)
ll_null = obs_model.compute_log_likelihood(observed_counts, latent_states_null)

# 4. Compare
delta_ll = ll_true - ll_null  # Should be > 2 for identifiable constraints
```

**Performance:** 3 seconds  
**Result:** Δ LL = 9.38 (strong signal)  
**Status:** ✓ Validated and working

---

## Potentially Useful Code for Future

### 1. BaselineFamily Abstraction
**File:** `src/persiste/plugins/assembly/baselines/baseline_family.py`  
**Status:** Well-designed, could be useful  
**Use case:** If we want joint baseline+constraint inference later

### 2. ConstraintResult Dataclass
**File:** `src/persiste/plugins/assembly/inference/constraint_result.py`  
**Status:** Clean API, good structure  
**Use case:** Standardized result format with diagnostics

### 3. Evidence Classification Logic
**From:** `constraint_result.py`  
**Status:** Useful heuristic  
**Thresholds:**
- None: Δ LL < 2
- Weak: 2 ≤ Δ LL < 6
- Moderate: 6 ≤ Δ LL < 10
- Strong: Δ LL ≥ 10

---

## Files to Keep (WIP Directory)

**Potentially useful for future attempts:**
1. `baseline_family.py` - Clean abstraction
2. `constraint_result.py` - Good result structure
3. `state_cache.py` - Caching concept (needs importance sampling)

**Documentation worth keeping:**
1. `ASSEMBLY_USER_GUIDE.md` - Good conceptual overview
2. `ASSEMBLY_TECHNICAL.md` - Implementation details
3. This summary document

---

## Files to Remove

**Failed experiments:**
1. `robust_inference.py` - Too slow, not production ready
2. `robust_inference_v2.py` - Caching doesn't work without importance sampling
3. `assembly_robust_example.py` - Example for non-working code
4. `test_fast_inference.py` - Tests for failed V2
5. `validate_v2_*.py` - Validation of failed approaches
6. `profile_inference_bottleneck.py` - Profiling (findings documented here)

**Redundant documentation:**
1. `ASSEMBLY_ARCHITECTURE_V2.md` - V2 didn't work
2. `ASSEMBLY_V2_SUMMARY.md` - V2 didn't work
3. `ASSEMBLY_V2_VALIDATION.md` - V2 didn't work
4. `ASSEMBLY_README.md` - For non-working framework
5. `ASSEMBLY_PHASE1_*.md` - Interim summaries

---

## Recommendations for Next Attempt

### If we want to revisit robustness:

1. **Start with importance sampling**
   - Cache states at θ_ref
   - Reweight using `w(s) = p(s | θ) / p(s | θ_ref)`
   - Requires computing likelihood ratio for states
   - More complex but theoretically sound

2. **Or use deterministic approximation**
   - Mean-field ODE for screening
   - Stochastic simulation only for validation
   - Fast first pass, accurate second pass

3. **Or accept the 3-second baseline**
   - Current approach works
   - Fast enough for most use cases
   - Simple and maintainable

### If we want better diagnostics:

1. **Keep it separate from inference**
   - Don't bundle diagnostics with optimization
   - Run diagnostics as post-processing
   - User can choose when to run them

2. **Use the simple baseline**
   - Direct likelihood evaluation
   - Profile likelihood as separate function
   - Null testing as separate function

---

## Current Status

**Working and validated:**
- ✓ Original validation approach (`assembly_validation_fixed.py`)
- ✓ Direct likelihood evaluation (3s, Δ LL = 9.38)
- ✓ Constraints are identifiable
- ✓ No false positives on null data

**Not working:**
- ✗ Complex robustness framework (too slow)
- ✗ V2 caching optimization (fundamentally flawed)
- ✗ Joint baseline+constraint inference (not needed yet)

**Recommendation:**
- Use the simple validated baseline
- Shelve robustness work until we have importance sampling
- Document learnings (this file)
- Clean up experimental code

---

## Summary

We attempted to build a robust, safe-by-default inference framework with built-in diagnostics. The API design was good, but performance was unacceptable (5-10 min vs 3s baseline). 

We then tried to optimize with caching, but discovered a fundamental flaw: you can't cache states at one parameter value and use them to optimize over different values without importance sampling.

The original simple validation approach works perfectly and is fast enough. We should use that as the baseline and only revisit robustness if we implement proper importance sampling.

**Key lesson:** Don't add complexity without clear benefit. The simple approach works.
