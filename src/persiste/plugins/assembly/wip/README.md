# Robustness Experiments - WIP

**Status:** Experimental - NOT production ready  
**Date:** December 28, 2024

---

## Purpose

This directory contains experimental code from attempts to build a robust inference framework for the assembly plugin. The work is shelved pending implementation of importance sampling or other advanced techniques.

---

## What's Here

### Potentially Useful Components

1. **`baseline_family.py`** - Abstraction for inferable baseline parameters
   - Clean API design
   - Could be useful if we want joint baseline+constraint inference
   - Includes priors, bounds, parameter management

2. **`constraint_result.py`** - Structured result format with diagnostics
   - Evidence classification (none/weak/moderate/strong)
   - Profile likelihood diagnostics
   - Warnings and robustness scoring
   - Good for standardizing outputs

3. **`state_cache.py`** - State caching concept (NEEDS IMPORTANCE SAMPLING)
   - Idea: Pre-simulate states, reuse for likelihood
   - Problem: Only valid near reference point
   - Solution needed: Importance sampling to reweight

### Failed Experiments

1. **`robust_inference.py`** (V1)
   - Full-featured safe-by-default API
   - Problem: 5-10 minutes per inference (too slow)
   - Bottleneck: Repeated simulation in optimization loop

2. **`robust_inference_v2.py`** (V2)
   - Attempted caching optimization
   - Problem: Can't cache at θ=0 and optimize away from it
   - Needs importance sampling to work properly

### Example/Test Scripts

- `assembly_robust_example.py` - Example usage of V1 (slow)
- `test_fast_inference.py` - Performance tests for V2
- `validate_v2_*.py` - Validation scripts
- `profile_inference_bottleneck.py` - Profiling results

---

## Key Findings

### The Caching Problem

**What we tried:**
```python
# Cache states at θ=0
cache.populate(baseline, null_constraint, n_samples=100)

# Try to use for optimization at arbitrary θ
# PROBLEM: cached states don't reflect parameter changes
ll = compute_likelihood(observed, cached_states)  # Wrong!
```

**Why it doesn't work:**
- Cached states: `p(state | θ_ref=0)`
- Need: `p(state | θ)` for arbitrary θ
- Invalid approximation when θ ≠ θ_ref

**What would fix it:**
- Importance sampling: `w(s) = p(s|θ) / p(s|θ_ref)`
- Adaptive caching: Refresh when parameters move
- Deterministic approximation: ODE for screening

### Performance Results

| Approach | Runtime | Accuracy | Status |
|----------|---------|----------|--------|
| Original simple | 3s | ✓ Validated | ✓ Works |
| V1 (robust) | 5-10min | ✓ Accurate | ✗ Too slow |
| V2 (cached) | 0.5s | ✗ Wrong | ✗ Broken |
| V2 (no cache) | 15-20min | ✓ Accurate | ✗ Too slow |

---

## What Actually Works

The **original simple validation approach** from `examples/assembly_validation_fixed.py`:

```python
# Direct likelihood evaluation
simulator_true = GillespieSimulator(graph, baseline, constraint_true)
latent_states_true = simulator_true.sample_final_states(n_samples=100)
ll_true = obs_model.compute_log_likelihood(observed_counts, latent_states_true)

simulator_null = GillespieSimulator(graph, baseline, null_constraint)
latent_states_null = simulator_null.sample_final_states(n_samples=100)
ll_null = obs_model.compute_log_likelihood(observed_counts, latent_states_null)

delta_ll = ll_true - ll_null  # 9.38 (strong signal)
# Total time: 3 seconds ✓
```

---

## Recommendations

### For Future Attempts

1. **If revisiting robustness:**
   - Implement importance sampling first
   - Or use deterministic approximation (mean-field ODE)
   - Or accept that 3s is fast enough

2. **If adding diagnostics:**
   - Keep separate from inference
   - Post-processing, not bundled
   - User chooses when to run

3. **General principle:**
   - Don't add complexity without clear benefit
   - Simple working code > complex broken code

### What to Salvage

- `baseline_family.py` - Good abstraction
- `constraint_result.py` - Good structure
- Evidence classification thresholds (Δ LL: 2, 6, 10)

### What to Abandon

- Complex optimization frameworks
- Caching without importance sampling
- Bundled diagnostics

---

## See Also

- `docs/ROBUSTNESS_WORK_SUMMARY.md` - Comprehensive summary of all attempts
- `examples/assembly_validation_fixed.py` - Working baseline approach
- `docs/ASSEMBLY_SCALING_CURVES.md` - Original validation results
