# GeneContent Performance Profiling Results

**Date:** December 29, 2025  
**Dataset:** 50 taxa × 500 families  
**Total Runtime:** ~61 seconds for retention test

---

## Executive Summary

Profiling identified **4 major bottlenecks** accounting for 95%+ of runtime:

1. **`expm()` - Matrix exponentiation (43% of simulation, 27% of inference)**
2. **`_compute_likelihood_numpy()` - Pruning algorithm (48% of inference)**
3. **`get_transition_matrix()` - Repeated computation (27% of inference, no caching)**
4. **`approx_derivative()` - Numerical gradients (52% of optimizer time)**

**Key Finding:** The per-family loop is the fundamental bottleneck - vectorization with JAX will provide 10-100x speedup.

---

## Detailed Profiling Results

### 1. Data Simulation (100 taxa × 1000 families)

**Total Time:** 9.18 seconds

| Function | Calls | Time (s) | % of Total | Notes |
|----------|-------|----------|------------|-------|
| `expm()` | 198,000 | 3.96 | **43%** | Matrix exponentiation |
| `issubdtype()` | 396,000 | 0.90 | 10% | Type checking overhead |
| `pick_pade_structure()` | 198,000 | 0.68 | 7% | Padé approximation setup |
| `pade_UV_calc()` | 198,000 | 0.44 | 5% | Padé computation |
| Other | - | 3.20 | 35% | Tree traversal, RNG, etc. |

**Key Insight:** `expm()` is called once per family per branch. For 1000 families × 198 branches = 198,000 calls.

**Optimization:** 
- ✅ **Consolidate simulation code** (DONE - now in `persiste.core.simulation`)
- 🔥 **JAX vectorization** - compute all transition matrices in parallel
- 🟡 **Analytical solution** for 2×2 matrices (faster than Padé)

---

### 2. Global Rate Estimation (50 taxa × 500 families)

**Total Time:** ~19 seconds

| Function | Calls | Time (s) | % of Total | Notes |
|----------|-------|----------|------------|-------|
| `minimize()` | 1 | 19.44 | 100% | Scipy L-BFGS-B optimizer |
| `log_likelihood()` | 45 | - | - | Called by optimizer |
| `get_transition_matrix()` | 2,871,000 | 9.77 | **50%** | No caching! |
| `_compute_likelihood_numpy()` | - | - | - | Pruning algorithm |
| `approx_derivative()` | 15 | 12.65 | **65%** | Numerical gradients |

**Key Insights:**
1. **2.87 million calls to `get_transition_matrix()`** - massive redundancy
2. **Numerical differentiation** takes 65% of optimizer time
3. Each likelihood evaluation processes 500 families sequentially

**Optimization:**
- 🔥 **LRU cache for transition matrices** - easy 2-3x speedup
- 🔥 **JAX autodiff** - eliminate numerical gradient computation
- 🔥 **Vectorize family loop** - compute all families in parallel

---

### 3. Retention Test (50 taxa × 500 families)

**Total Time:** 61.3 seconds (2 model fits + LRT)

| Function | Calls | Time (s) | % of Total | Notes |
|----------|-------|----------|------------|-------|
| `_compute_likelihood_numpy()` | 71,500 | 29.32 | **48%** | Pruning bottleneck |
| `get_transition_matrix()` | 7,078,500 | 16.44 | **27%** | Still no caching |
| `approx_derivative()` | 30 | 31.65 | **52%** | Numerical gradients |
| `minimize()` | 2 | 45.34 | 74% | Two optimization runs |
| `_compute_hessian()` | 2 | 16.08 | 26% | Standard error estimation |

**Key Insights:**
1. **7+ million transition matrix computations** - extreme redundancy
2. **Pruning is the core bottleneck** - 29s out of 61s total
3. **Two optimization runs** (null + alternative) double the cost

**Optimization:**
- 🔥 **JAX vectorization** - biggest impact here (10-100x)
- 🔥 **Cache transition matrices** - 2-3x speedup
- 🟡 **Warm start alternative model** from null parameters

---

## Bottleneck Analysis

### Bottleneck #1: Per-Family Sequential Loop ⚠️ CRITICAL

**Location:** `gene_inference.py:167`

```python
# Current: Sequential processing
for fam_idx, fam_name in enumerate(self.data.family_names):
    effect = self.constraint.get_effect(fam_name)
    # ... compute transition matrix
    result = self._pruning.compute_likelihood(...)
    total_ll += result.log_likelihood
```

**Problem:** 
- 500 families = 500 sequential likelihood calls
- Each call has Python overhead
- No parallelization

**JAX Solution:**
```python
# Proposed: Vectorized batch computation
transition_matrices = jax.vmap(compute_transition)(
    gain_rates, loss_rates, branch_lengths
)  # Shape: (n_families, n_branches, 2, 2)

log_likelihoods = jax.vmap(felsenstein_pruning)(
    transition_matrices, tip_data
)  # Shape: (n_families,)

total_ll = jnp.sum(log_likelihoods)
```

**Expected Speedup:**
- CPU: 10-15x (vectorization + JIT)
- GPU: 50-100x (parallel computation)

---

### Bottleneck #2: Matrix Exponentiation (`expm`) ⚠️ HIGH PRIORITY

**Location:** Called from `get_transition_matrix()` → `scipy.linalg.expm()`

**Problem:**
- Padé approximation is overkill for 2×2 matrices
- Called millions of times with no caching
- 43% of simulation time, 27% of inference time

**Solutions:**

**Option A: Analytical 2×2 Exponentiation** (fastest)
```python
def expm_2x2_analytical(Q, t):
    """Analytical matrix exponential for 2×2 rate matrices."""
    # For Q = [[-λ, λ], [μ, -μ]]
    # P(t) has closed-form solution
    λ, μ = Q[0,1], Q[1,0]
    total = λ + μ
    exp_term = np.exp(-total * t)
    
    P = np.array([
        [μ/total + λ/total * exp_term, λ/total * (1 - exp_term)],
        [μ/total * (1 - exp_term), λ/total + μ/total * exp_term]
    ])
    return P
```

**Option B: LRU Cache** (easy win)
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_transition_matrix_cached(gain, loss, t):
    Q = np.array([[-gain, gain], [loss, -loss]])
    return expm(Q * t)
```

**Option C: JAX** (best long-term)
```python
import jax.numpy as jnp
from jax.scipy.linalg import expm as jax_expm

# Vectorized over all branches and families
P_matrices = jax.vmap(jax.vmap(jax_expm))(Q_matrices * t_vectors)
```

**Expected Speedup:**
- Analytical: 5-10x faster than Padé
- LRU cache: 2-3x for global baseline
- JAX: 10-100x with vectorization

---

### Bottleneck #3: Numerical Differentiation ⚠️ HIGH PRIORITY

**Location:** `scipy.optimize.minimize()` → `approx_derivative()`

**Problem:**
- Finite differences require 2n+1 function evaluations per gradient
- 52% of optimizer time spent on gradients
- Inaccurate for poorly scaled problems

**JAX Solution:**
```python
import jax

# Automatic differentiation
@jax.jit
def log_likelihood_jax(params, data):
    # ... compute likelihood
    return ll

# Get gradient automatically
grad_fn = jax.grad(log_likelihood_jax)
gradient = grad_fn(params, data)  # Fast and exact!
```

**Expected Speedup:**
- 2-5x faster gradient computation
- More accurate gradients → faster convergence
- Enables second-order methods (Newton, L-BFGS with exact Hessian)

---

### Bottleneck #4: No Caching ⚠️ MEDIUM PRIORITY

**Problem:**
- `get_transition_matrix()` called 7+ million times
- Many calls with identical parameters
- No memoization

**Solution:**
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_transition_matrix(gain: float, loss: float, t: float):
    # Cache by (gain, loss, t) tuple
    # Helps when many families share rates
    pass
```

**Expected Speedup:**
- 2-3x for global baseline (all families same rates)
- 1.5-2x for per-family rates (some sharing)
- Minimal overhead (<1% memory)

---

## Code Consolidation Completed ✅

### Simulation Code Unified

**Before:** Duplicated 10+ times across validation scripts
- `validation/diagnose_bias.py`
- `validation/diagnose_bias_large.py`
- `validation/demo_small_dataset.py`
- `validation/null_calibration.py`
- `validation/validation_full.py`
- `analyses/demo_standard_analyses.py`
- `analyses/validation/tool_comparison_validation.py`
- ...and 3 more files

**After:** Single source of truth in `persiste.core.simulation`

```python
from persiste.core.simulation import simulate_binary_evolution

# Clean, reusable API
presence_matrix = simulate_binary_evolution(
    tree=tree,
    gain_rate=1.5,
    loss_rate=2.0,
    n_sites=100,
    rng=rng,
    site_specific_rates={0: (1.5, 0.5)}  # Family 0 has reduced loss
)
```

**Benefits:**
- ✅ Reduced codebase by ~500 lines
- ✅ Single place to optimize (JAX, Rust, etc.)
- ✅ Consistent behavior across all validation
- ✅ Better tested and documented
- ✅ Easier to maintain

---

## Recommended Optimization Roadmap

### Phase 1: Quick Wins (1-2 days) 🔥

1. **Add LRU cache to `get_transition_matrix()`**
   - Effort: 30 minutes
   - Impact: 2-3x speedup
   - Risk: None

2. **Use analytical 2×2 matrix exponential**
   - Effort: 2 hours
   - Impact: 5-10x faster than Padé
   - Risk: Low (well-known formula)

3. **Profile with different dataset sizes**
   - Effort: 1 hour
   - Impact: Confirm scaling behavior
   - Risk: None

**Expected Total Speedup:** 10-30x

---

### Phase 2: JAX Vectorization (1-2 weeks) 🔥🔥🔥

1. **Vectorize per-family likelihood loop**
   - Effort: 3-5 days
   - Impact: 10-50x speedup (CPU)
   - Risk: Medium (need to test numerical stability)

2. **JAX autodiff for gradients**
   - Effort: 2-3 days
   - Impact: 2-5x faster optimization
   - Risk: Low

3. **GPU support**
   - Effort: 1 day (if JAX already working)
   - Impact: 50-100x speedup
   - Risk: Low (deployment may need GPU)

**Expected Total Speedup:** 50-500x (CPU: 50-100x, GPU: 200-500x)

---

### Phase 3: Advanced Optimizations (optional)

1. **Rust pruning kernel**
   - Effort: 2 weeks
   - Impact: 5-10x without GPU
   - Risk: High (new dependency)

2. **Warm start for alternative models**
   - Effort: 1 day
   - Impact: 1.5-2x for LRT
   - Risk: Low

3. **Adaptive precision**
   - Effort: 1 week
   - Impact: 2-3x for large datasets
   - Risk: Medium

---

## Performance Projections

### Current Performance (50 taxa × 500 families)

| Operation | Time | Notes |
|-----------|------|-------|
| Simulation | 9s | 100 taxa × 1000 families |
| Global rates | 19s | Single optimization |
| Retention test | 61s | Two optimizations + LRT |

### After Phase 1 (Cache + Analytical expm)

| Operation | Time | Speedup | Notes |
|-----------|------|---------|-------|
| Simulation | 1s | **9x** | Analytical expm |
| Global rates | 6s | **3x** | Cached matrices |
| Retention test | 20s | **3x** | Cached matrices |

### After Phase 2 (JAX Vectorization, CPU)

| Operation | Time | Speedup | Notes |
|-----------|------|---------|-------|
| Simulation | 0.5s | **18x** | JAX JIT |
| Global rates | 1s | **19x** | Vectorized + autodiff |
| Retention test | 3s | **20x** | Vectorized + autodiff |

### After Phase 2 (JAX Vectorization, GPU)

| Operation | Time | Speedup | Notes |
|-----------|------|---------|-------|
| Simulation | 0.1s | **90x** | GPU parallel |
| Global rates | 0.3s | **63x** | GPU parallel |
| Retention test | 0.6s | **100x** | GPU parallel |

---

## Next Steps

1. ✅ **DONE:** Profile code and identify bottlenecks
2. ✅ **DONE:** Consolidate simulation code to `persiste.core.simulation`
3. 🔄 **IN PROGRESS:** Update remaining validation scripts
4. ⏭️ **NEXT:** Implement Phase 1 optimizations (cache + analytical expm)
5. ⏭️ **NEXT:** Design JAX vectorization strategy
6. ⏭️ **NEXT:** Implement JAX vectorization
7. ⏭️ **NEXT:** Benchmark and validate numerical stability

---

## Conclusion

The genecontent plugin has clear optimization opportunities:

**Immediate (Phase 1):** 10-30x speedup with simple caching and analytical solutions  
**Near-term (Phase 2):** 50-500x speedup with JAX vectorization  
**Long-term (Phase 3):** Additional 2-5x with advanced techniques

The per-family sequential loop is the fundamental bottleneck. JAX vectorization will provide the biggest impact, enabling:
- 10-50x speedup on CPU
- 50-100x speedup on GPU
- Automatic differentiation (faster, more accurate gradients)
- Easy deployment (no Rust/C++ compilation)

**Recommendation:** Proceed with Phase 1 (quick wins), then Phase 2 (JAX vectorization).
