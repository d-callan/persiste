# Codebase Cleanup Summary

## Overview

Cleaned up PERSISTE phylo plugin to remove abandoned optimization attempts and ensure only the JAX path remains.

---

## Files Removed

### Test/Benchmark Scripts (10 files)
All non-essential testing and benchmarking scripts removed from `tests/plugins/phylo/`:

- ❌ `benchmark_fel_perf.py` - Multi-site performance benchmark
- ❌ `benchmark_multiproc.py` - Multiprocessing test (failed)
- ❌ `benchmark_single_site_jax.py` - Single-site JAX benchmark
- ❌ `compare_fel_results.py` - Result comparison script
- ❌ `compare_fel_with_fitted_tree.py` - HyPhy comparison script
- ❌ `debug_fel_likelihood.py` - Debugging script
- ❌ `debug_pruning_detailed.py` - Pruning debug script
- ❌ `debug_pruning_order.py` - Pruning order debug script
- ❌ `test_fel_benchmark.py` - Benchmark test
- ❌ `test_fel_yokoyama.py` - Yokoyama-specific test

### Kept (Official Unit Tests)
- ✅ `test_fel.py` - Core FEL unit tests
- ✅ `test_fel_hyphy_comparison.py` - HyPhy comparison test
- ✅ `test_phylo_ctmc.py` - Observation model tests
- ✅ `test_pruning.py` - Pruning algorithm tests
- ✅ `test_*.py` - All other unit tests

---

## Code Changes

### 1. `fel.py` - Removed Multiprocessing

**Before:**
```python
from multiprocessing import Pool, cpu_count

def __init__(self, ..., n_jobs: Optional[int] = 1):
    self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()

def run(self):
    if self.n_jobs > 1:
        # Multiprocessing code...
        with Pool(processes=self.n_jobs) as pool:
            self.site_results = pool.map(...)
```

**After:**
```python
# No multiprocessing imports

def __init__(self, ..., use_site_patterns: bool = True):
    # No n_jobs parameter

def run(self):
    # Simple serial execution
    for site_idx in range(n_sites):
        result = self.analyze_site(site_idx)
```

**Reason:** Multiprocessing failed due to JAX pickling issues and had worse performance than serial execution.

---

### 2. `phylo_ctmc.py` - Complete Rewrite (JAX-Only)

**Before:** 395 lines with NumPy fallback
- `_ensure_pruning_initialized()` - NumPy pruning setup
- `log_likelihood()` - Generic likelihood method
- `log_likelihood_with_omega()` - NumPy-based
- `log_likelihood_with_alpha_beta()` - NumPy-based
- `site_log_likelihoods_with_omega()` - NumPy-based
- Fallback logic for non-JAX environments

**After:** 170 lines, JAX-only
- Single method: `site_log_likelihood_with_alpha_beta()` - JAX-accelerated
- Requires JAX (raises `ImportError` if not available)
- No NumPy fallback
- No backward compatibility code
- Clean, focused implementation

**Key Changes:**
```python
# Before
def __init__(self, ..., use_jax: bool = True):
    self.use_jax = use_jax and JAX_AVAILABLE
    self._pruning = None  # NumPy pruning
    self._pruning_jax = None  # JAX pruning

# After
def __init__(self, ...,):
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required...")
    self._pruning_jax = None  # Only JAX pruning
```

---

### 3. `pruning_jax.py` - Removed Factory Function

**Removed:**
```python
def create_pruning(tree, n_states, use_jax: bool = True):
    """Factory function with fallback to NumPy pruning"""
    if use_jax and JAX_AVAILABLE:
        return JAXFelsensteinPruning(tree, n_states)
    else:
        from .pruning import FelsensteinPruning
        return FelsensteinPruning(...)  # NumPy fallback
```

**Reason:** No longer needed since we're JAX-only. Direct instantiation of `JAXFelsensteinPruning` is clearer.

---

## Current State

### Architecture
```
┌─────────────────────────────────────┐
│         FELAnalysis                 │
│  - Serial execution only            │
│  - No multiprocessing               │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   PhyloCTMCObservationModel         │
│  - JAX-only (no NumPy fallback)     │
│  - Single method for FEL            │
│  - 170 lines (was 395)              │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   JAXFelsensteinPruning             │
│  - JIT-compiled pruning             │
│  - Fast likelihood computation      │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   MG94Baseline                      │
│  - Eigendecomposition caching       │
│  - Fast matrix exponential          │
└─────────────────────────────────────┘
```

### Dependencies
- **Required:** JAX, jaxlib
- **Removed:** multiprocessing (abandoned)
- **Removed:** NumPy pruning fallback

### Performance
- **Current:** ~10s per site with JAX (52 min for 300 sites)
- **Target:** ~1-2s per site with Rust (5-10 min for 300 sites)
- **Next step:** Rust implementation (see `RUST_IMPLEMENTATION_PLAN.md`)

---

## What Was Kept

### Optimizations That Work
1. ✅ **JAX JIT compilation** - 1.6x speedup
2. ✅ **Eigendecomposition caching** - Faster matrix exponential
3. ✅ **Observation model reuse** - No per-site object creation
4. ✅ **Site-indexed likelihood** - Efficient single-site computation

### Optimizations Removed
1. ❌ **Multiprocessing** - Failed due to JAX pickling, worse than serial
2. ❌ **NumPy fallback** - Unnecessary complexity, JAX is required anyway
3. ❌ **Transition matrix caching in pruning** - Superseded by eigendecomposition caching

---

## Testing

### Unit Tests Remain
All official unit tests in `tests/plugins/phylo/` are preserved:
- `test_fel.py`
- `test_fel_hyphy_comparison.py`
- `test_phylo_ctmc.py`
- `test_pruning.py`
- `test_site_patterns.py`
- `test_tree.py`
- `test_codon_model.py`

### To Verify
Run unit tests to ensure cleanup didn't break functionality:
```bash
cd /home/dcallan-adm/Documents/veg/persiste
conda run -n persiste pytest tests/plugins/phylo/ -v
```

---

## Documentation Updated

### New Documents
- `PERFORMANCE_OPTIMIZATIONS.md` - What we implemented (JAX, caching)
- `BENCHMARK_RESULTS.md` - Performance analysis (1.6x speedup, 52 min for 300 sites)
- `RUST_IMPLEMENTATION_PLAN.md` - Roadmap for Rust rewrite (target: 5-10 min)
- `CLEANUP_SUMMARY.md` - This document

### Key Takeaways
1. **JAX is now required** - No fallback to NumPy
2. **Serial execution only** - Multiprocessing removed
3. **Clean, focused codebase** - 170 lines vs 395 lines for observation model
4. **Ready for Rust** - Clean Python baseline to compare against

---

## Next Steps

1. **Verify tests pass** - Run pytest on phylo plugin
2. **Begin Rust implementation** - Follow `RUST_IMPLEMENTATION_PLAN.md`
3. **Phase 1 target** - Matrix exponential in Rust (2-3x speedup)
4. **Final target** - Full Rust core (5-10x speedup, matches HyPhy)
