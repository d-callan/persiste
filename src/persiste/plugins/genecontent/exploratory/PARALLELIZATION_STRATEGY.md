# Parallelization Strategy Analysis: Rust/C++ vs JAX

**Goal:** Parallelize the per-family likelihood loop for 10-100x speedup

---

## The Bottleneck

Current code (sequential):
```python
for fam_idx in range(n_families):  # 500 iterations
    # Compute transition matrices for this family
    # Run Felsenstein pruning
    # Accumulate log-likelihood
```

**Problem:** Each family is processed sequentially. With 500 families, we're doing 500 separate pruning computations that could run in parallel.

---

## Option 1: Rust/C++ with Rayon/OpenMP ⭐ RECOMMENDED

### Pros
1. **Trivial parallelization** - The per-family loop is embarrassingly parallel
2. **No algorithmic changes needed** - Keep existing pruning logic
3. **Excellent CPU utilization** - Rayon (Rust) or OpenMP (C++) handle thread pools automatically
4. **Predictable performance** - 5-10x speedup on 8-16 core machines
5. **Easy deployment** - Compile once, works everywhere
6. **Mature ecosystem** - PyO3 (Rust) and pybind11 (C++) are battle-tested
7. **Better for CPU-bound workloads** - Most users don't have GPUs

### Cons
1. **Compilation required** - Need Rust/C++ toolchain
2. **More code to maintain** - Separate language
3. **Debugging is harder** - Cross-language debugging
4. **No GPU support** - Stuck with CPU parallelism

### Implementation Complexity: **MEDIUM**
- Rust with PyO3: ~500 lines of code
- C++ with pybind11: ~400 lines of code
- Time estimate: 2-3 days

### Example (Rust with Rayon):
```rust
use rayon::prelude::*;

fn compute_likelihoods_parallel(
    families: &[FamilyData],
    tree: &Tree,
) -> Vec<f64> {
    families.par_iter()  // Parallel iterator
        .map(|family| {
            // Each family runs on a separate thread
            compute_single_family_likelihood(family, tree)
        })
        .collect()
}
```

**Speedup:** 5-10x on typical machines (8-16 cores)

---

## Option 2: JAX Vectorization

### Pros
1. **GPU support** - 50-100x speedup with CUDA
2. **Automatic differentiation** - Free gradients (2-5x faster optimization)
3. **Pure Python** - No compilation needed
4. **JIT compilation** - Optimizes automatically
5. **Future-proof** - GPU computing is the future

### Cons
1. **Requires algorithmic redesign** ⚠️ **MAJOR ISSUE**
   - Tree traversal doesn't vectorize naturally
   - Need to reformulate pruning as matrix operations
   - Complex to implement correctly
2. **Static shapes required** - JAX can't handle dynamic tree structures easily
3. **GPU not always available** - Many users CPU-only
4. **Numerical stability concerns** - Need careful testing
5. **Steep learning curve** - JAX paradigm is different

### Implementation Complexity: **HIGH**
- Need to redesign pruning algorithm
- Matrix formulation of tree traversal
- Extensive testing for correctness
- Time estimate: 1-2 weeks

### Why JAX is Hard for Pruning:
```python
# Current pruning (post-order traversal):
for node in post_order(tree):
    if node.is_tip:
        conditional[node] = tip_data[node]
    else:
        for child in node.children:  # Dynamic!
            P = transition_matrix[child]
            conditional[node] *= P @ conditional[child]

# JAX needs: Fixed computation graph, no loops, no conditionals
# This requires reformulating as matrix operations
```

**Speedup:** 
- CPU: 10-20x (if successfully vectorized)
- GPU: 50-100x (requires GPU hardware)

---

## Option 3: Hybrid Approach (Python multiprocessing)

### Pros
1. **Zero dependencies** - Built into Python
2. **Simple implementation** - Just wrap existing code
3. **Works immediately** - No compilation

### Cons
1. **Overhead is high** - Process spawning, pickling
2. **Limited speedup** - 2-4x typical (overhead dominates)
3. **Memory inefficient** - Each process copies data
4. **GIL doesn't matter here** - We're CPU-bound, not I/O-bound

### Implementation Complexity: **LOW**
```python
from multiprocessing import Pool

with Pool(processes=8) as pool:
    results = pool.map(compute_family_likelihood, families)
```

**Speedup:** 2-4x (overhead limits gains)

---

## Recommendation: **Rust with Rayon** 🎯

### Why Rust > C++
1. **Memory safety** - No segfaults, no undefined behavior
2. **Better Python integration** - PyO3 is excellent
3. **Modern tooling** - Cargo is better than CMake
4. **Growing ecosystem** - More momentum in scientific Python

### Why Rust > JAX (for now)
1. **Easier to implement** - No algorithmic redesign needed
2. **Predictable performance** - 5-10x guaranteed
3. **Works for everyone** - No GPU required
4. **Faster time to production** - 2-3 days vs 1-2 weeks

### Why Not Python multiprocessing
1. **Too much overhead** - Only 2-4x speedup
2. **Not worth the complexity** - If we're going parallel, do it right

---

## Implementation Plan: Rust Parallelization

### Phase 1: Rust Pruning Kernel (2-3 days)

**Step 1:** Create Rust crate structure
```
persiste-rust/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── tree.rs
│   ├── pruning.rs
│   └── parallel.rs
└── pyproject.toml
```

**Step 2:** Implement core pruning in Rust
- Tree structure (nodes, branches, topology)
- Felsenstein pruning algorithm
- Transition matrix computation

**Step 3:** Add Rayon parallelization
```rust
pub fn compute_all_families_parallel(
    tree: &Tree,
    families: Vec<FamilyData>,
    gain_rates: Vec<f64>,
    loss_rates: Vec<f64>,
) -> Vec<f64> {
    families.par_iter()
        .zip(gain_rates.par_iter())
        .zip(loss_rates.par_iter())
        .map(|((family, &gain), &loss)| {
            felsenstein_pruning(tree, family, gain, loss)
        })
        .collect()
}
```

**Step 4:** PyO3 bindings
```rust
#[pyfunction]
fn compute_likelihoods_rust(
    tree_newick: &str,
    presence_matrix: PyReadonlyArray2<i8>,
    gain_rates: PyReadonlyArray1<f64>,
    loss_rates: PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    // Convert Python objects to Rust
    // Call parallel computation
    // Return results
}
```

**Step 5:** Python integration
```python
# Fallback to NumPy if Rust not available
try:
    from persiste_rust import compute_likelihoods_rust
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

class GeneContentModel:
    def log_likelihood(self, parameters):
        if HAS_RUST:
            return self._log_likelihood_rust(parameters)
        else:
            return self._log_likelihood_numpy(parameters)
```

### Phase 2: Optimization & Testing (1 day)

- Benchmark against NumPy baseline
- Verify numerical correctness
- Profile Rust code for hotspots
- Tune Rayon thread pool size

### Phase 3: JAX Autodiff (Optional, 2-3 days)

Once Rust handles the heavy lifting, add JAX for gradients:
```python
import jax
from persiste_rust import compute_likelihoods_rust

@jax.custom_vjp
def log_likelihood_with_rust(params):
    # Forward pass: Use Rust
    ll = compute_likelihoods_rust(...)
    return ll

def log_likelihood_fwd(params):
    ll = log_likelihood_with_rust(params)
    return ll, params

def log_likelihood_bwd(params, g):
    # Backward pass: JAX autodiff
    grad = jax.grad(log_likelihood_numpy)(params)
    return (g * grad,)

log_likelihood_with_rust.defvjp(log_likelihood_fwd, log_likelihood_bwd)
```

This gives us:
- **Rust for forward pass** (5-10x speedup)
- **JAX for gradients** (2-5x faster optimization)
- **Best of both worlds**

---

## Expected Performance

### Current (NumPy, sequential)
- 50 taxa × 500 families: **38s** (with caching)

### With Rust + Rayon (8 cores)
- 50 taxa × 500 families: **5-8s** (5-8x speedup)
- 100 taxa × 1000 families: **30-40s** (vs 150s+ baseline)

### With Rust + JAX autodiff
- 50 taxa × 500 families: **3-5s** (8-12x total speedup)
- Faster optimization convergence

### With JAX full vectorization (if we solve it)
- 50 taxa × 500 families: **1-2s** (CPU) or **0.3-0.5s** (GPU)
- But requires 1-2 weeks of algorithm redesign

---

## Decision Matrix

| Criterion | Rust | JAX | Python MP |
|-----------|------|-----|-----------|
| **Speedup** | 5-10x | 10-100x* | 2-4x |
| **Implementation Time** | 2-3 days | 1-2 weeks | 1 day |
| **Complexity** | Medium | High | Low |
| **Deployment** | Easy | Easy | Easy |
| **GPU Support** | No | Yes | No |
| **Reliability** | High | Medium | High |
| **Maintenance** | Medium | Low | Low |

*JAX speedup assumes successful vectorization (not guaranteed)

---

## Recommendation

**Start with Rust + Rayon for immediate 5-10x gains, then add JAX autodiff for gradients.**

This gives us:
1. ✅ Guaranteed 5-10x speedup in 2-3 days
2. ✅ Works for all users (no GPU needed)
3. ✅ Can add JAX later for gradients (another 2-5x)
4. ✅ Leaves door open for full JAX vectorization if we solve it

**Don't pursue full JAX vectorization yet** - it's high risk, high reward. Get the Rust wins first, then revisit JAX if needed.

---

## Next Steps

1. **Create Rust crate skeleton** (30 min)
2. **Implement tree structure** (2 hours)
3. **Port pruning algorithm** (4 hours)
4. **Add Rayon parallelization** (1 hour)
5. **PyO3 bindings** (3 hours)
6. **Testing & benchmarking** (4 hours)

**Total: 2-3 days for 5-10x speedup** 🚀

Would you like me to start implementing the Rust parallelization?
