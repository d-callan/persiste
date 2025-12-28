# Rust Implementation Plan for PERSISTE FEL

## Problem Statement

**Current Performance:**
- Python + JAX: **10.4s per site** → 52 minutes for 300 sites
- HyPhy (C++): **~1-2s per site** → 5-10 minutes for 300 sites

**Issue:** 52 minutes is too slow for iterative development and testing. Need **5-10x speedup** to match HyPhy.

**Solution:** Rewrite performance-critical paths in Rust with Python bindings.

---

## Architecture

### Hybrid Approach: Python + Rust Core

```
┌─────────────────────────────────────────┐
│         Python Layer (High-Level)        │
│  - FELAnalysis orchestration             │
│  - Data loading/parsing                  │
│  - Result formatting/output              │
│  - API/CLI interfaces                    │
└──────────────┬──────────────────────────┘
               │ PyO3 bindings
┌──────────────▼──────────────────────────┐
│         Rust Core (Performance)          │
│  - Matrix exponential (expm)             │
│  - Felsenstein pruning                   │
│  - Likelihood computation                │
│  - Optimization inner loop               │
└─────────────────────────────────────────┘
```

**Benefits:**
- Keep Python for high-level logic (easy to modify)
- Rust for computational bottlenecks (C++ speed)
- Clean interface via PyO3
- Type safety and memory safety from Rust

---

## Critical Paths to Rewrite

### Priority 1: Matrix Exponential (Highest Impact)

**Current bottleneck:** Computing `P(t) = expm(Q*t)` for ~75 branches, 100-200 times per site

**Rust implementation:**
```rust
// Use nalgebra for linear algebra
use nalgebra::{DMatrix, DVector};

pub fn matrix_exponential(
    Q: &DMatrix<f64>,
    t: f64,
) -> DMatrix<f64> {
    // Eigendecomposition: Q = V D V^-1
    let eigen = Q.symmetric_eigen();
    
    // P(t) = V * diag(exp(λ_i * t)) * V^-1
    let exp_diag = eigen.eigenvalues.map(|λ| (λ * t).exp());
    
    &eigen.eigenvectors * DMatrix::from_diagonal(&exp_diag) * eigen.eigenvectors.transpose()
}

// Cached version for MG94
pub struct MG94Cache {
    eigen_values: DVector<f64>,
    eigen_vectors: DMatrix<f64>,
    eigen_vectors_inv: DMatrix<f64>,
}

impl MG94Cache {
    pub fn fast_expm(&self, alpha: f64, beta: f64, t: f64) -> DMatrix<f64> {
        // Exact computation for Q(α,β)
        // Much faster than Python
    }
}
```

**Expected speedup:** 5-10x for matrix operations

---

### Priority 2: Felsenstein Pruning

**Current bottleneck:** Nested loops over tree nodes and states

**Rust implementation:**
```rust
pub struct PruningAlgorithm {
    tree_structure: Vec<(usize, usize, usize)>, // (parent, child1, child2)
    branch_lengths: Vec<f64>,
    n_states: usize,
}

impl PruningAlgorithm {
    pub fn compute_likelihood(
        &self,
        alignment: &[Vec<usize>],
        transition_matrices: &[DMatrix<f64>],
        frequencies: &DVector<f64>,
    ) -> f64 {
        // Vectorized operations
        // No Python overhead
        // Cache-friendly memory layout
    }
}
```

**Expected speedup:** 3-5x for pruning

---

### Priority 3: Optimization Loop

**Current bottleneck:** scipy.optimize overhead, many Python function calls

**Rust implementation:**
```rust
use argmin::core::{CostFunction, Executor};
use argmin::solver::linesearch::MoreThuente;
use argmin::solver::quasinewton::LBFGS;

pub struct FELObjective {
    pruning: PruningAlgorithm,
    mg94_cache: MG94Cache,
    site_data: Vec<usize>,
    // ... other data
}

impl CostFunction for FELObjective {
    type Param = Vec<f64>; // [alpha, beta]
    type Output = f64;     // negative log-likelihood
    
    fn cost(&self, params: &Self::Param) -> Result<Self::Output> {
        // Compute -log L(α, β) entirely in Rust
        // No Python boundary crossing
    }
}

pub fn fit_site(
    site_data: &[usize],
    pruning: &PruningAlgorithm,
    mg94_cache: &MG94Cache,
) -> (f64, f64, f64) {
    // Returns (alpha_mle, beta_mle, log_lik)
    // Entire optimization in Rust
}
```

**Expected speedup:** 2-3x for optimization

---

## Implementation Phases

### Phase 1: Matrix Exponential (Week 1)
**Goal:** Replace scipy.linalg.expm with Rust implementation

**Tasks:**
1. Set up Rust project with PyO3
2. Implement basic matrix exponential
3. Add MG94 eigendecomposition caching
4. Create Python bindings
5. Benchmark against scipy

**Deliverable:** `persiste_rust` Python package with `matrix_exponential()` function

**Expected result:** 2-3x speedup (52 min → 20-25 min)

---

### Phase 2: Pruning Algorithm (Week 2)
**Goal:** Replace Python pruning with Rust implementation

**Tasks:**
1. Port tree structure representation
2. Implement conditional likelihood computation
3. Vectorize operations
4. Optimize memory layout
5. Integrate with matrix exponential

**Deliverable:** `PruningAlgorithm` class in Rust, callable from Python

**Expected result:** 3-4x speedup (20-25 min → 6-8 min)

---

### Phase 3: Optimization Loop (Week 3)
**Goal:** Move entire site fitting to Rust

**Tasks:**
1. Implement L-BFGS optimizer in Rust
2. Port FEL objective function
3. Handle constrained optimization
4. Add LRT computation
5. Return results to Python

**Deliverable:** `fit_site_rust()` function that does entire site analysis

**Expected result:** 5-10x total speedup (52 min → 5-10 min, matches HyPhy)

---

### Phase 4: Polish & Testing (Week 4)
**Goal:** Production-ready implementation

**Tasks:**
1. Comprehensive testing (compare to Python/HyPhy)
2. Error handling and edge cases
3. Documentation
4. Performance profiling
5. Optional: parallel processing in Rust (rayon)

**Deliverable:** Production-ready Rust core

---

## Technology Stack

### Rust Crates
- **PyO3** - Python bindings
- **nalgebra** - Linear algebra (matrix operations)
- **argmin** - Optimization algorithms
- **rayon** - Data parallelism (optional)
- **ndarray** - N-dimensional arrays (alternative to nalgebra)

### Build System
- **maturin** - Build and publish Rust Python packages
- Handles compilation, linking, wheel generation

### Development
```bash
# Create Rust project
maturin new persiste-rust --bindings pyo3

# Build and install
cd persiste-rust
maturin develop

# Use in Python
import persiste_rust
result = persiste_rust.fit_site(...)
```

---

## Integration Strategy

### Gradual Migration
1. **Phase 1:** Python calls Rust for matrix exponential only
2. **Phase 2:** Python calls Rust for pruning (uses Rust matrix exponential)
3. **Phase 3:** Python calls Rust for entire site fitting
4. **Final:** Python is just orchestration + I/O

### Fallback Support
Keep Python implementation as fallback:
```python
try:
    import persiste_rust
    USE_RUST = True
except ImportError:
    USE_RUST = False

if USE_RUST:
    result = persiste_rust.fit_site(...)
else:
    result = python_fit_site(...)  # JAX version
```

---

## Performance Targets

| Implementation | Time per Site | 300 Sites | vs HyPhy |
|---------------|---------------|-----------|----------|
| Python baseline | 17s | 86 min | 10x slower |
| Python + JAX | 10s | 52 min | 6x slower |
| **Rust Phase 1** | **6s** | **30 min** | **3x slower** |
| **Rust Phase 2** | **3s** | **15 min** | **1.5x slower** |
| **Rust Phase 3** | **1-2s** | **5-10 min** | **matches HyPhy** |

---

## Risks & Mitigation

### Risk 1: Rust Learning Curve
**Mitigation:** 
- Start with simple matrix operations
- Use well-documented crates (nalgebra, PyO3)
- Incremental development

### Risk 2: PyO3 Overhead
**Mitigation:**
- Minimize Python/Rust boundary crossings
- Pass entire site data at once
- Return results in bulk

### Risk 3: Numerical Differences
**Mitigation:**
- Extensive testing against Python implementation
- Compare to HyPhy results
- Use same algorithms (just faster)

---

## Alternative: C++ with pybind11

If Rust is too unfamiliar, could use C++ instead:

**Pros:**
- More similar to HyPhy codebase
- Mature ecosystem (Eigen, Armadillo for linear algebra)
- pybind11 similar to PyO3

**Cons:**
- Manual memory management
- No safety guarantees
- Harder to maintain

**Recommendation:** Stick with Rust for safety and modern tooling

---

## Next Steps

1. **Decide:** Commit to Rust implementation?
2. **Setup:** Create `persiste-rust` project with maturin
3. **Prototype:** Implement matrix exponential (Phase 1)
4. **Benchmark:** Verify 2-3x speedup
5. **Iterate:** Continue through phases

**Timeline:** 3-4 weeks to full implementation

**Outcome:** Match HyPhy's 5-10 minute performance for 300 sites

---

## Development Workflow

### For Testing During Development
```python
# Quick test on 10 sites
fel = FELAnalysis(obs_model, baseline, use_rust=True)
results = fel.run()  # ~10-20 seconds with Rust

# vs current JAX
fel = FELAnalysis(obs_model, baseline, use_jax=True)
results = fel.run()  # ~100 seconds (10s per site)
```

**Rust makes iterative development feasible** - test changes in seconds, not minutes.
