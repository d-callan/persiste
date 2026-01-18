# Assembly Plugin: Technical Documentation

**Version:** 1.0  
**Date:** December 28, 2024

---

## System Architecture

### Design Philosophy

**Safe-by-default:** Users can't accidentally overclaim.

**Comparative model selection:** Constraints are only meaningful relative to a generative null.

**Transparent uncertainty:** All diagnostics are automatic and visible.

---

## Implementation Details

### 1. Joint Inference

**Mathematical formulation:**

```
θ*, φ* = argmax_θ,φ [ log P(D | baseline(φ), constraints(θ)) - λ||θ|| ]
```

Where:
- `D` = observed data (frequency counts)
- `θ` = constraint parameters
- `φ` = baseline parameters (e.g., join_exponent)
- `λ` = regularization strength

**Why it works:**

1. **Baseline absorbs baseline errors** - φ adjusts to fit baseline-specific patterns
2. **Constraints explain residuals** - θ only captures constraint-specific deviations
3. **L1/L2 regularizes constraints** - prevents spurious constraints from noisy data

**Implementation:**

The `fit_assembly_constraints` entry point in `cli.py` manages this joint optimization. It uses `CachedAssemblyObservationModel` to efficiently handle the stochastic simulations required for likelihood evaluation via importance sampling.

```python
# Conceptual loop managed by fit_assembly_constraints
while not converged:
    # 1. Propose theta and phi
    # 2. Get latent states (CachedModel handles IS reweighting)
    latent_states = cached_model.get_latent_states(constraint)
    
    # 3. Compute flux likelihood
    ll = compute_observation_ll(latent_states, observed, primitives, ...)
    
    # 4. Step optimizer (Hill climbing with safety thresholds)
```

### 2. Automatic Null Testing

**Workflow:**

1. Fit null model: `θ = 0, φ* = argmax P(D | baseline(φ))`
2. Fit constrained: `θ*, φ* = argmax P(D | baseline(φ), constraints(θ))`
3. Compute: `Δ LL = log P(D | θ*, φ*) - log P(D | 0, φ*_null)`
4. Classify evidence based on Δ LL

**Thresholds:**

| Δ LL | Evidence | Justification |
|------|----------|---------------|
| < 2  | None     | Within stochastic noise |
| 2-5  | Weak     | Suggestive but not conclusive |
| 5-10 | Moderate | Likely real, validate |
| > 10 | Strong   | Well-supported (conservative) |

**Conservative choice:** Δ LL > 10 accounts for:
- Finite sample size (n=80-100)
- Stochastic simulation noise
- Mild baseline misspecification
- Multiple hypothesis testing

### 3. Profile Likelihood Diagnostics

**Purpose:** Test identifiability of each parameter.

**Method:**

1. Fix parameter at grid values: `θ_i ∈ [-4, 4]`
2. Optimize other parameters: `θ_{-i}*, φ* = argmax P(D | θ_i, θ_{-i}, φ)`
3. Record likelihood: `LL(θ_i)`
4. Compute range: `max(LL) - min(LL)`

**Interpretation:**

- **Range < 2:** Flat profile → not identifiable
- **Range 2-10:** Broad profile → weakly identifiable
- **Range > 10:** Sharp profile → identifiable

**Curvature:**

```python
curvature = |LL(θ-1) - 2*LL(θ) + LL(θ+1)|
```

Higher curvature = sharper peak = better identifiability.

### 4. Regularization

**L2 penalty:**

```
Penalty = λ Σ_i θ_i²
```

**Effect:**

- Shrinks small parameters toward zero
- Doesn't eliminate large parameters
- Equivalent to Gaussian prior: `θ_i ~ N(0, 1/√(2λ))`

**Recommended values:**

- `λ = 0.1` - Standard (recommended)
- `λ = 0.0` - No regularization (risky)
- `λ = 0.2` - Strong regularization (conservative)

**Bayesian interpretation:**

```
λ = 0.1 ⟺ θ ~ N(0, σ=2.24)
```

This is a **weak prior** - doesn't constrain strong constraints.

### 5. Baseline Sensitivity

**Purpose:** Test robustness to baseline specification.

**Method:**

1. Define baseline variations (e.g., join_exponent ∈ [-0.4, -0.5, -0.6, -0.7])
2. Fit constraints for each baseline
3. Compute standard deviation of estimates

**Interpretation:**

- **σ < 0.2:** Stable (robust)
- **σ 0.2-0.5:** Moderate sensitivity
- **σ > 0.5:** Unstable (baseline matters!)

**When to use:**

- Always for publication claims
- When baseline is uncertain
- When Δ LL is borderline (5-15)

### 6. Cross-Validation

**Purpose:** Test generalization to held-out data.

**Method:** K-fold CV

1. Split data into K folds
2. For each fold:
   - Train on K-1 folds
   - Test on held-out fold
3. Average test log-likelihood

**Interpretation:**

- High CV score → good generalization
- Low CV score → overfitting

**When to use:**

- When sample size is large (n > 150)
- When testing complex models (many features)
- When baseline is uncertain

---

## Computational Complexity

### Time Complexity

**Per likelihood evaluation:**

```
O(n_samples × t_max × n_states)
```

Where:
- `n_samples` = number of Gillespie simulations (default: 100)
- `t_max` = simulation time (default: 50)
- `n_states` = size of state space (exponential in depth)

**Full optimization:**

```
O(n_iter × n_params × n_samples × t_max × n_states)
```

Where:
- `n_iter` = optimizer iterations (typically 50-200)
- `n_params` = number of parameters (2-4)

**Typical runtime:**

- Basic fit: 5-10 seconds
- With profile diagnostics: 30-60 seconds
- With baseline sensitivity (4 baselines): 2-4 minutes
- With cross-validation (5 folds): 1-2 minutes

### Space Complexity

**State space:**

```
n_states ≈ Σ_{d=0}^{max_depth} C(n_primitives + d - 1, d)
```

For n=5, depth=5: ~1000 states (manageable).

**Memory usage:**

- State cache: O(n_states)
- Trajectory storage: O(t_max)
- Latent samples: O(n_samples × avg_compounds)

**Typical:** <100 MB for standard problems.

---

## Numerical Stability

### Challenges

1. **Exponential rates:** `exp(θ·features)` can overflow
2. **Log-likelihood:** Can be very negative
3. **Optimization:** Non-convex landscape

### Solutions

1. **Rate clamping:**
   ```python
   rate = baseline_rate * exp(clip(θ·features, -10, 10))
   ```

2. **Log-space computation:**
   ```python
   log_lik = Σ log P(obs | latent)
   ```

3. **Robust optimizer:**
   ```python
   minimize(..., method='Nelder-Mead', options={'xatol': 0.01})
   ```

4. **Regularization:**
   - Prevents extreme parameter values
   - Smooths objective function

---

## Statistical Properties

### Consistency

**Theorem (informal):** As n → ∞, θ̂ → θ_true (under correct model).

**Proof sketch:**
1. MLE is consistent for exponential families
2. Gillespie simulation is unbiased
3. Regularization vanishes as n → ∞

**Caveat:** Assumes baseline is correct or jointly inferred.

### Asymptotic Normality

**Theorem (informal):** √n (θ̂ - θ_true) → N(0, I⁻¹)

Where I = Fisher information matrix.

**Practical implication:** Can construct confidence intervals using:
- Profile likelihood (exact)
- Bootstrap (robust)
- Asymptotic approximation (fast but approximate)

### Bias-Variance Tradeoff

**Regularization introduces bias:**

```
E[θ̂_regularized] ≠ θ_true (biased toward 0)
```

**But reduces variance:**

```
Var[θ̂_regularized] < Var[θ̂_MLE]
```

**Optimal λ:** Minimizes MSE = Bias² + Variance

For typical problems: λ = 0.1 is near-optimal.

---

## Comparison to Alternatives

### vs Fixed Baseline

**Fixed baseline:**
- Pros: Faster, simpler
- Cons: False positives if baseline wrong

**Joint inference:**
- Pros: Robust, fewer false positives
- Cons: Slower, more complex

**Recommendation:** Use joint inference unless baseline is well-validated.

### vs No Regularization

**No regularization (λ=0):**
- Pros: Unbiased
- Cons: High variance, false positives

**With regularization (λ=0.1):**
- Pros: Lower variance, fewer false positives
- Cons: Small bias

**Recommendation:** Use regularization by default.

### vs Liberal Thresholds

**Liberal (Δ LL > 2):**
- Pros: More discoveries
- Cons: ~20% false positive rate

**Conservative (Δ LL > 10):**
- Pros: <5% false positive rate
- Cons: Fewer discoveries

**Recommendation:** Use Δ LL > 10 for publication claims.

---

## Validation Results

### Scaling Curves

**Minimal data requirements:**
- Primitives: ≥3
- Max depth: ≥3
- Samples: ≥20
- Simulation time: ≥10

**Recommended configuration:**
- Primitives: 5
- Max depth: 5
- Samples: 80-100
- Simulation time: 50

**See:** `docs/ASSEMBLY_SCALING_CURVES.md`

### Robustness Tests

**Robust to:**
- ✓ Missing low-frequency states (<5%)
- ✓ Poisson measurement noise (~5% error)

**Sensitive to:**
- ⚠️ Baseline misspecification (20% error → false positives)

**Mitigation:**
- Use joint inference (phi adjustment in CLI)
- Use conservative thresholds (Δ LL > 10)
- Validate with baseline sensitivity analysis

**See:** `validation/experiments/assembly_validation.py`

---

## Future Improvements

### Phase 2 Enhancements

1. **Parallel simulation** - 10x speedup
2. **Adaptive sampling** - Focus on high-uncertainty regions
3. **Hierarchical models** - Multiple datasets
4. **Bayesian inference** - Full posterior, not just MAP
5. **Model averaging** - Robust to baseline uncertainty

### Real Data Applications

1. **Chemistry datasets** - Validate on experimental data
2. **Metagenomics** - Molecular assembly in nature
3. **Synthetic biology** - Design principles
4. **Astrobiology** - Origins of life

---

## References

### Theoretical Foundations

1. **Gillespie algorithm:** Gillespie, D. T. (1977). "Exact stochastic simulation of coupled chemical reactions."
2. **Assembly theory:** Marshall et al. (2021). "Identifying molecules as biosignatures with assembly theory."
3. **Profile likelihood:** Pawitan, Y. (2001). "In All Likelihood: Statistical Modelling and Inference Using Likelihood."

### Statistical Methods

1. **Regularization:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning."
2. **Model selection:** Burnham, K. P., & Anderson, D. R. (2002). "Model Selection and Multimodel Inference."
3. **Cross-validation:** Stone, M. (1974). "Cross-validatory choice and assessment of statistical predictions."

### Related Work

1. **HyPhy:** Pond, S. L. K., & Muse, S. V. (2005). "HyPhy: hypothesis testing using phylogenies."
2. **Phylogenetic models:** Yang, Z. (2006). "Computational Molecular Evolution."
3. **Chemical kinetics:** Érdi, P., & Tóth, J. (1989). "Mathematical Models of Chemical Reactions."

---

## Appendix: Mathematical Derivations

### A. Likelihood Function

**Observation model:** Frequency-weighted presence (Poisson)

```
P(count_i | latent_states) = Poisson(count_i | λ_i)

λ_i = detection_prob × (frequency of compound i in latent_states)
```

**Log-likelihood:**

```
log P(counts | latent) = Σ_i [ count_i × log(λ_i) - λ_i - log(count_i!) ]
```

### B. Constraint Model

**Rate modification:**

```
rate(s → s') = baseline_rate(s → s') × exp(θ · features(s'))
```

**Features:**
- `reuse_count`: Number of reused components
- `depth_change`: Change in depth

**Interpretation:**
- `θ > 0`: Feature is favored
- `θ < 0`: Feature is disfavored
- `θ = 0`: No constraint (null model)

### C. Baseline Model

**Join rate:**

```
λ_join(s1, s2) = κ × depth^α
```

**Split rate:**

```
λ split(s → s1, s2) = κ × depth^γ
```

Where:
- `κ` = rate constant
- `α` = join exponent (typically < 0)
- `γ` = split exponent (typically > 0)

---

## Contact

For questions, issues, or contributions:
- GitHub: [PERSISTE repository]
- Email: [PERSISTE team]
- Documentation: `docs/ASSEMBLY_USER_GUIDE.md`

---

**This is research software. Use with appropriate scientific rigor.**
