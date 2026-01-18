# Assembly Plugin: Robustness Reboot Plan

**Date:** January 2026  
**Status:** Implementation plan for review  
**Predecessor:** `docs/ROBUSTNESS_WORK_SUMMARY.md` (lessons learned from V1/V2)

---

## Overview

Clean-slate implementation of robust inference for the assembly plugin, addressing the fundamental flaws in the shelved V1/V2 approaches:

| Problem | V1/V2 Approach | Reboot Approach |
|---------|----------------|-----------------|
| Slow inference (5-10 min) | Ran Gillespie per likelihood eval | Importance-sampling cache with path stats |
| Cache invalidation | Cached states at θ=0, ignored θ changes | ESS monitoring + trust region + topology guards |
| Coupled diagnostics | Diagnostics bundled with inference | Decoupled post-fit diagnostic suite |
| No fast screening | Full stochastic for all hypotheses | Deterministic steady-state screening pass |

**Design principles:**
- `ConstraintInference` stays pure—Monte Carlo machinery lives in the adapter
- Cache at path-stats level, not raw states
- Rust-first for all computationally intensive code
- Guard explicitly against topology-changing θ
- Defaults must be conservative, explainable, and never silently sacrifice correctness

---

## Phase 1: Importance-Sampling Latent-State Cache

### 1.1 Path Statistics Structure

```rust
// rust/src/assembly/path_stats.rs

/// Sufficient statistics from a single Gillespie trajectory
#[derive(Clone, Debug)]
pub struct PathStats {
    /// Counts of each transition feature observed in trajectory
    pub feature_counts: HashMap<String, u32>,
    /// Final state (for observation model)
    pub final_state: AssemblyStateId,
    /// Total path log-probability under reference θ
    pub log_prob_ref: f64,
    /// Trajectory duration (for time-slice models)
    pub duration: f64,
}

impl PathStats {
    /// Compute importance weight for new θ relative to θ_ref
    /// log w(τ) = Σ_k N_k(τ) · (θ_k - θ_k,ref)
    pub fn log_weight(&self, theta: &HashMap<String, f64>, theta_ref: &HashMap<String, f64>) -> f64 {
        let mut log_w = 0.0;
        for (feature, count) in &self.feature_counts {
            let theta_k = theta.get(feature).unwrap_or(&0.0);
            let theta_k_ref = theta_ref.get(feature).unwrap_or(&0.0);
            log_w += (*count as f64) * (theta_k - theta_k_ref);
        }
        log_w
    }
}
```

**Benefits of path-stats caching:**
- Avoids recomputing full path probabilities
- Numerically stable (log-space weights)
- Reusable across different observation models

### 1.2 Cache Manager

```rust
// rust/src/assembly/cache_manager.rs

pub struct CacheManager {
    /// Cached path statistics from reference simulation
    paths: Vec<PathStats>,
    /// Reference θ at which paths were simulated
    theta_ref: HashMap<String, f64>,
    /// Trust region radius (default: L∞ norm in θ-space)
    trust_radius: f64,
    /// Minimum effective sample size ratio (default: 0.3; advanced: clamp to [0.1, 0.8])
    ess_threshold: f64,
    /// Maximum weight variance before early warning
    max_weight_variance: f64,
}

pub enum CacheStatus {
    /// Cache valid, return reweighted distribution
    Valid { latent_states: HashMap<AssemblyStateId, f64>, ess: f64 },
    /// Cache invalid, must resimulate
    Invalid { reason: InvalidationReason },
}

pub enum InvalidationReason {
    EssBelowThreshold { ess: f64, threshold: f64 },
    OutsideTrustRegion { distance: f64, radius: f64 },
    WeightVarianceExplosion { variance: f64 },
    TopologyChange { affected_features: Vec<String> },
}

impl CacheManager {
    /// Evaluate cache validity and return reweighted states or invalidation reason
    pub fn evaluate(&self, theta: &HashMap<String, f64>) -> CacheStatus {
        // 1. Check trust region (L∞ distance)
        let distance = self.linf_distance(theta);
        if distance > self.trust_radius {
            return CacheStatus::Invalid {
                reason: InvalidationReason::OutsideTrustRegion {
                    distance,
                    radius: self.trust_radius,
                },
            };
        }

        // 2. Compute weights
        let log_weights: Vec<f64> = self.paths
            .iter()
            .map(|p| p.log_weight(theta, &self.theta_ref))
            .collect();

        // 3. Check weight variance (early warning)
        let weight_var = self.weight_variance(&log_weights);
        if weight_var > self.max_weight_variance {
            return CacheStatus::Invalid {
                reason: InvalidationReason::WeightVarianceExplosion { variance: weight_var },
            };
        }

        // 4. Normalize weights and compute ESS
        let (weights, ess) = self.normalize_and_ess(&log_weights);
        if ess < self.ess_threshold * (self.paths.len() as f64) {
            return CacheStatus::Invalid {
                reason: InvalidationReason::EssBelowThreshold {
                    ess,
                    threshold: self.ess_threshold * (self.paths.len() as f64),
                },
            };
        }

        // 5. Aggregate to state distribution
        let latent_states = self.aggregate_to_states(&weights);

        CacheStatus::Valid { latent_states, ess }
    }

    fn linf_distance(&self, theta: &HashMap<String, f64>) -> f64 {
        // ... L∞ norm implementation
    }

    fn weight_variance(&self, log_weights: &[f64]) -> f64 {
        // ... variance of exp(log_weights)
    }

    fn normalize_and_ess(&self, log_weights: &[f64]) -> (Vec<f64>, f64) {
        // ... log-sum-exp normalization + ESS = 1 / Σ w_i²
    }

    fn aggregate_to_states(&self, weights: &[f64]) -> HashMap<AssemblyStateId, f64> {
        // ... weighted aggregation by final_state
    }
}
```

**Three orthogonal safety valves:**
1. ESS < α·N (e.g., α = 0.3)
2. θ proposal exits trust region
3. Weight variance explosion (cheap early warning)

### 1.2.1 Recommended Defaults (and Advanced Knobs)

- **Trust region**
  - Default: L∞ trust region with `--trust-radius 1.0` (each θ component may move ±1.0 before resimulation)
  - Experimental: Mahalanobis trust region for late-stage refinement only
- **ESS threshold**
  - Default: `--ess-threshold 0.3` (interpreted as 30 % of cached trajectories)
  - Advanced: expose as a ratio (clamped to `[0.1, 0.8]`), strongly documented

### 1.3 Topology-Change Guards

```rust
// rust/src/assembly/topology_guard.rs

/// Features that can change graph topology when their θ changes significantly
pub struct TopologyGuard {
    /// Features that affect edge pruning
    topology_sensitive_features: HashSet<String>,
    /// Soft floor rate to prevent edges from vanishing
    soft_floor_rate: f64,
}

impl TopologyGuard {
    /// Check if θ change would invalidate cached topology
    pub fn check(&self, theta: &HashMap<String, f64>, theta_ref: &HashMap<String, f64>) -> Option<Vec<String>> {
        let mut affected = Vec::new();
        for feature in &self.topology_sensitive_features {
            let delta = (theta.get(feature).unwrap_or(&0.0) - theta_ref.get(feature).unwrap_or(&0.0)).abs();
            // Threshold where rate scaling exp(Δθ) could cross pruning boundary
            if delta > 2.0 {
                affected.push(feature.clone());
            }
        }
        if affected.is_empty() { None } else { Some(affected) }
    }
}
```

**Mitigations for topology changes:**
1. Forbid topology-changing constraints during IS (document explicitly)
2. Include soft floor rate so edges never truly vanish
3. Treat topology change as hard resimulation trigger

### 1.4 Python Adapter Integration

```python
# src/persiste/plugins/assembly/observation/cached_observation.py

class CachedAssemblyObservationModel(ObservationModel):
    """
    Assembly observation model with importance-sampling cache.
    
    Design: Adapter owns cache validity; ConstraintInference just asks for log_likelihood(θ).
    This keeps ConstraintInference pure and unaware of Monte Carlo machinery.
    """

    def __init__(
        self,
        *,
        graph: AssemblyGraph,
        baseline: AssemblyBaseline,
        obs_model: ObservationModel,
        initial_state: AssemblyState,
        simulation: SimulationSettings,
        cache_config: CacheConfig,
        rng_seed: int | None = None,
    ):
        self.graph = graph
        self.baseline = baseline
        self.obs_model = obs_model
        self.initial_state = initial_state
        self.simulation = simulation
        self.cache_config = cache_config
        
        # Rust cache manager (None until first simulation)
        self._cache: CacheManager | None = None
        self._rng_seed = rng_seed

    def log_likelihood(
        self,
        data: ObservedTransitions,
        constraint: AssemblyConstraint,
        graph: Any,
    ) -> float:
        """
        Compute log-likelihood, using cached paths when valid.
        
        The adapter transparently handles cache management—ConstraintInference
        never knows whether we resimulated or reweighted.
        """
        theta = constraint.feature_weights

        if self._cache is None:
            # First call: simulate and initialize cache
            self._initialize_cache(theta)
        
        # Check cache validity (Rust call)
        status = self._cache.evaluate(theta)

        if isinstance(status, CacheValid):
            latent_states = status.latent_states
            # Optionally log ESS for diagnostics
        else:
            # Cache invalid: resimulate at new θ_ref
            self._resimulate(theta, reason=status.reason)
            latent_states = self._cache.get_current_states()

        # Attach latent states to data for observation model
        data.latent_states = latent_states
        return self.obs_model.log_likelihood(data, self.baseline, self.graph)

    def _initialize_cache(self, theta_ref: dict[str, float]) -> None:
        """Run Gillespie simulation and populate cache with path stats."""
        # Rust call: parallel trajectory simulation
        paths = simulate_trajectories_rust(
            graph=self.graph,
            baseline=self.baseline,
            theta=theta_ref,
            initial_state=self.initial_state,
            n_samples=self.simulation.n_samples,
            t_max=self.simulation.t_max,
            burn_in=self.simulation.burn_in,
            seed=self._rng_seed,
        )
        self._cache = CacheManager(
            paths=paths,
            theta_ref=theta_ref,
            trust_radius=self.cache_config.trust_radius,
            ess_threshold=self.cache_config.ess_threshold,
            max_weight_variance=self.cache_config.max_weight_variance,
        )

    def _resimulate(self, theta_ref: dict[str, float], reason: InvalidationReason) -> None:
        """Resimulate at new reference point, logging invalidation reason."""
        logger.info(f"Cache invalidated: {reason}. Resimulating at θ_ref={theta_ref}")
        self._initialize_cache(theta_ref)
```

---

## Phase 2: Deterministic Screening Pass

### 2.1 Deterministic Screening Model (Steady-State First)

```python
# src/persiste/plugins/assembly/screening/mean_field.py

class SteadyStateAssemblyModel:
    """
    Deterministic approximation of assembly dynamics.
    
    Default: steady-state / algebraic approximation suitable for ranking.

    Scope (be explicit):
    - Approximates expected occupancy of state classes
    - Ignores path correlations
    - NOT used for parameter estimation
    - Screening is monotonic but biased

    Optional (experimental): coarse-grained ODE screening if steady-state
    fails to separate hypotheses cleanly.
    """

    def __init__(self, graph: AssemblyGraph, baseline: AssemblyBaseline):
        self.graph = graph
        self.baseline = baseline

    def expected_occupancy(
        self,
        theta: dict[str, float],
        initial_state: AssemblyState,
    ) -> dict[AssemblyState, float]:
        """
        Compute steady-state / algebraic approximation of occupancy.
        
        Returns approximate P(state) under the deterministic approximation.
        """
        # ... steady-state approximation (Python initially; Rust if needed)

    def approximate_log_likelihood(
        self,
        theta: dict[str, float],
        observed_compounds: set[str],
        obs_model: PresenceObservationModel,
    ) -> float:
        """
        Cheap approximate log-likelihood for screening.
        """
        occupancy = self.expected_occupancy(theta, ...)
        return obs_model.compute_log_likelihood(observed_compounds, occupancy)
```

### 2.2 Screening Criterion

```python
# src/persiste/plugins/assembly/screening/screening.py

@dataclass
class ScreeningResult:
    theta: dict[str, float]
    delta_ll: float           # ΔLL relative to θ=0
    normalized_delta_ll: float  # ΔLL / approximate_stderr
    passed: bool

def screen_hypotheses(
    hypotheses: list[dict[str, float]],
    mean_field: SteadyStateAssemblyModel,
    observed_compounds: set[str],
    obs_model: PresenceObservationModel,
    threshold: float = 2.0,
) -> list[ScreeningResult]:
    """
    Screen candidate θ values using deterministic approximation.
    
    Criterion: normalized ΔLL to avoid θ with large but noisy effects
    crowding out stable ones.
    """
    # Compute baseline (θ=0) likelihood
    ll_null = mean_field.approximate_log_likelihood({}, observed_compounds, obs_model)
    
    results = []
    for theta in hypotheses:
        ll = mean_field.approximate_log_likelihood(theta, observed_compounds, obs_model)
        delta_ll = ll - ll_null
        
        # Fisher-ish scaling (cheap variance approximation)
        approx_var = estimate_screening_variance(theta, mean_field)
        normalized = delta_ll / (approx_var ** 0.5 + 1e-6)
        
        results.append(ScreeningResult(
            theta=theta,
            delta_ll=delta_ll,
            normalized_delta_ll=normalized,
            passed=normalized > threshold,
        ))
    
    return results
```

### 2.3 CLI Modes

```python
# src/persiste/plugins/assembly/cli.py

class InferenceMode(Enum):
    SCREEN_ONLY = "screen-only"           # Fast, deterministic only
    SCREEN_AND_REFINE = "screen-and-refine"  # Screen then stochastic
    FULL_STOCHASTIC = "full-stochastic"   # No shortcuts, explicit slow

def fit_assembly_constraints(
    observed_compounds: set[str],
    primitives: list[str],
    mode: InferenceMode = InferenceMode.FULL_STOCHASTIC,
    screening_threshold: float = 2.0,
    **kwargs,
) -> ConstraintResult:
    """
    Fit assembly constraints with configurable inference mode.
    
    Modes:
    - screen-only: Fast deterministic screening, no stochastic refinement
    - screen-and-refine: Screen candidates, refine winners stochastically
    - full-stochastic: No shortcuts, run full Gillespie-based inference
    """
    # ...
```

### 2.4 Screening Grid Strategy

Default: lightweight adaptive grid search (predictable + explainable).

- **Auto (default):**
  - Start with a coarse symmetric grid around θ=0
  - Keep top `K` candidates by normalized ΔLL
  - Refine locally around those candidates
  - Stop after a fixed budget
- **Manual (advanced):** user provides explicit grid/ranges

API sketch (default values shown):

```bash
--screen-grid auto          # default (coarse+refine loop)
--screen-grid manual        # user provides ranges/grid
--screen-budget 100         # default total deterministic evaluations
--screen-topk 10            # default number of local refinements
--screen-refine-radius 0.5  # size of local neighborhood around winners
```

---

## Phase 3: Decoupled Diagnostics Suite

### 3.1 Diagnostic Inputs

```python
# src/persiste/plugins/assembly/diagnostics/inputs.py

@dataclass
class InferenceArtifacts:
    """Artifacts from inference, input to diagnostics."""
    theta_hat: dict[str, float]
    log_likelihood: float
    cache_id: str  # Reference to cached path stats
    baseline_config: AssemblyBaselineConfig
    graph_config: AssemblyGraphConfig

@dataclass  
class CachedPathData:
    """Cached trajectories with sufficient stats for diagnostics."""
    paths: list[PathStats]
    theta_ref: dict[str, float]
    
    def reweight_to(self, theta: dict[str, float]) -> dict[AssemblyState, float]:
        """Reweight cached paths to new θ without resimulation."""
        # ...
```

### 3.2 Diagnostic Functions

```python
# src/persiste/plugins/assembly/diagnostics/suite.py

def null_resampling(
    artifacts: InferenceArtifacts,
    cache: CachedPathData,
    n_resamples: int = 1000,
) -> NullResamplingResult:
    """
    Generate null ΔLL distribution using cached paths.
    
    No resimulation needed—uses importance reweighting.
    """
    # ...

def profile_likelihood(
    artifacts: InferenceArtifacts,
    cache: CachedPathData,
    feature: str,
    grid: np.ndarray,
) -> ProfileLikelihoodResult:
    """
    Compute profile likelihood for a single feature.
    
    Uses importance reweighting where valid, resimulates when ESS drops.
    """
    # ...

def baseline_sensitivity(
    artifacts: InferenceArtifacts,
    cache: CachedPathData,
    perturbations: list[AssemblyBaselineConfig],
) -> BaselineSensitivityResult:
    """
    Test sensitivity of θ̂ to baseline specification.
    
    Minimal reruns via importance sampling.
    """
    # ...
```

### 3.3 Artifact Separation

```python
# src/persiste/plugins/assembly/diagnostics/artifacts.py

@dataclass
class DiagnosticArtifacts:
    """Output from diagnostic suite, separate from inference artifacts."""
    null_distribution: NullResamplingResult | None
    profile_likelihoods: dict[str, ProfileLikelihoodResult]
    baseline_sensitivity: BaselineSensitivityResult | None
    
    def to_json(self, path: Path) -> None:
        """Serialize for CI pipelines / Datamonkey-style UIs."""
        # ...

    def plot_summary(self, output_dir: Path) -> None:
        """Generate diagnostic plots."""
        # ...
```

---

## Phase 4: Rust Implementation

### 4.1 Parallelism Priority

1. **Independent Gillespie trajectories** (highest ROI)
2. **Importance-weight evaluation** (diminishing returns)
3. **Screening θ grid** (optional, likely sufficient in Python)

### 4.2 Rust Crate Structure

```
rust/src/
├── lib.rs                    # PyO3 module entry
├── assembly/
│   ├── mod.rs
│   ├── state.rs              # AssemblyState, AssemblyStateId
│   ├── baseline.rs           # Rate calculations
│   ├── constraint.rs         # Constraint evaluation
│   ├── graph.rs              # Lazy graph with neighbor generation
│   ├── gillespie.rs          # Parallel Gillespie simulator
│   ├── path_stats.rs         # Trajectory sufficient statistics
│   ├── cache_manager.rs      # Importance-sampling cache
│   └── topology_guard.rs     # Topology change detection
└── bindings.rs               # PyO3 bindings
```

### 4.3 Parallel Gillespie

```rust
// rust/src/assembly/gillespie.rs

use rayon::prelude::*;

pub fn simulate_trajectories_parallel(
    graph: &AssemblyGraph,
    baseline: &AssemblyBaseline,
    theta: &HashMap<String, f64>,
    initial_state: AssemblyStateId,
    n_samples: usize,
    t_max: f64,
    burn_in: f64,
    seed: u64,
) -> Vec<PathStats> {
    // Create independent RNGs for each trajectory
    let rngs: Vec<_> = (0..n_samples)
        .map(|i| StdRng::seed_from_u64(seed.wrapping_add(i as u64)))
        .collect();

    // Parallel simulation
    rngs.into_par_iter()
        .map(|mut rng| {
            simulate_single_trajectory(graph, baseline, theta, initial_state, t_max, burn_in, &mut rng)
        })
        .collect()
}
```

### 4.4 Structured Logging

```rust
// rust/src/assembly/timing.rs

#[derive(Debug, Serialize)]
pub struct TimingLog {
    pub simulation_ms: u64,
    pub reweighting_ms: u64,
    pub screening_ms: u64,
    pub diagnostics_ms: u64,
    pub cache_hits: u32,
    pub cache_misses: u32,
    pub resimulation_reasons: Vec<String>,
}
```

---

## Phase 5: Validation

### 5.1 Success Criteria

**Correctness:**
Does θ̂ converge to the true θ under:
- θ_ref ≠ 0
- Moderate ESS degradation (ESS ≈ 0.5·N)
- Screening enabled

**Stability:**
Does inference NOT snap back to θ=0 when cache reuse is heavy?

### 5.2 Test Cases

```python
# tests/plugins/assembly/test_robustness_reboot.py

class TestImportanceSamplingCache:
    def test_weight_computation_matches_exact(self):
        """Verify log w(τ) = Σ N_k(τ) · (θ_k - θ_k,ref)."""
        
    def test_ess_triggers_resimulation(self):
        """Cache invalidates when ESS drops below threshold."""
        
    def test_trust_region_triggers_resimulation(self):
        """Cache invalidates when θ exits trust region."""
        
    def test_no_snapback_to_zero(self):
        """θ̂ does NOT collapse to zero under heavy cache reuse."""

class TestDeterministicScreening:
    def test_screening_monotonic(self):
        """Screening preserves relative ordering of strong signals."""
        
    def test_normalized_criterion_stabilizes(self):
        """Normalized ΔLL prevents noisy features from dominating."""

class TestDecoupledDiagnostics:
    def test_null_resampling_without_resimulation(self):
        """Null distribution generated via reweighting only."""
        
    def test_profile_likelihood_uses_cache(self):
        """Profile likelihood reuses cache where ESS permits."""

class TestParameterRecovery:
    @pytest.mark.parametrize("theta_true", [
        {"reuse_count": 1.5},
        {"reuse_count": 2.0, "depth_change": -0.5},
    ])
    def test_recovery_with_importance_sampling(self, theta_true):
        """θ̂ recovers θ_true with IS-enabled inference."""
```

### 5.3 Benchmark Suite

```python
# benchmarks/assembly_robustness.py

def benchmark_inference_modes():
    """Compare runtime and accuracy across inference modes."""
    results = {}
    for mode in [InferenceMode.SCREEN_ONLY, InferenceMode.SCREEN_AND_REFINE, InferenceMode.FULL_STOCHASTIC]:
        start = time.perf_counter()
        result = fit_assembly_constraints(..., mode=mode)
        elapsed = time.perf_counter() - start
        results[mode] = {
            "runtime_s": elapsed,
            "theta_hat": result.parameters,
            "log_likelihood": result.log_likelihood,
        }
    return results
```

---

## Implementation Order

| Phase | Component | Effort | Dependencies |
|-------|-----------|--------|--------------|
| 1a | Rust: PathStats, Gillespie parallel | 2-3 days | None |
| 1b | Rust: CacheManager, TopologyGuard | 2 days | 1a |
| 1c | Python: CachedAssemblyObservationModel | 1 day | 1b |
| 2a | Python: SteadyStateAssemblyModel | 2 days | None |
| 2b | Python: Screening criterion + CLI | 1 day | 2a |
| 3a | Python: Diagnostic inputs/artifacts | 1 day | 1c |
| 3b | Python: Diagnostic functions | 2 days | 3a |
| 4 | Validation suite + benchmarks | 2 days | All above |

**Total estimate:** ~13-15 days

---

## Defaults & Advanced Options (Summary)

| Topic | Default | Advanced / Experimental |
|------|---------|--------------------------|
| Trust region | L∞ with `--trust-radius 1.0`, ±1.0 per coordinate | `--trust-region=mahalanobis` (late-stage only) |
| ESS threshold | `--ess-threshold 0.3` (ratio of N_paths) | Advanced tuning (clamped `[0.1, 0.8]`) |
| Screening model | Steady-state / algebraic approximation | Coarse ODE screening (opt-in) |
| Screening grid | Hybrid adaptive (`--screen-grid auto`, `--screen-budget 100`, `--screen-topk 10`) | Manual grid (`--screen-grid manual`, user budget) |

### Remaining Open Question

1. **Topology-changing θ features:** define and document which constraint features can change pruning/topology, and enforce hard resimulation triggers for those.

---

## References

- Original robustness work: `docs/ROBUSTNESS_WORK_SUMMARY.md`
- Rust implementation guide: `docs/development/RUST_IMPLEMENTATION_GUIDE.md`
- Assembly plugin summary: `src/persiste/plugins/assembly/docs/ASSEMBLY_PLUGIN_SUMMARY.md`
