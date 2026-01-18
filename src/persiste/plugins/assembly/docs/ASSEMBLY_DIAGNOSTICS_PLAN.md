# Assembly Plugin: Two-Tier Diagnostics Implementation Plan

## Executive Summary

This document specifies the implementation plan for a two-tier diagnostics architecture for the Assembly plugin:

- **Tier 1: Automatic Safety Checks** - Fast, mandatory diagnostics baked into the plugin that answer "Is it safe to interpret θ at all?"
- **Tier 2: Optional Deep Diagnostics** - Opt-in "recipes" for comprehensive robustness analysis, following the genecontent/copynumber pattern

## Motivation

Current robustness tests reveal that:
1. Users cannot reliably detect baseline misspecification themselves
2. False positives under wrong baselines are large and convincing (ΔLL ≈ 48)
3. Once θ inference runs, users will trust the result unless warned

Therefore, some diagnostics must be automatic to prevent PERSISTE from producing confidently wrong answers in realistic scenarios.

## Pattern Review: Recipes in genecontent/copynumber

### Key Observations

From `genecontent/strain_recipes.py` and `copynumber/recipes/`:

1. **Structure**: Separate `recipes/` directory with numbered recipe files
2. **Naming**: `recipe_N_descriptive_name.py` (e.g., `recipe_1_dosage_stability.py`)
3. **Result Classes**: Each recipe returns a dataclass with:
   - Raw results (parameters, likelihoods)
   - Interpretation (human-readable text)
   - Recommendation (what to do next)
   - `.print_summary()` method for formatted output
4. **Public API**: Recipes exposed via `recipes/__init__.py` with clear `__all__`
5. **Documentation**: Comprehensive `recipes/README.md` with:
   - Question each recipe answers
   - Purpose (diagnostic vs descriptive)
   - Usage examples
   - Interpretation guidelines
   - Recommended workflow
6. **Diagnostics Module**: Separate `diagnostics.py` for quick checks (e.g., `expected_vs_observed_cn`)

### Pattern Applied to Assembly

Assembly diagnostics should follow this proven structure:
- `src/persiste/plugins/assembly/recipes/` for Tier 2
- `src/persiste/plugins/assembly/safety/` for Tier 1 (new)
- Integration into `ConstraintResult` for automatic checks

---

## Tier 1: Automatic Safety Checks

### Design Principles

1. **Fast**: Must complete in <5% of inference time
2. **Conservative**: Prefer false alarms over silent failures
3. **Non-blocking**: Never silently block inference; always return θ̂ with warnings
4. **Actionable**: Warnings must tell users what to do next

### Implementation Location

**New module**: `src/persiste/plugins/assembly/safety/`

```
safety/
├── __init__.py
├── baseline_check.py      # Baseline sanity diagnostics
├── identifiability.py     # Identifiability screening
├── cache_reliability.py   # Cache/IS stability checks
└── safety_report.py       # SafetyReport dataclass
```

### Safety Check 1: Baseline Sanity

**File**: `safety/baseline_check.py`

**Purpose**: Detect whether baseline dynamics poorly explain observations

**Implementation**:

```python
@dataclass
class BaselineSanityResult:
    baseline_ok: bool
    warning_level: str  # 'none', 'mild', 'severe'
    observed_summary: dict  # Low-dimensional, robust summaries
    expected_summary: dict  # from baseline simulation
    divergence_score: float  # Simple heuristic, not KL
    message: str
    delta_ll_multiplier: float  # Threshold adjustment multiplier

def check_baseline_sanity(
    observed_compounds: set[str],
    primitives: list[str],
    baseline: AssemblyBaseline,
    initial_state: AssemblyState,
    n_samples: int = 100,
    max_depth: int = 5,
) -> BaselineSanityResult:
    """
    Quick baseline sanity check.
    
    Strategy:
    1. Simulate trajectories under baseline (θ=0)
    2. Compare observed vs expected summaries:
       - Mean assembly depth
       - Compound diversity (Shannon entropy)
       - Reuse frequency (if detectable)
    3. Flag if divergence is large
    
    Cost: ~100 simulations at θ=0 (already cached if screening ran)
    """
    # Simulate under null
    null_trajectories = simulate_assembly_trajectories(
        primitives=primitives,
        initial_parts=initial_state.get_parts_list(),
        theta={},
        n_samples=n_samples,
        t_max=50.0,
        burn_in=25.0,
        max_depth=max_depth,
        seed=42,
    )
    
    # Compute expected summaries (low-dimensional, robust)
    # 1. Total compound count distribution
    expected_compound_count = len(set(c for t in null_trajectories for c in t['final_state'].get_parts_list()))
    
    # 2. Depth histogram
    expected_depths = [t['final_state'].assembly_depth for t in null_trajectories]
    expected_depth_mean = np.mean(expected_depths)
    expected_depth_std = np.std(expected_depths)
    
    # 3. Transition type ratios (from feature counts if available)
    # 4. Presence frequency of top-k compounds
    expected_diversity = len(set(c for t in null_trajectories for c in t['final_state'].get_parts_list())) / len(primitives)
    
    # Compute observed summaries
    observed_compound_count = len(observed_compounds)
    observed_diversity = len(observed_compounds) / len(primitives)
    
    # Avoid rare states, exact topology - use robust aggregates only
    
    # Divergence score (simple heuristic)
    depth_diff = abs(observed_depth - expected_depth) / max(expected_depth, 1)
    diversity_diff = abs(observed_diversity - expected_diversity)
    divergence_score = max(depth_diff, diversity_diff)
    
    # Classify with multiplier-based threshold adjustment
    delta_ll_multipliers = {
        "none": 1.0,
        "mild": 2.0,
        "severe": 3.0,
    }
    
    if divergence_score > 2.0:
        warning_level = 'severe'
        baseline_ok = False
        message = "Baseline dynamics poorly explain observations; constraint inference may be biased"
    elif divergence_score > 1.0:
        warning_level = 'mild'
        baseline_ok = False
        message = "Moderate baseline mismatch detected; interpret constraints cautiously"
    else:
        warning_level = 'none'
        baseline_ok = True
        message = "Baseline appears consistent with observations"
    
    return BaselineSanityResult(
        baseline_ok=baseline_ok,
        warning_level=warning_level,
        observed_summary={
            'compound_count': observed_compound_count,
            'diversity': observed_diversity,
        },
        expected_summary={
            'compound_count': expected_compound_count,
            'depth_mean': expected_depth_mean,
            'depth_std': expected_depth_std,
            'diversity': expected_diversity,
        },
        divergence_score=divergence_score,
        message=message,
        delta_ll_multiplier=delta_ll_multipliers[warning_level],
    )
```

**Threshold Adjustment Policy**:
- Use multiplier-based adjustment instead of hardcoded values
- Base ΔLL threshold (e.g., 2.0) × multiplier
- Multipliers: `none=1.0`, `mild=2.0`, `severe=3.0`
- This is more extensible and easier to explain: "We require stronger evidence when the baseline is suspect"

**Baseline Summaries** (explicit specification):
- **Total compound count distribution**: Robust to rare states
- **Depth histogram**: Mean and std of assembly depths
- **Transition type ratios**: JOIN/SPLIT/DECAY frequencies (if available)
- **Presence frequency of top-k compounds**: Avoids rare state sensitivity
- **Avoid**: Exact topology, rare states, high-dimensional features

---

### Safety Check 2: Identifiability Screen

**File**: `safety/identifiability.py`

**Purpose**: Detect flat likelihood surfaces or collapse-to-null behavior

**Implementation**:

```python
@dataclass
class IdentifiabilityResult:
    status: str  # 'ok', 'flat', 'collapse_to_null'
    identifiable: bool
    evidence: dict  # screening_variance, top_k_separation, etc.
    message: str
    recommendation: str

def check_identifiability(
    screening_results: list[ScreeningResult],
    theta_hat: dict[str, float],
) -> IdentifiabilityResult:
    """
    Identifiability screen from screening results.
    
    Strategy:
    1. If screening was run, reuse variance in normalized ΔLL
    2. Check if θ̂ is near zero (collapse-to-null)
    3. Estimate curvature from top-3 screening results
    
    Cost: Zero (reuses screening data)
    """
    if not screening_results:
        # No screening data - assume identifiable (conservative)
        return IdentifiabilityResult(
            identifiable=True,
            screening_variance=0.0,
            theta_hat_near_zero=False,
            curvature_proxy=0.0,
            message="No screening data available; identifiability unknown",
        )
    
    # Variance in normalized ΔLL
    delta_lls = [r.normalized_delta_ll for r in screening_results]
    screening_variance = np.var(delta_lls)
    
    # Check if θ̂ near zero
    theta_norm = np.linalg.norm(list(theta_hat.values()))
    theta_hat_near_zero = theta_norm < 0.1
    
    # Curvature proxy: difference between top-1 and top-3
    if len(screening_results) >= 3:
        curvature_proxy = screening_results[0].normalized_delta_ll - screening_results[2].normalized_delta_ll
    else:
        curvature_proxy = 0.0
    
    # Classify with explicit failure modes
    if theta_hat_near_zero and screening_results[0].normalized_delta_ll < 2.0:
        status = 'collapse_to_null'
        identifiable = False
        message = "θ̂ collapsed to zero; no constraint signal detected"
        recommendation = "Constraints unsupported or baseline dominant; do not interpret θ"
    elif screening_variance < 1.0 and curvature_proxy < 2.0:
        status = 'flat'
        identifiable = False
        message = "Flat likelihood surface detected; constraints may not be identifiable"
        recommendation = "Needs more data or stronger constraints; increase sample size"
    else:
        status = 'ok'
        identifiable = True
        message = "Constraints appear identifiable from screening"
        recommendation = "Proceed with interpretation"
    
    return IdentifiabilityResult(
        status=status,
        identifiable=identifiable,
        evidence={
            'screening_variance': screening_variance,
            'theta_hat_near_zero': theta_hat_near_zero,
            'curvature_proxy': curvature_proxy,
            'top_k_separation': curvature_proxy,
        },
        message=message,
        recommendation=recommendation,
    )
```

---

### Safety Check 3: Cache Reliability

**File**: `safety/cache_reliability.py`

**Purpose**: Detect importance sampling degradation

**Implementation**:

```python
@dataclass
class CacheReliabilityResult:
    status: str  # 'ok', 'warning', 'severe'
    inference_stable: bool
    ess_at_theta_hat: float
    n_resimulations: int
    ess_threshold: float
    message: str

def check_cache_reliability(
    cache_stats: dict,
    ess_threshold: float = 0.3,
) -> CacheReliabilityResult:
    """
    Cache reliability check from inference.
    
    Strategy:
    1. Check ESS at θ̂
    2. Count resimulations triggered
    3. Flag if ESS too low or excessive resimulations
    
    Cost: Zero (reads cache_stats from inference)
    """
    ess_at_theta_hat = cache_stats.get('ess_at_theta_hat', 1.0)
    n_resimulations = cache_stats.get('resimulation_count', 0)
    n_paths = cache_stats.get('n_paths', 100)
    
    # Classify with soft vs hard failure distinction
    ess_ratio = ess_at_theta_hat / n_paths
    
    if ess_ratio <= 0.15:
        status = 'severe'
        inference_stable = False
        message = f"Severe ESS degradation at θ̂ ({ess_at_theta_hat:.1f}/{n_paths}, {ess_ratio:.2%}); importance sampling unreliable"
    elif ess_ratio <= ess_threshold:
        status = 'warning'
        inference_stable = False
        message = f"Low ESS at θ̂ ({ess_at_theta_hat:.1f}/{n_paths}, {ess_ratio:.2%}); importance sampling may be unreliable"
    elif n_resimulations > 5:
        status = 'warning'
        inference_stable = False
        message = f"Excessive resimulations ({n_resimulations}); inference may be unstable"
    else:
        status = 'ok'
        inference_stable = True
        message = f"Cache stable (ESS={ess_at_theta_hat:.1f}/{n_paths}, {ess_ratio:.2%}, {n_resimulations} resims)"
    
    return CacheReliabilityResult(
        status=status,
        inference_stable=inference_stable,
        ess_at_theta_hat=ess_at_theta_hat,
        n_resimulations=n_resimulations,
        ess_threshold=ess_threshold,
        message=message,
    )
```

---

### Safety Report Integration

**File**: `safety/safety_report.py`

```python
@dataclass
class SafetyReport:
    """
    Consolidated safety report from Tier 1 checks.
    
    Attributes:
        overall_status: Machine-readable severity ('ok', 'warning', 'unsafe')
        baseline_check: BaselineSanityResult
        identifiability_check: IdentifiabilityResult
        cache_check: CacheReliabilityResult
        overall_safe: bool (deprecated, use overall_status)
        warnings: list[str]
        recommendations: list[str]
        adjusted_delta_ll_threshold: float
    """
    overall_status: str  # 'ok', 'warning', 'unsafe'
    baseline_check: BaselineSanityResult
    identifiability_check: IdentifiabilityResult
    cache_check: CacheReliabilityResult
    overall_safe: bool  # Kept for backward compatibility
    warnings: list[str]
    recommendations: list[str]
    adjusted_delta_ll_threshold: float
    
    def print_summary(self):
        """Print safety summary."""
        print("=" * 70)
        print("SAFETY REPORT")
        print("=" * 70)
        
        print(f"\n1. Baseline Sanity: {'✓ PASS' if self.baseline_check.baseline_ok else '⚠ WARNING'}")
        print(f"   {self.baseline_check.message}")
        
        print(f"\n2. Identifiability: {'✓ PASS' if self.identifiability_check.identifiable else '⚠ WARNING'}")
        print(f"   {self.identifiability_check.message}")
        
        print(f"\n3. Cache Reliability: {'✓ PASS' if self.cache_check.inference_stable else '⚠ WARNING'}")
        print(f"   {self.cache_check.message}")
        
        if not self.overall_safe:
            print(f"\n⚠ OVERALL: UNSAFE FOR INTERPRETATION")
            print(f"   ΔLL threshold adjusted: {self.adjusted_delta_ll_threshold:.1f}")
            print(f"\n   Warnings:")
            for w in self.warnings:
                print(f"   - {w}")
        else:
            print(f"\n✓ OVERALL: SAFE FOR INTERPRETATION")
        
        print("=" * 70)

def run_safety_checks(
    observed_compounds: set[str],
    primitives: list[str],
    baseline: AssemblyBaseline,
    initial_state: AssemblyState,
    theta_hat: dict[str, float],
    screening_results: list[ScreeningResult],
    cache_stats: dict,
) -> SafetyReport:
    """
    Run all Tier 1 safety checks.
    
    Returns consolidated SafetyReport with adjusted thresholds.
    """
    baseline_check = check_baseline_sanity(
        observed_compounds, primitives, baseline, initial_state
    )
    
    identifiability_check = check_identifiability(
        screening_results, theta_hat
    )
    
    cache_check = check_cache_reliability(cache_stats)
    
    # Aggregate with machine-readable status
    overall_safe = (
        baseline_check.baseline_ok and
        identifiability_check.identifiable and
        cache_check.inference_stable
    )
    
    # Determine overall_status
    if not baseline_check.baseline_ok and baseline_check.warning_level == 'severe':
        overall_status = 'unsafe'
    elif not identifiability_check.identifiable and identifiability_check.status == 'collapse_to_null':
        overall_status = 'unsafe'
    elif cache_check.status == 'severe':
        overall_status = 'unsafe'
    elif not overall_safe:
        overall_status = 'warning'
    else:
        overall_status = 'ok'
    
    warnings = []
    recommendations = []
    
    if not baseline_check.baseline_ok:
        warnings.append(baseline_check.message)
    if not identifiability_check.identifiable:
        warnings.append(identifiability_check.message)
        recommendations.append(identifiability_check.recommendation)
    if not cache_check.inference_stable:
        warnings.append(cache_check.message)
    
    # Adjust ΔLL threshold using multiplier
    base_threshold = 2.0
    adjusted_threshold = base_threshold * baseline_check.delta_ll_multiplier
    
    return SafetyReport(
        overall_status=overall_status,
        baseline_check=baseline_check,
        identifiability_check=identifiability_check,
        cache_check=cache_check,
        overall_safe=overall_safe,
        warnings=warnings,
        recommendations=recommendations,
        adjusted_delta_ll_threshold=adjusted_threshold,
    )
```

---

### Machine-Readable Status Benefits

The `overall_status` field enables:
- **UI badges**: Datamonkey can show green/yellow/red status
- **CLI exit codes**: Optional non-zero exit for `unsafe` status
- **Automated pipelines**: Gate downstream steps on safety status
- **Batch reporting**: Aggregate safety across multiple analyses

---

## Tier 2: Optional Deep Diagnostics (Recipes)

### Design Principles

1. **Opt-in**: Users must explicitly request these
2. **Expensive**: Can take minutes to hours
3. **Comprehensive**: Full robustness analysis
4. **Post-hoc**: Runnable after inference completes

### Standardized Recipe Output

All recipes inherit from a shared base class for consistency:

```python
from abc import ABC, abstractmethod
from typing import Literal

class DiagnosticReport(ABC):
    """
    Base class for all diagnostic recipe reports.
    
    Ensures consistent interface across recipes.
    """
    severity: Literal["ok", "warning", "fail"]
    recommendation: str
    
    @abstractmethod
    def print_summary(self):
        """Print human-readable summary."""
        pass
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}
```

Benefits:
- Keeps recipes consistent
- Easy to add Recipe 3, 4, etc.
- Enables batch execution / reporting
- Facilitates automated testing

### Implementation Location

**New module**: `src/persiste/plugins/assembly/recipes/`

```
recipes/
├── __init__.py
├── README.md                          # Comprehensive documentation
├── recipe_0_null_resampling.py        # Null distribution diagnostic
├── recipe_1_profile_likelihood.py     # Profile likelihood sweeps
├── recipe_2_baseline_perturbation.py  # Baseline sensitivity
└── recipe_3_parameter_recovery.py     # Synthetic validation (future)
```

### Recipe 0: Null Resampling Distribution

**File**: `recipes/recipe_0_null_resampling.py`

**Question**: "Is θ̂ significantly different from θ=0?"

**Purpose**: Diagnostic - tests whether inferred constraints are real or noise

```python
@dataclass
class NullResamplingReport(DiagnosticReport):
    """
    Report from null resampling diagnostic.
    
    Attributes:
        severity: 'ok', 'warning', 'fail'
        observed_delta_ll: Observed ΔLL at θ̂
        null_distribution: Array of ΔLL values under null
        p_value: Fraction of null samples ≥ observed
        interpretation: Human-readable interpretation
        recommendation: What to do next
    """
    severity: str  # 'ok', 'warning', 'fail'
    observed_delta_ll: float
    null_distribution: np.ndarray
    p_value: float
    interpretation: str
    recommendation: str
    
    def print_summary(self):
        """Print null resampling summary."""
        print("=" * 70)
        print("NULL RESAMPLING DIAGNOSTIC")
        print("=" * 70)
        print(f"\nObserved ΔLL: {self.observed_delta_ll:.2f}")
        print(f"Null distribution: mean={np.mean(self.null_distribution):.2f}, "
              f"std={np.std(self.null_distribution):.2f}")
        print(f"P-value: {self.p_value:.3f}")
        print(f"\nInterpretation: {self.interpretation}")
        print(f"Recommendation: {self.recommendation}")
        print("=" * 70)

def null_resampling_diagnostic(
    artifacts: InferenceArtifacts,
    cache: CachedPathData,
    n_resamples: int = 1000,
) -> NullResamplingReport:
    """
    Recipe 0: Null Resampling Diagnostic
    
    Tests whether θ̂ is significantly better than θ=0 by
    resampling the null distribution via importance weights.
    
    Args:
        artifacts: InferenceArtifacts from fit_assembly_constraints
        cache: CachedPathData from inference
        n_resamples: Number of bootstrap resamples
    
    Returns:
        NullResamplingReport with p-value and interpretation
    
    Example:
        >>> from persiste.plugins.assembly.recipes import null_resampling_diagnostic
        >>> report = null_resampling_diagnostic(artifacts, cache)
        >>> report.print_summary()
        >>> if report.p_value < 0.05:
        ...     print("Constraints are significant!")
    """
    # Use existing implementation from diagnostics/suite.py
    result = null_resampling(artifacts, cache, n_resamples)
    
    # Generate interpretation with severity
    if result.p_value < 0.01:
        severity = 'ok'
        interpretation = "STRONG evidence for constraints (p < 0.01)"
        recommendation = "Constraints are well-supported; proceed with interpretation"
    elif result.p_value < 0.05:
        severity = 'ok'
        interpretation = "MODERATE evidence for constraints (p < 0.05)"
        recommendation = "Constraints likely real; consider additional validation"
    else:
        severity = 'warning'
        interpretation = "WEAK evidence for constraints (p ≥ 0.05)"
        recommendation = "Constraints may be noise; do not over-interpret"
    
    return NullResamplingReport(
        severity=severity,
        observed_delta_ll=result.observed_delta_ll,
        null_distribution=result.null_samples,
        p_value=result.p_value,
        interpretation=interpretation,
        recommendation=recommendation,
    )
```

---

### Recipe 1: Profile Likelihood Sweeps

**File**: `recipes/recipe_1_profile_likelihood.py`

**Question**: "How well-constrained is each parameter?"

**Purpose**: Diagnostic - characterizes uncertainty in θ̂

```python
@dataclass
class ProfileLikelihoodReport(DiagnosticReport):
    """
    Report from profile likelihood analysis.
    
    Attributes:
        severity: 'ok', 'warning', 'fail'
        feature_name: Feature being profiled
        mle: Maximum likelihood estimate
        confidence_interval: 95% CI
        grid_values: Grid of θ values tested
        log_likelihoods: Log-likelihood at each grid point
        interpretation: Human-readable interpretation
        recommendation: What to do next
    """
    severity: str
    feature_name: str
    mle: float
    confidence_interval: tuple[float, float]
    grid_values: np.ndarray
    log_likelihoods: np.ndarray
    interpretation: str
    recommendation: str
    
    def print_summary(self):
        """Print profile likelihood summary."""
        print("=" * 70)
        print(f"PROFILE LIKELIHOOD: {self.feature_name}")
        print("=" * 70)
        print(f"\nMLE: {self.mle:.2f}")
        print(f"95% CI: [{self.confidence_interval[0]:.2f}, {self.confidence_interval[1]:.2f}]")
        print(f"CI width: {self.confidence_interval[1] - self.confidence_interval[0]:.2f}")
        print(f"\nInterpretation: {self.interpretation}")
        print("=" * 70)
    
    def plot(self, save_path: str = None):
        """Plot profile likelihood curve."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.grid_values, self.log_likelihoods, 'b-', lw=2)
        ax.axvline(self.mle, color='r', linestyle='--', label=f'MLE = {self.mle:.2f}')
        ax.axvline(self.confidence_interval[0], color='g', linestyle=':', label='95% CI')
        ax.axvline(self.confidence_interval[1], color='g', linestyle=':')
        ax.set_xlabel(f'{self.feature_name}', fontsize=12)
        ax.set_ylabel('Log-likelihood', fontsize=12)
        ax.set_title(f'Profile Likelihood: {self.feature_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def profile_likelihood_sweep(
    artifacts: InferenceArtifacts,
    cache: CachedPathData,
    feature_name: str,
    n_grid: int = 21,
) -> ProfileLikelihoodReport:
    """
    Recipe 1: Profile Likelihood Sweep
    
    Computes profile likelihood for a single feature to
    characterize uncertainty in θ̂.
    
    Args:
        artifacts: InferenceArtifacts from fit_assembly_constraints
        cache: CachedPathData from inference
        feature_name: Feature to profile
        n_grid: Number of grid points
    
    Returns:
        ProfileLikelihoodReport with CI and interpretation
    
    Example:
        >>> from persiste.plugins.assembly.recipes import profile_likelihood_sweep
        >>> report = profile_likelihood_sweep(artifacts, cache, 'reuse_count')
        >>> report.print_summary()
        >>> report.plot(save_path='profile_reuse_count.png')
    """
    # Use existing implementation from diagnostics/suite.py
    result = profile_likelihood(artifacts, cache, feature_name, n_grid)
    
    # Generate interpretation with severity
    ci_width = result.confidence_interval[1] - result.confidence_interval[0]
    if ci_width < 0.5:
        severity = 'ok'
        interpretation = "WELL-CONSTRAINED parameter (narrow CI)"
        recommendation = "Parameter estimate is reliable"
    elif ci_width < 1.5:
        severity = 'ok'
        interpretation = "MODERATELY constrained parameter"
        recommendation = "Parameter estimate has moderate uncertainty"
    else:
        severity = 'warning'
        interpretation = "POORLY constrained parameter (wide CI)"
        recommendation = "Parameter estimate is uncertain; interpret cautiously"
    
    return ProfileLikelihoodReport(
        severity=severity,
        feature_name=feature_name,
        mle=result.mle,
        confidence_interval=result.confidence_interval,
        grid_values=result.grid_values,
        log_likelihoods=result.log_likelihoods,
        interpretation=interpretation,
        recommendation=recommendation,
    )
```

---

### Recipe 2: Baseline Perturbation Sensitivity

**File**: `recipes/recipe_2_baseline_perturbation.py`

**Question**: "How sensitive is θ̂ to baseline misspecification?"

**Purpose**: Stress test - reveals fragility under model violations

```python
@dataclass
class BaselinePerturbationReport(DiagnosticReport):
    """
    Report from baseline perturbation analysis.
    
    Attributes:
        severity: 'ok', 'warning', 'fail'
        baseline_original: Original baseline parameters
        perturbations: List of perturbed baselines tested
        theta_hats: θ̂ under each perturbation
        theta_shifts: Relative shifts in θ̂
        max_shift: Maximum shift observed
        interpretation: Human-readable interpretation
        recommendation: What to do next
    """
    severity: str
    baseline_original: dict
    perturbations: list[dict]
    theta_hats: list[dict]
    theta_shifts: list[float]
    max_shift: float
    interpretation: str
    recommendation: str
    
    def print_summary(self):
        """Print baseline perturbation summary."""
        print("=" * 70)
        print("BASELINE PERTURBATION SENSITIVITY")
        print("=" * 70)
        print(f"\nOriginal baseline: {self.baseline_original}")
        print(f"\nTested {len(self.perturbations)} perturbations:")
        for i, (pert, shift) in enumerate(zip(self.perturbations, self.theta_shifts)):
            print(f"  {i+1}. {pert}: shift = {shift:.1f}%")
        print(f"\nMax shift: {self.max_shift:.1f}%")
        print(f"\nInterpretation: {self.interpretation}")
        print(f"Recommendation: {self.recommendation}")
        print("=" * 70)

def baseline_perturbation_sensitivity(
    observed_compounds: set[str],
    primitives: list[str],
    baseline_original: AssemblyBaseline,
    initial_state: AssemblyState,
    perturbation_levels: list[float] = [0.1, 0.2, 0.5],
    n_samples: int = 100,
) -> BaselinePerturbationReport:
    """
    Recipe 2: Baseline Perturbation Sensitivity
    
    Tests how θ̂ changes when baseline parameters are perturbed.
    Reveals fragility under model misspecification.
    
    Args:
        observed_compounds: Set of observed compound identifiers
        primitives: List of primitive building blocks
        baseline_original: Original baseline used in inference
        initial_state: Initial assembly state
        perturbation_levels: List of relative perturbations (e.g., [0.1, 0.2])
        n_samples: Number of simulation samples per perturbation
    
    Returns:
        BaselinePerturbationReport with sensitivity analysis
    
    Example:
        >>> from persiste.plugins.assembly.recipes import baseline_perturbation_sensitivity
        >>> report = baseline_perturbation_sensitivity(
        ...     observed_compounds, primitives, baseline, initial_state
        ... )
        >>> report.print_summary()
        >>> if report.max_shift > 50:
        ...     print("WARNING: Inference highly sensitive to baseline!")
    """
    from persiste.plugins.assembly.cli import fit_assembly_constraints, InferenceMode
    
    # Fit with original baseline
    result_original = fit_assembly_constraints(
        observed_compounds=observed_compounds,
        primitives=primitives,
        mode=InferenceMode.SCREEN_AND_REFINE,
        n_samples=n_samples,
    )
    theta_original = result_original['theta_hat']
    
    # Perturb baseline and refit
    perturbations = []
    theta_hats = []
    theta_shifts = []
    
    for level in perturbation_levels:
        # Perturb kappa
        baseline_perturbed = AssemblyBaseline(
            kappa=baseline_original.kappa * (1 + level),
            join_exponent=baseline_original.join_exponent,
            split_exponent=baseline_original.split_exponent,
            decay_rate=baseline_original.decay_rate,
        )
        
        result_perturbed = fit_assembly_constraints(
            observed_compounds=observed_compounds,
            primitives=primitives,
            mode=InferenceMode.SCREEN_AND_REFINE,
            n_samples=n_samples,
            kappa=baseline_perturbed.kappa,
        )
        theta_perturbed = result_perturbed['theta_hat']
        
        # Compute shift
        if theta_original:
            shift = 100 * np.linalg.norm(
                [theta_perturbed.get(k, 0) - theta_original.get(k, 0) for k in theta_original]
            ) / max(np.linalg.norm(list(theta_original.values())), 1e-6)
        else:
            shift = 0.0
        
        perturbations.append({'kappa': f'+{level*100:.0f}%'})
        theta_hats.append(theta_perturbed)
        theta_shifts.append(shift)
    
    max_shift = max(theta_shifts)
    
    # Generate interpretation with severity
    if max_shift > 100:
        severity = 'fail'
        interpretation = "EXTREME sensitivity to baseline (shifts > 100%)"
        recommendation = "Validate baseline before trusting constraints"
    elif max_shift > 50:
        severity = 'warning'
        interpretation = "HIGH sensitivity to baseline (shifts > 50%)"
        recommendation = "Use conservative ΔLL thresholds"
    elif max_shift > 20:
        severity = 'warning'
        interpretation = "MODERATE sensitivity to baseline"
        recommendation = "Constraints likely robust but verify baseline"
    else:
        severity = 'ok'
        interpretation = "LOW sensitivity to baseline (robust)"
        recommendation = "Constraints appear stable under perturbations"
    
    return BaselinePerturbationReport(
        severity=severity,
        baseline_original={'kappa': baseline_original.kappa},
        perturbations=perturbations,
        theta_hats=theta_hats,
        theta_shifts=theta_shifts,
        max_shift=max_shift,
        interpretation=interpretation,
        recommendation=recommendation,
    )
```

---

### Recipe 3: Parameter Recovery (Future)

**File**: `recipes/recipe_3_parameter_recovery.py`

**Question**: "Can we recover known θ from synthetic data?"

**Purpose**: Validation - tests inference pipeline end-to-end

**Status**: Interface defined, implementation deferred to future work

---

### Recipes README

**File**: `recipes/README.md`

Structure:
1. Overview of available recipes
2. Question each recipe answers
3. Usage examples
4. Interpretation guidelines
5. Recommended workflow
6. Sample size requirements
7. Technical details

(See copynumber/recipes/README.md as template)

---

## Integration: ConstraintResult Extension

### Current ConstraintResult (Hypothetical)

```python
@dataclass
class ConstraintResult:
    theta_hat: dict[str, float]
    log_likelihood: float
    delta_ll: float
    screening_results: list[ScreeningResult]
    cache_stats: dict
```

### Extended ConstraintResult

```python
@dataclass
class ConstraintResult:
    theta_hat: dict[str, float]
    log_likelihood: float
    delta_ll: float
    screening_results: list[ScreeningResult]
    cache_stats: dict
    
    # NEW: Safety section
    safety: SafetyReport
    
    def is_safe(self) -> bool:
        """Check if result is safe for interpretation."""
        return self.safety.overall_safe
    
    def print_safety_summary(self):
        """Print safety summary."""
        self.safety.print_summary()
```

### CLI Integration: Safety-Only Mode

**New CLI flag**: `--safety-only`

Enables pre-flight checks without running full inference:

```bash
fit_assembly_constraints --safety-only \
  --compounds A,B,C \
  --primitives A,B
```

Behavior:
1. Runs baseline + identifiability + cache checks
2. Returns a `SafetyReport`
3. Does not optimize θ
4. Fast (~seconds instead of minutes)

Use cases:
- Pre-flight checks before expensive inference
- Dataset vetting
- Teaching users good diagnostic habits
- Batch safety screening

### Integration Point: fit_assembly_constraints

**File**: `src/persiste/plugins/assembly/cli.py`

```python
def fit_assembly_constraints(
    observed_compounds: set[str],
    primitives: list[str],
    *,
    mode: InferenceMode = InferenceMode.FULL_STOCHASTIC,
    skip_safety_checks: bool = False,  # NEW: opt-out flag
    safety_only: bool = False,  # NEW: pre-flight mode
    **kwargs,
) -> dict:
    """
    Fit assembly constraints with automatic safety checks.
    
    Args:
        ...
        skip_safety_checks: Skip Tier 1 safety checks (advanced users only)
        safety_only: Run only safety checks, skip inference (pre-flight mode)
    
    Returns:
        Dict with keys:
        - theta_hat: Estimated feature weights (None if safety_only=True)
        - screening_results: List of screening results
        - cache_stats: Cache statistics
        - safety: SafetyReport
    """
    # Safety-only mode: run checks and return early
    if safety_only:
        from persiste.plugins.assembly.safety import run_safety_checks
        
        # Run minimal screening for identifiability check
        screening_results = screen_hypotheses(
            hypotheses=[{}],  # Just null model
            model=SteadyStateAssemblyModel(...),
            observed=observed_compounds,
            initial_state=initial_state,
        )
        
        safety_report = run_safety_checks(
            observed_compounds=observed_compounds,
            primitives=primitives,
            baseline=baseline,
            initial_state=initial_state,
            theta_hat={},
            screening_results=screening_results,
            cache_stats={'initialized': False},
        )
        
        return {
            'mode': 'safety-only',
            'theta_hat': None,
            'safety': safety_report,
        }
    
    # ... existing inference logic ...
    
    # Run Tier 1 safety checks (unless skipped)
    if not skip_safety_checks:
        from persiste.plugins.assembly.safety import run_safety_checks
        
        safety_report = run_safety_checks(
            observed_compounds=observed_compounds,
            primitives=primitives,
            baseline=baseline,
            initial_state=initial_state,
            theta_hat=result['theta_hat'],
            screening_results=result['screening_results'],
            cache_stats=result['cache_stats'],
        )
        
        result['safety'] = safety_report
        
        # Adjust ΔLL threshold if needed
        if not safety_report.overall_safe:
            logger.warning("Safety checks failed; adjusting ΔLL threshold")
            # Apply adjusted threshold to screening results
            for r in result['screening_results']:
                r.passed = r.normalized_delta_ll > safety_report.adjusted_delta_ll_threshold
    else:
        result['safety'] = None
    
    return result
```

---

## File Structure Summary

```
src/persiste/plugins/assembly/
├── safety/                          # NEW: Tier 1 automatic diagnostics
│   ├── __init__.py
│   ├── baseline_check.py
│   ├── identifiability.py
│   ├── cache_reliability.py
│   └── safety_report.py
├── recipes/                         # NEW: Tier 2 opt-in diagnostics
│   ├── __init__.py
│   ├── README.md
│   ├── recipe_0_null_resampling.py
│   ├── recipe_1_profile_likelihood.py
│   ├── recipe_2_baseline_perturbation.py
│   └── recipe_3_parameter_recovery.py  # Future
├── diagnostics/                     # EXISTING: Diagnostic artifacts
│   ├── artifacts.py
│   └── suite.py
├── cli.py                           # MODIFIED: Add safety integration
└── ...
```

---

## Usage Examples

### Tier 1: Automatic (Default Behavior)

```python
from persiste.plugins.assembly.cli import fit_assembly_constraints, InferenceMode

# Tier 1 checks run automatically
result = fit_assembly_constraints(
    observed_compounds={"A", "B", "C"},
    primitives=["A", "B"],
    mode=InferenceMode.SCREEN_AND_REFINE,
)

# Check safety
if not result['safety'].overall_safe:
    print("WARNING: Inference may be unreliable!")
    result['safety'].print_summary()

# Advanced users can skip (not recommended)
result_unsafe = fit_assembly_constraints(
    observed_compounds={"A", "B", "C"},
    primitives=["A", "B"],
    mode=InferenceMode.SCREEN_AND_REFINE,
    skip_safety_checks=True,  # Danger zone
)
```

### Tier 2: Opt-in Recipes

```python
from persiste.plugins.assembly.recipes import (
    null_resampling_diagnostic,
    profile_likelihood_sweep,
    baseline_perturbation_sensitivity,
)

# After inference, run deep diagnostics
artifacts = result['artifacts']  # Assume this is saved
cache = result['cache']

# Recipe 0: Null resampling
null_report = null_resampling_diagnostic(artifacts, cache)
null_report.print_summary()

# Recipe 1: Profile likelihood
profile_report = profile_likelihood_sweep(artifacts, cache, 'reuse_count')
profile_report.print_summary()
profile_report.plot(save_path='profile_reuse_count.png')

# Recipe 2: Baseline perturbation
pert_report = baseline_perturbation_sensitivity(
    observed_compounds, primitives, baseline, initial_state
)
pert_report.print_summary()
```

---

## Implementation Checklist

### Phase 1: Tier 1 Safety Checks
- [ ] Create `safety/` module structure
- [ ] Implement `baseline_check.py`
- [ ] Implement `identifiability.py`
- [ ] Implement `cache_reliability.py`
- [ ] Implement `safety_report.py`
- [ ] Integrate into `cli.py`
- [ ] Add tests for each safety check
- [ ] Document safety thresholds

### Phase 2: Tier 2 Recipes
- [ ] Create `recipes/` module structure
- [ ] Implement `recipe_0_null_resampling.py`
- [ ] Implement `recipe_1_profile_likelihood.py`
- [ ] Implement `recipe_2_baseline_perturbation.py`
- [ ] Write comprehensive `recipes/README.md`
- [ ] Add examples to `examples/` directory
- [ ] Add tests for each recipe

### Phase 3: Integration & Documentation
- [ ] Extend `ConstraintResult` with `safety` field
- [ ] Update CLI help text with safety flags
- [ ] Write user guide for diagnostics
- [ ] Add diagnostic workflow to main README
- [ ] Create tutorial notebook

---

## Design Rationale

### Why Two Tiers?

1. **Tier 1 must be automatic** because users cannot reliably detect baseline misspecification
2. **Tier 2 must be opt-in** to avoid 5-10 min workflows and opaque behavior
3. **Separation preserves existing decoupling** of diagnostics from inference

### Why Not Bundle Everything?

Previous experience showed that bundling diagnostics into inference creates:
- Long runtimes (5-10 min)
- Opaque behavior
- User confusion

The two-tier model balances safety with usability.

### Why Follow copynumber/genecontent Pattern?

1. **Proven design**: Already works well for other plugins
2. **User familiarity**: Consistent API across plugins
3. **Maintainability**: Clear separation of concerns
4. **Extensibility**: Easy to add new recipes

---

## User-Facing Documentation Framing

### Critical Framing for Users

**Key Message**:
> "Safety checks assess whether constraint inference is interpretable, not whether it is true."

### What Safety Checks Are

- **Guardrails**, not judges
- Detect when inference may be unreliable
- Flag when interpretation should be cautious
- Provide actionable recommendations

### What Safety Checks Are NOT

- **Not guarantees of correctness**: Passing checks ≠ guaranteed correct inference
- **Not blockers**: Failing checks ≠ analysis is useless
- **Not substitutes for domain knowledge**: Users must still interpret results critically

### Documentation Language

**Do say**:
- "Safety checks suggest the baseline may not fit the data well"
- "Constraints appear identifiable from screening"
- "Importance sampling is stable at θ̂"

**Don't say**:
- "Safety checks prove the baseline is wrong" (too strong)
- "Constraints are correct" (not testable)
- "You must not interpret these results" (too prescriptive)

### User Guide Structure

1. **What are safety checks?** (conceptual overview)
2. **When do they run?** (automatic by default)
3. **What do the warnings mean?** (interpretation guide)
4. **What should I do if checks fail?** (actionable steps)
5. **Can I skip them?** (yes, but not recommended)
6. **How do I run deep diagnostics?** (Tier 2 recipes)

### Example Warning Messages

**Good**:
> ⚠️ Baseline sanity check failed (severe). The baseline model poorly explains observed compound distributions. Constraint estimates may be biased. Consider validating baseline parameters before interpretation.

**Bad**:
> ❌ Baseline is wrong. Do not use these results.

### Defusing User Frustration

Anticipate common reactions:

**User**: "Why did my analysis fail safety checks?"
**Response**: "Safety checks flag potential issues, not failures. Your θ̂ is still valid, but we recommend extra caution when interpreting it. See the recommendations for next steps."

**User**: "Can I just skip the checks?"
**Response**: "Yes, use `--skip-safety-checks`, but this is not recommended unless you have strong reasons to trust your baseline and data quality."

**User**: "My reviewer says safety checks prove my results are wrong."
**Response**: "Safety checks assess interpretability, not truth. Failing checks means 'interpret cautiously,' not 'results are invalid.' Share the SafetyReport with your reviewer to show you've done due diligence."

---

## Open Questions

1. **Baseline sanity check**: Simple heuristics (compound count, depth histogram) or more sophisticated metrics?
2. **Identifiability threshold**: Is variance < 1.0 too strict? Should it be adaptive based on sample size?
3. **Cache reliability**: Should we track ESS history across resimulations for trend detection?
4. **Recipe naming**: Numbers (recipe_0, recipe_1) or descriptive names (recipe_null_resampling)?
5. **CLI flags**: Should `--skip-safety-checks` require `--force-inference` for extra confirmation?

---

## Next Steps

1. Review this plan with user
2. Prioritize implementation phases
3. Start with Tier 1 (highest impact)
4. Validate thresholds with existing robustness tests
5. Implement Tier 2 recipes incrementally
