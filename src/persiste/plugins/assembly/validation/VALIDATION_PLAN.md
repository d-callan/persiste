# Assembly Plugin: External Validation Implementation Plan

**Date:** January 2026  
**Status:** Implementation roadmap  
**Context:** Tier-1 safety checks are complete; now validating their effectiveness under adversarial conditions

---

## Overview

This plan addresses the remaining validation gaps before real-world deployment:

1. **Baseline adequacy validation** (highest priority) - Does safety catch baseline misspecification?
2. **Power/identifiability envelope** - When should users expect failure?
3. **Cache + IS drift testing** - Does resimulation happen correctly during optimization?
4. **Observation-model stress testing** - Do warnings fire under weak observations?

---

## A. Baseline Adequacy Validation (PRIORITY)

### Goal
Quantify how Tier-1 baseline checks track baseline quality and suppress false positives.

### A1: Dataset Generator for Baseline Misspecification

**File:** `validation/scripts/generate_baseline_scenarios.py`

**Functionality:**
```python
def generate_baseline_scenario(
    scenario: Literal["correct", "mild", "severe"],
    primitives: list[str],
    max_depth: int,
    n_trajectories: int,
    seed: int,
) -> dict:
    """
    Generate synthetic dataset with controlled baseline error.
    
    Scenarios:
    - correct: Use ground-truth baseline parameters
    - mild: Perturb join_exponent by ±0.2, decay by ±20%
    - severe: Perturb join_exponent by ±0.5, decay by ±50%
    
    Returns:
        {
            "primitives": [...],
            "observed_compounds": {...},
            "true_baseline": {...},
            "inference_baseline": {...},  # what we'll use for inference
            "true_theta": {...},
            "scenario": "correct|mild|severe",
            "baseline_error_magnitude": float,
        }
    """
```

**Implementation steps:**
1. Extend `generate_benchmark_dataset.py` with baseline perturbation logic
2. Add three scenario templates (correct/mild/severe)
3. Store both true and inference baselines in output JSON
4. Include error magnitude metric for plotting

**Output:** `validation/results/baseline_scenarios/{correct,mild,severe}_*.json`

---

### A2: Run Tier-1 Checks + Inference on All Scenarios

**File:** `validation/experiments/baseline_adequacy_validation.py`

**Functionality:**
```python
def validate_baseline_adequacy():
    """
    For each baseline scenario:
    1. Run safety-only mode to get baseline check severity
    2. Run full inference with ΔLL escalation
    3. Record:
       - Baseline check severity (PASS/LOW/MEDIUM/HIGH)
       - ΔLL threshold used
       - Whether false positive occurred (θ̂ ≠ 0 when true θ = 0)
       - Whether true positive suppressed (θ̂ = 0 when true θ ≠ 0)
    """
```

**Implementation steps:**
1. Load all scenario datasets from A1
2. For each scenario:
   - Call `fit_assembly_constraints(..., mode=SAFETY_ONLY)`
   - Parse `SafetyReport` severity
   - Call `fit_assembly_constraints(..., mode=FULL_STOCHASTIC)` with normal + null θ
   - Compute ΔLL and check against escalated threshold
3. Aggregate results into structured output

**Dependencies:** A1 complete

**Output:** `validation/results/baseline_adequacy_results.json`

---

### A3: Produce Table/Plot of Baseline Error → Safety Severity → FP Rate

**File:** `validation/experiments/plot_baseline_adequacy.py`

**Deliverable:**
- **Table:** 
  ```
  Scenario | Baseline Error | Safety Severity | ΔLL Threshold | FP Rate | FN Rate
  ---------|----------------|-----------------|---------------|---------|--------
  Correct  | 0.0            | PASS            | 2.0           | 0%      | 0%
  Mild     | 0.25           | LOW/MEDIUM      | 4.0           | 5%      | 0%
  Severe   | 0.6            | HIGH            | 10.0          | 0%      | 15%
  ```

- **Plot:** Baseline error (x-axis) vs FP rate (y-axis), colored by severity level

**Implementation steps:**
1. Parse `baseline_adequacy_results.json`
2. Generate markdown table
3. Create matplotlib plot (error vs FP rate, severity as color)
4. Save to `validation/results/baseline_adequacy_summary.{md,png}`

**Dependencies:** A2 complete

**Output:** `validation/results/baseline_adequacy_summary.{md,png}`

---

## B. Power / Identifiability Envelope

### Goal
Map the regime where θ recovery is reliable vs unreliable.

### B1: Design Coarse Grid

**File:** `validation/scripts/generate_power_grid.py`

**Grid dimensions:**
- **Primitives:** [4, 6, 8]
- **Max depth:** [3, 5, 7]
- **Trajectories:** [50, 100, 200]
- **Observation richness:** ["presence", "frequency"]
- **True θ:** [0.0 (null), 1.0 (signal)]

**Total scenarios:** 3 × 3 × 3 × 2 × 2 = 108 datasets

**Implementation steps:**
1. Generate all combinations using itertools.product
2. For each combination, simulate dataset with known θ
3. Store as `validation/results/power_grid/{primitives}_{depth}_{traj}_{obs}_{theta}.json`

**Output:** 108 JSON files in `validation/results/power_grid/`

---

### B2: Run Simulation Sweep and Record θ Recovery Probability

**File:** `validation/experiments/power_envelope_sweep.py`

**Functionality:**
```python
def compute_recovery_probability(dataset_path: Path, n_replicates: int = 10) -> dict:
    """
    For a given dataset:
    1. Run inference n_replicates times with different seeds
    2. Record how often θ̂ correctly identifies signal (ΔLL > threshold)
    3. Return recovery rate
    """
```

**Implementation steps:**
1. Iterate over all power grid datasets
2. For each dataset, run 10 inference replicates
3. Compute recovery probability: `P(ΔLL > threshold | true θ ≠ 0)`
4. Aggregate into summary table

**Dependencies:** B1 complete

**Output:** `validation/results/power_envelope_results.json`

---

### B3: Generate Heatmap/Table and Integrate into Docs

**File:** `validation/experiments/plot_power_envelope.py`

**Deliverable:**
- **Heatmap:** Primitives × Depth, colored by recovery probability, faceted by trajectories/obs type
- **Table:** Summary showing "green/yellow/red" zones:
  ```
  Regime                          | Recovery Probability | Recommendation
  --------------------------------|----------------------|---------------
  ≥6 primitives, depth≤5, ≥100 traj | >90%               | Safe
  4 primitives, depth=7, 50 traj    | 40%                | Expect failure
  ```

**Implementation steps:**
1. Parse `power_envelope_results.json`
2. Generate seaborn heatmap
3. Create markdown summary table with actionable recommendations
4. Save to `validation/results/power_envelope_summary.{md,png}`
5. Add reference to `docs/ASSEMBLY_DIAGNOSTICS_PLAN.md` and recipes README

**Dependencies:** B2 complete

**Output:** `validation/results/power_envelope_summary.{md,png}` + doc updates

---

## C. Cache + Importance Sampling Drift Test

### Goal
Verify cache resimulation triggers correctly during realistic optimization.

### C1: Script Realistic Optimization Trace

**File:** `validation/experiments/cache_drift_validation.py`

**Functionality:**
```python
def simulate_optimization_trace(
    initial_theta: dict[str, float],
    target_theta: dict[str, float],
    n_steps: int = 20,
) -> list[dict]:
    """
    Simulate iterative optimization from θ=0 toward θ̂~1.0.
    
    At each step:
    1. Propose new θ (linear interpolation + noise)
    2. Call fit_assembly_constraints with that θ
    3. Log cache status from returned artifacts
    
    Returns list of:
        {
            "step": int,
            "theta": dict,
            "cache_status": "valid|invalid",
            "invalidation_reason": str | None,
            "ess": float,
            "resimulation_count": int,
        }
    """
```

**Implementation steps:**
1. Start at θ = {feature: 0.0 for all features}
2. Linearly interpolate toward θ̂ = {feature: 1.0} over 20 steps
3. At each step, call CLI with that θ and capture cache stats
4. Store full trace as JSON

**Output:** `validation/results/cache_drift_trace.json`

---

### C2: Instrument and Verify Resimulation Triggers

**File:** `validation/experiments/analyze_cache_drift.py`

**Assertions to verify:**
1. **Trust region crossing:** When |Δθ| > trust_radius, cache invalidates
2. **ESS degradation:** When ESS < threshold, cache invalidates
3. **No stale reuse:** Cache never marked valid when θ far from θ_ref
4. **Optimizer stability:** ΔLL doesn't oscillate wildly due to resampling noise

**Implementation steps:**
1. Parse `cache_drift_trace.json`
2. For each invalidation, verify it matches expected trigger (trust region or ESS)
3. Check for any "valid" cache status when θ is far from θ_ref (bug indicator)
4. Plot timeline: step vs cache status, ESS, ΔLL
5. Generate pass/fail report

**Dependencies:** C1 complete

**Output:** `validation/results/cache_drift_report.{md,png}`

---

## D. Observation-Model Stress Testing

### Goal
Confirm Tier-1 identifiability warnings fire under weak observations.

### D1: Test Tier-1 Identifiability Under Weak Observation Regimes

**File:** `validation/experiments/observation_stress_test.py`

**Test scenarios:**
1. **Presence-only, high detection:** Standard case (baseline)
2. **Presence-only, low detection (p=0.3):** Should trigger identifiability warning
3. **Time-sliced presence (3 slices):** Richer, should pass
4. **Noisy counts (±30% error):** Should trigger warning or elevate threshold

**Functionality:**
```python
def test_observation_regime(
    regime: str,
    detection_prob: float,
    noise_level: float,
) -> dict:
    """
    Run Tier-1 safety checks under specified observation regime.
    
    Returns:
        {
            "regime": str,
            "identifiability_severity": str,
            "null_resampling_p_value": float,
            "baseline_check_severity": str,
            "overall_safety": str,
        }
    """
```

**Implementation steps:**
1. Generate dataset for each regime
2. Run `fit_assembly_constraints(..., mode=SAFETY_ONLY)`
3. Parse `SafetyReport` for each regime
4. Verify identifiability severity escalates in weak regimes

**Output:** `validation/results/observation_stress_results.json`

---

### D2: Document Trigger Thresholds and Warning Behavior

**File:** `validation/results/observation_stress_summary.md`

**Deliverable:**
- **Table:**
  ```
  Observation Regime           | Identifiability Severity | Recommended Action
  -----------------------------|--------------------------|-------------------
  Presence, p_detect > 0.7     | PASS                     | Proceed
  Presence, p_detect < 0.5     | HIGH                     | Add time slices or increase detection
  Time-sliced (≥3 slices)      | PASS                     | Proceed
  Noisy counts (>20% error)    | MEDIUM                   | Use robust obs model or increase sample size
  ```

**Implementation steps:**
1. Parse `observation_stress_results.json`
2. Generate markdown summary table
3. Add to `recipes/README.md` as "When to trust results" section

**Dependencies:** D1 complete

**Output:** `validation/results/observation_stress_summary.md` + doc updates

---

## E. Baseline Calibration Helper (Nice to Have)

### Goal
Enable users to fit baseline parameters independently, improving confidence in downstream inference.

### E1: Implement `--baseline-only` CLI Mode

**File:** `src/persiste/plugins/assembly/cli.py`

**Functionality:**
```python
# Add to InferenceMode enum:
class InferenceMode(Enum):
    ...
    BASELINE_ONLY = "baseline-only"

# In fit_assembly_constraints():
if mode == InferenceMode.BASELINE_ONLY:
    """
    Fit baseline parameters (κ, join_exponent, decay) using grid search + refinement.
    Return goodness-of-fit metrics and diagnostics.
    """
    return {
        "mode": "baseline-only",
        "baseline_fit": {
            "kappa": float,
            "join_exponent": float,
            "decay": float,
        },
        "goodness_of_fit": {
            "log_likelihood": float,
            "aic": float,
            "residuals": list[float],
        },
        "diagnostics": {
            "rate_scaling_plot": str,  # path to PNG
            "residual_plot": str,      # path to PNG
        },
    }
```

**Implementation steps:**
1. Add `BASELINE_ONLY` to `InferenceMode` enum
2. Implement baseline-only inference path (grid search on κ, exponents, decay)
3. Compute AIC/residuals for goodness-of-fit
4. Generate diagnostic plots (rate scaling, residuals)
5. Add CLI argument `--baseline-only` that sets mode

**Output:** Baseline fit JSON + diagnostic plots

---

### E2: Create Baseline Calibration Script

**File:** `validation/scripts/calibrate_baseline.py`

**Functionality:**
```python
def calibrate_baseline_from_data(
    observed_compounds: set[str],
    primitives: list[str],
    max_depth: int,
    n_samples: int = 200,
) -> dict:
    """
    Fit baseline parameters from observed data.
    
    Returns:
        {
            "baseline_parameters": {...},
            "goodness_of_fit": {...},
            "confidence_intervals": {...},
            "recommendation": str,  # "Use this baseline" or "Collect more data"
        }
    """
```

**Implementation steps:**
1. Wrap `fit_assembly_constraints(..., mode=BASELINE_ONLY)`
2. Add confidence interval estimation (bootstrap or profile likelihood)
3. Provide recommendation based on AIC/residuals
4. Output JSON + plots for user review

**Output:** Baseline calibration report

---

## F. Enhanced Safe-Failure Messaging

### Goal
Replace generic warnings with actionable guidance that helps users move forward.

### F1: Extend SafetyReport with Diagnostic Context

**File:** `src/persiste/plugins/assembly/safety/safety_report.py`

**Enhancements:**
```python
@dataclass
class SafetyReport:
    # ... existing fields ...
    
    # NEW: Actionable guidance
    baseline_diagnostics: dict = field(default_factory=dict)
    # {
    #     "observed_decay_rate": float,
    #     "baseline_decay_rate": float,
    #     "decay_ratio": float,
    #     "recommendation": str,  # e.g., "Decay ~3× faster than baseline; consider refitting baseline"
    # }
    
    identifiability_diagnostics: dict = field(default_factory=dict)
    # {
    #     "observation_richness": str,  # "presence-only", "frequency", "time-sliced"
    #     "detection_probability": float,
    #     "recommendation": str,  # e.g., "Low detection (p=0.3); add time slices or increase sampling"
    # }
    
    def print_summary(self) -> str:
        """Human-readable summary with actionable next steps."""
        # Format: severity level + specific issue + recommended action
```

**Implementation steps:**
1. Add diagnostic context fields to `SafetyReport`
2. Populate during baseline_check, identifiability, cache_reliability
3. Implement `print_summary()` that formats as:
   ```
   ⚠️  BASELINE SUSPECT (severity: MEDIUM)
   
   Issue: Observed decay rate exceeds baseline by ~3×
   
   Possible causes:
   - Baseline parameters (κ, exponents) are incorrect
   - Data includes unmodeled assembly dynamics
   
   Recommended actions:
   1. Run: persiste assembly --baseline-only <data>
   2. Compare fitted baseline to current baseline
   3. If different, rerun inference with fitted baseline
   4. If similar, consider adding constraints for decay
   ```

**Output:** Enhanced `SafetyReport` with guidance

---

### F2: Update CLI to Display Enhanced Messages

**File:** `src/persiste/plugins/assembly/cli.py`

**Functionality:**
```python
def fit_assembly_constraints(...):
    ...
    result = {...}
    
    # NEW: Print enhanced safety messages if warnings present
    if result.get("safety_report"):
        report = result["safety_report"]
        if report.severity != "PASS":
            print(report.print_summary())
            print("\nFor more details, see: docs/ASSEMBLY_DIAGNOSTICS_PLAN.md")
```

**Implementation steps:**
1. Call `report.print_summary()` after inference
2. Pipe to stdout with clear formatting
3. Add `--verbose` flag for detailed diagnostics
4. Reference docs for deeper guidance

**Output:** User-friendly CLI output

---

## Implementation Order

### Phase 1: Baseline Adequacy (Week 1)
- [ ] A1: Dataset generator
- [ ] A2: Run validation suite
- [ ] A3: Generate table/plot

**Blocker removal:** This is the critical path before real users.

### Phase 2: Power Envelope (Week 2)
- [ ] B1: Generate power grid
- [ ] B2: Run sweep
- [ ] B3: Heatmap + docs

**Value:** Prevents users from attempting impossible regimes.

### Phase 3: Cache Drift (Week 2-3)
- [ ] C1: Optimization trace script
- [ ] C2: Verification + report

**Value:** Ensures robustness reboot machinery works as designed.

### Phase 4: Observation Stress (Week 3)
- [ ] D1: Test weak observation regimes
- [ ] D2: Document thresholds

**Value:** Prevents misinterpretation of "clean" runs with no signal.

### Phase 5: Baseline Calibration Helper (Week 3-4, optional)
- [ ] E1: Implement `--baseline-only` CLI mode
- [ ] E2: Create calibration script

**Value:** Enables users to fit baseline independently; high ROI for serious users.

### Phase 6: Enhanced Safe-Failure Messaging (Week 4, optional)
- [ ] F1: Extend SafetyReport with diagnostic context
- [ ] F2: Update CLI to display enhanced messages

**Value:** Converts generic warnings into actionable guidance.

---

## Success Criteria

**Baseline adequacy:**
- ✓ Table shows severity tracks baseline error
- ✓ False positive rate <5% in severe scenarios (due to threshold escalation)
- ✓ No silent failures (all bad baselines flagged)

**Power envelope:**
- ✓ Clear "safe zone" documented
- ✓ Users know when to expect failure
- ✓ Integrated into diagnostics docs

**Cache drift:**
- ✓ All resimulations match expected triggers
- ✓ No stale cache reuse
- ✓ Optimizer stable across trust region crossings

**Observation stress:**
- ✓ Identifiability warnings fire in weak regimes
- ✓ Thresholds documented
- ✓ Actionable guidance provided

**Baseline calibration (E):**
- ✓ `--baseline-only` mode works end-to-end
- ✓ Goodness-of-fit metrics computed
- ✓ Diagnostic plots generated
- ✓ Users can compare fitted vs provided baseline

**Enhanced messaging (F):**
- ✓ SafetyReport includes diagnostic context
- ✓ `print_summary()` generates actionable guidance
- ✓ CLI displays messages automatically
- ✓ Users understand next steps instead of stopping

---

## Files to Create

### Scripts
- `validation/scripts/generate_baseline_scenarios.py`
- `validation/scripts/generate_power_grid.py`
- `validation/scripts/calibrate_baseline.py` (Phase 5)

### Experiments
- `validation/experiments/baseline_adequacy_validation.py`
- `validation/experiments/plot_baseline_adequacy.py`
- `validation/experiments/power_envelope_sweep.py`
- `validation/experiments/plot_power_envelope.py`
- `validation/experiments/cache_drift_validation.py`
- `validation/experiments/analyze_cache_drift.py`
- `validation/experiments/observation_stress_test.py`

### Results
- `validation/results/baseline_scenarios/*.json`
- `validation/results/baseline_adequacy_results.json`
- `validation/results/baseline_adequacy_summary.{md,png}`
- `validation/results/power_grid/*.json`
- `validation/results/power_envelope_results.json`
- `validation/results/power_envelope_summary.{md,png}`
- `validation/results/cache_drift_trace.json`
- `validation/results/cache_drift_report.{md,png}`
- `validation/results/observation_stress_results.json`
- `validation/results/observation_stress_summary.md`

### Code Changes (Phases 5-6)
- `src/persiste/plugins/assembly/cli.py` (add BASELINE_ONLY mode, enhanced output)
- `src/persiste/plugins/assembly/safety/safety_report.py` (add diagnostic context + print_summary)

---

## Regression Test Hooks

### Smoke Tests (import + basic execution)
**File:** `tests/plugins/assembly/test_validation_scripts.py`

```python
def test_baseline_scenario_generator_imports():
    """Verify generator script can be imported and called."""
    from validation.scripts.generate_baseline_scenarios import generate_baseline_scenario
    result = generate_baseline_scenario("correct", ["A", "B"], 3, 50, seed=42)
    assert "primitives" in result
    assert "baseline_error_magnitude" in result

def test_power_grid_generator_imports():
    """Verify power grid generator can be imported."""
    from validation.scripts.generate_power_grid import generate_power_grid
    grid = generate_power_grid()
    assert len(grid) == 108

def test_baseline_adequacy_imports():
    """Verify baseline adequacy validation can be imported."""
    from validation.experiments.baseline_adequacy_validation import validate_baseline_adequacy
    # Don't run full validation, just verify it exists and is callable
    assert callable(validate_baseline_adequacy)
```

### Golden Output Comparison (deterministic scenarios)
**File:** `tests/plugins/assembly/test_validation_golden.py`

```python
def test_baseline_adequacy_golden_output():
    """Compare baseline adequacy results against golden output."""
    result = validate_baseline_adequacy()
    golden = load_golden("baseline_adequacy_results.json")
    # Check that severity ordering matches: correct < mild < severe
    assert result["correct"]["severity"] <= result["mild"]["severity"]
    assert result["mild"]["severity"] <= result["severe"]["severity"]
```

---

## Notes

- All scripts should be runnable via CLI with `--help` documentation
- Results should be JSON for programmatic access + markdown/PNG for human review
- Smoke tests verify imports and basic execution; full validation runs are manual or CI-gated
- Golden output tests ensure results don't regress unexpectedly
- Integrate key findings into `docs/ASSEMBLY_DIAGNOSTICS_PLAN.md` and `recipes/README.md`
- Phase 5-6 items are optional but high-value; prioritize Phases 1-4 for real-world readiness
