# Assembly Constraint Validation Plan

This document captures the validation protocol for **all** assembly constraints:

| Layer / Feature family | Examples | Current regression coverage | Validation expansion plan |
| --- | --- | --- | --- |
| Core mechanics | `reuse_count`, `depth_change`, `size_change` | `tests/plugins/assembly/test_basics.py`, `test_interface.py`, `test_robustness_reboot.py` | Expand deterministic ΔLL asserts in `test_robustness_reboot.py` (Stage III). |
| Motif signals | `motif_gained_*`, `motif_lost_*` | `tests/plugins/assembly/test_constraints.py` motif fixtures | Add Stage II feature-null script (`src/persiste/plugins/assembly/wip/legacy_validation/run_motif_nulls.py`). |
| Transition type indicators | `transition_join/split/decay` | Smoke assertions in `test_core.py` | Plan new pytest parametrization (quick to run) mirroring Stage III. |
| Symmetry breaks A/B/C | depth gate, context-class, founder bias | None yet (new work) | Follow full validation stack + reusable scripts under `src/persiste/plugins/assembly/wip/legacy_validation/`. |

The goal is to ensure every constraint (legacy and new symmetry breaks) is calibrated, identifiable, specific, and power-profiled before being interpreted. Fast checks that keep regressions out of CI go into pytest where feasible; heavier sweeps live under `src/persiste/plugins/assembly/wip/legacy_validation/` so we can rerun them ad-hoc.

---

## V1. Null Calibration (θ = 0)

**Question:** When θ = 0 and data are generated from the baseline, do we get ΔLL ≈ 0?

### Requirements
- Run per symmetry break (A, B, C) individually and with all breaks switched off together.
- Both deterministic screening and stochastic refinement must report ΔLL ≈ 0 relative to the null baseline.
- Run enough seeds to measure variance; treat |ΔLL| > ε as a failure (ε tuned by dataset size, e.g., 0.5 for small grids).

### Procedure
1. Generate datasets with `simulate_assembly_trajectories` at θ = 0, using baseline config only.
2. For each inference mode:
   - Deterministic-only screen
   - Stochastic-only (full) inference
   - Screen + refine
3. Record ΔLL, normalized ΔLL, θ̂.
4. **Stop immediately** if ΔLL deviates materially from zero—this indicates a bug or scaling mismatch.

### Deliverables
- Table: symmetry break × mode × seed → ΔLL, θ̂.
- Plots: histogram of ΔLL under null for each mode.
- Additions to tests:
  - Extend `tests/plugins/assembly/test_robustness_reboot.py::TestCLI` with a `test_null_delta_ll_per_constraint` parametrized over constraint families (fast deterministic + small stochastic sample counts).
- Validation scripts (longer runs):
  - `src/persiste/plugins/assembly/wip/legacy_validation/run_null_calibration.py` sweeps seeds / modes; stores CSV of ΔLL summaries.

---

## V2. Identifiability / Power (θ ≠ 0)

**Question:** When θ ≠ 0 in the generator, does inference recover it?

### Axes to Sweep
1. **Effect size** (per constraint): 0 (null), weak (barely life-like), medium, strong.
2. **Data volume**: number of trajectories (50 / 100 / 200), max depth (5 / 7 / 9).
3. **Observation richness**:
   - Frequency only
   - Frequency + duration
   - Frequency + duration + reuse/depth stats

### Metrics
- Mean ΔLL and ΔLL variance per grid point.
- Recovery rate: % runs where θ̂ exceeds detection threshold.
- Bias: θ̂ − θ_true.
- ESS at θ̂ (to monitor importance sampling health).
- False positive rate at θ = 0 control points.

### Interpretation
- Locate detection thresholds (minimum effect size/data volume required).
- Determine whether constraints are sample-limited, resolution-limited, or unidentifiable with current observations.

---

## V3. Specificity (Constraint Isolation)

**Question:** Does constraint A lift only when A is present?

### Protocol
- For each constraint (A, B, C):
  1. Generate datasets with only that constraint’s θ ≠ 0, others at 0.
  2. Run inference allowing all features.
  3. Verify only the intended feature shows significant ΔLL / θ̂.
- Repeat with correlated stats (e.g., A + mild B) to stress-test feature leakage.

### Failure Modes to Catch
- Feature leakage (another feature mimics the signal)
- Correlated observation summaries causing cross-talk
- Over-general “life detector” behavior (everything lifts on any asymmetry)

---

## II. Validation Stage 1 — Feature-Level Null Tests (Fast, Surgical)

Before full power grids, validate each feature in isolation.

### Steps per Feature (legacy + symmetry breaks)
1. Run the baseline simulator (θ = 0) and extract only the target feature from path stats.
   - For legacy features (reuse, depth, size, motifs, transition flags), reuse existing `tests/plugins/assembly/test_constraints.py` helpers to keep the runtime low and add assertions on mean/variance.
   - For symmetry breaks, add `src/persiste/plugins/assembly/wip/legacy_validation/run_feature_nulls.py` (script) that uses a few dozen trajectories per seed.
2. Compute mean, variance, and distribution shape of that feature across trajectories.
3. Check for drifts with depth, trajectory count, primitive count.

### Expectations
- Mean ≈ 0 (or defined neutral value).
- No systematic drift.

### If Failure
- Feature is implicitly breaking symmetry (e.g., depth gate turning on unintentionally).
- Fix bug or scaling before integrating with inference.

This stage is quick and prevents chasing inference artifacts later.

---

## III. Validation Stage 2 — Deterministic Screen Sanity Check

Reintroduce the deterministic screening phase alone.

### Procedure
- For each null dataset (from Stage 1), run deterministic inference only.
- Log per-feature ΔLL and θ̂.
- **Tests to expand:** `TestCLI.test_screen_only_null_constraints` (new) to guard this pathway in CI.
- **Script counterpart:** `src/persiste/plugins/assembly/wip/legacy_validation/run_deterministic_nulls.py` for deeper sweeps.

### Expected Outcome
- ΔLL ≈ 0
- θ̂ ≈ 0
- No threshold escalation or auto-accept behavior

### Diagnosis Guide
- Deterministic lifts but stochastic does not → feature scaling / likelihood mismatch.
- Stochastic lifts but deterministic does not → importance sampling / ESS pathology (seen previously).

---

## IV. Power Grid Design (Core Experiment)

Combine results into full power grids.

### Minimum Viable Grid
1. **Effect size (θ)**: 0, weak, medium, strong per constraint.
2. **Data volume**: (n_samples, max_depth) combinations.
3. **Observation richness**: frequency-only, +duration, +reuse/depth features.

### Output Metrics per Grid Point
- Mean ΔLL
- Recovery rate (% θ̂ above threshold)
- Bias (θ̂ − θ_true)
- ESS at θ̂
- False positive rate at θ = 0

### Deliverables
- Heatmaps or contour plots per constraint showing detection frontier.
- Tables summarizing ESS behavior to flag IS breakdown regions.

---

## V. Stratified / Heterogeneous Tests (Future-Facing)

After single-constraint power is characterized, extend to heterogeneous scenarios.

### Design Sketch
- Simulate two primitive classes (or lineages) with different θ values.
- Fit:
  1. Single global θ model
  2. Per-class θ model
- Compare ΔLL and information criteria to see if heterogeneity is preferred.

### Questions to Answer
- Does the model detect heterogeneity when it exists?
- How much data is required to distinguish per-class effects?
- Are there multiple “clusters” of constraints analogous to branch-site models in phylogenetics?

---

## Execution Checklist
1. **Stage 1**: Feature-only null tests (legacy + symmetry breaks) → sign-off.
2. **Stage 2**: Deterministic null sanity checks → sign-off.
3. **V1**: Full null calibration (deterministic + stochastic) → stop on failure.
4. **V2**: Power grids across effect size / data volume / observation richness.
5. **V3**: Specificity sweeps.
6. **V**: Stratified heterogeneity experiments (optional once 1–5 are green).
7. **Automation plan**: keep quick PyTest coverage in `tests/plugins/assembly`, and place reusable scripts under `src/persiste/plugins/assembly/wip/legacy_validation/` (referenced above) so we can re-run heavy grids without polluting CI.

Maintain notebooks / scripts for each stage so we can rerun as the code evolves. Use consistent seeds and record raw simulation configs for reproducibility.
