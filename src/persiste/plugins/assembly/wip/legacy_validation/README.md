# Assembly Constraint Validation Scripts

This directory contains reusable validation scripts corresponding to the stages and phases outlined in `docs/ASSEMBLY_VALIDATION_PLAN.md`.

## Quick Start

### Stage I: Feature-Level Null Tests
```bash
python src/persiste/plugins/assembly/wip/legacy_validation/run_feature_nulls.py --n-seeds 5 --n-samples 50 --output src/persiste/plugins/assembly/wip/legacy_validation/results/feature_nulls.json
```
Validates that individual features have mean ≈ 0 and stable variance under the null model (θ = 0).

### V1: Null Calibration Sweep
```bash
python src/persiste/plugins/assembly/wip/legacy_validation/run_null_calibration.py --n-seeds 3 --output src/persiste/plugins/assembly/wip/legacy_validation/results/null_calib.csv
```
Validates that ΔLL ≈ 0 across deterministic and stochastic inference modes when θ = 0.

## Scripts

| Script | Stage | Purpose | Output |
|--------|-------|---------|--------|
| `run_feature_nulls.py` | I | Feature-level null validation | JSON summary of feature statistics |
| `run_null_calibration.py` | V1 | Null calibration across modes | CSV of ΔLL per seed/mode |
| `run_deterministic_nulls.py` | II | Deterministic screen sanity checks | CSV of screening results on null data |
| `run_power_grid.py` | V2 | Power grid sweep (effect size × data volume × observation richness) | CSV of recovery rates, bias, ESS |
| `run_specificity_tests.py` | V3 | Constraint specificity validation | CSV of ΔLL per constraint when only that constraint is active |
| `run_stratified_heterogeneity.py` | V | Stratified/heterogeneous constraint tests | CSV of model comparison metrics |

## Integration with CI

- **Quick checks** (Stage I, II, V1 null calibration): Integrated into `tests/plugins/assembly/test_robustness_reboot.py` as `TestNullCalibration` and `TestDeterministicSanity`.
- **Heavier sweeps** (V2, V3, V): Run these scripts ad-hoc or in a separate validation workflow. Results are stored in CSV/JSON for analysis.

## Example Workflow

1. Run quick pytest checks:
   ```bash
   pytest tests/plugins/assembly/test_robustness_reboot.py::TestNullCalibration -v
   pytest tests/plugins/assembly/test_robustness_reboot.py::TestDeterministicSanity -v
   ```

2. If quick checks pass, run feature-level nulls:
   ```bash
   python src/persiste/plugins/assembly/wip/legacy_validation/run_feature_nulls.py --n-seeds 5 --output src/persiste/plugins/assembly/wip/legacy_validation/results/feature_nulls.json
   ```

3. If feature nulls pass, run null calibration sweep:
   ```bash
   python src/persiste/plugins/assembly/wip/legacy_validation/run_null_calibration.py --n-seeds 10 --output src/persiste/plugins/assembly/wip/legacy_validation/results/null_calib.csv
   ```

4. Once null calibration is confirmed, run power grids:
   ```bash
   python src/persiste/plugins/assembly/wip/legacy_validation/run_power_grid.py --output src/persiste/plugins/assembly/wip/legacy_validation/results/power_grid.csv
   ```

## Output Format

All scripts output results in CSV or JSON format for easy analysis. Use pandas or your favorite tool to load and visualize:

```python
import pandas as pd
df = pd.read_csv("results/null_calib.csv")
print(df.groupby("mode")[["deterministic_delta_ll", "stochastic_delta_ll"]].describe())
```
