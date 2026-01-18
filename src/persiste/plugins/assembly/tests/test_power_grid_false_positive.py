"""Regression test to ensure null datasets do not produce spurious ΔLL."""

from __future__ import annotations

import json
from pathlib import Path

from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints

FEATURE_NAMES = ["reuse_count", "depth_change", "size_change", "symmetry_score"]
DATASET_PATH = Path(
    "src/persiste/plugins/assembly/validation/results/power_grid/"
    "pr6_depth5_traj100_frequency_null_seed10050.json"
)


def _load_dataset(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_inference(data: dict) -> dict:
    observed_compounds = set(data["observed_compounds"])
    primitives = data["primitives"]
    baseline_kwargs = data.get("inference_baseline", {})
    observation_records = data.get("observation_records")
    max_depth = data.get("config", {}).get("max_depth") or data.get("max_depth")

    return fit_assembly_constraints(
        observed_compounds=observed_compounds,
        primitives=primitives,
        mode=InferenceMode.SCREEN_AND_REFINE,
        feature_names=FEATURE_NAMES,
        seed=int(data.get("config", {}).get("seed", 0)),
        skip_safety_checks=False,
        observation_records=observation_records,
        max_depth=max_depth,
        use_screening=False,
        **baseline_kwargs,
    )


def _extract_delta_ll(result: dict) -> float:
    screening = result.get("screening_results") or []
    max_delta_ll = max((r.get("delta_ll", 0.0) for r in screening), default=0.0)
    stochastic_delta_ll = result.get("stochastic_delta_ll", 0.0)
    return max(max_delta_ll, stochastic_delta_ll)


def test_null_power_grid_dataset_has_low_delta_ll() -> None:
    """Null dataset should not exceed ΔLL safety threshold."""
    data = _load_dataset(DATASET_PATH)
    result = _run_inference(data)
    delta_ll = _extract_delta_ll(result)
    safety_report = result.get("safety") or {}
    threshold = float(safety_report.get("adjusted_delta_ll_threshold", 6.0))

    assert (
        delta_ll < threshold
    ), f"Null dataset produced ΔLL={delta_ll:.2f} >= threshold={threshold}"
