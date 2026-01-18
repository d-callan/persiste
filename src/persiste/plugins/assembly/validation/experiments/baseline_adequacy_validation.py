"""
Baseline adequacy validation harness.

Evaluates how Tier-1 safety checks respond as baseline misspecification increases
by running both safety-only and full inference on the generated scenario datasets.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints

FEATURE_NAMES = ["reuse_count", "depth_change", "size_change", "symmetry_score"]
BASE_DELTA_LL_THRESHOLD = 2.0


@dataclass
class ScenarioResult:
    scenario: str
    seed: int
    baseline_error_magnitude: float
    baseline_warning_level: str
    baseline_overall_status: str
    adjusted_delta_ll_threshold: float
    max_delta_ll: float
    max_normalized_delta_ll: float
    passes_base_threshold: bool
    passes_adjusted_threshold: bool
    suppressed_false_positive: bool
    safety_report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "baseline_error_magnitude": self.baseline_error_magnitude,
            "baseline_warning_level": self.baseline_warning_level,
            "baseline_overall_status": self.baseline_overall_status,
            "adjusted_delta_ll_threshold": self.adjusted_delta_ll_threshold,
            "max_delta_ll": self.max_delta_ll,
            "max_normalized_delta_ll": self.max_normalized_delta_ll,
            "passes_base_threshold": self.passes_base_threshold,
            "passes_adjusted_threshold": self.passes_adjusted_threshold,
            "suppressed_false_positive": self.suppressed_false_positive,
            "safety_report": self.safety_report,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline adequacy validation across scenario datasets."
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=Path(
            "src/persiste/plugins/assembly/validation/results/baseline_scenarios"
        ),
        help="Directory containing scenario JSON files (default: validation results dir)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "src/persiste/plugins/assembly/validation/results/"
            "baseline_adequacy_results.json"
        ),
        help="Path to write aggregated validation results JSON.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["screen-and-refine", "full-stochastic"],
        default="screen-and-refine",
        help="Inference mode for full run (default: screen-and-refine).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of scenario files to evaluate.",
    )
    return parser.parse_args()


def load_scenario(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_scenario(
    data: dict[str, Any], *, inference_mode: InferenceMode
) -> ScenarioResult:
    observed_compounds = set(data["observed_compounds"])
    primitives = data["primitives"]
    baseline_kwargs = data.get("inference_baseline", {})

    # Run safety-only mode to capture baseline severity ordering quickly
    safety_only = fit_assembly_constraints(
        observed_compounds=observed_compounds,
        primitives=primitives,
        mode=InferenceMode.SAFETY_ONLY,
        feature_names=FEATURE_NAMES,
        **baseline_kwargs,
    )
    safety_report_safety_only = safety_only.get("safety", {})

    # Run full inference (screen+refine) with same baseline configuration
    full_result = fit_assembly_constraints(
        observed_compounds=observed_compounds,
        primitives=primitives,
        mode=inference_mode,
        feature_names=FEATURE_NAMES,
        **baseline_kwargs,
    )
    safety_report = full_result.get("safety", safety_report_safety_only)
    baseline_check = safety_report.get("baseline_check", {})

    screening_results = full_result.get("screening_results", [])
    if screening_results:
        max_delta_ll = max(r.get("delta_ll", 0.0) for r in screening_results)
        max_norm_delta_ll = max(
            r.get("normalized_delta_ll", 0.0) for r in screening_results
        )
    else:
        max_delta_ll = 0.0
        max_norm_delta_ll = 0.0

    adjusted_threshold = float(
        safety_report.get("adjusted_delta_ll_threshold", BASE_DELTA_LL_THRESHOLD)
    )
    passes_base = max_delta_ll >= BASE_DELTA_LL_THRESHOLD
    passes_adjusted = max_delta_ll >= adjusted_threshold
    suppressed_false_positive = passes_base and not passes_adjusted

    return ScenarioResult(
        scenario=data["scenario"],
        seed=data["seed"],
        baseline_error_magnitude=float(data["baseline_error_magnitude"]),
        baseline_warning_level=baseline_check.get("warning_level", "none"),
        baseline_overall_status=safety_report.get("overall_status", "ok"),
        adjusted_delta_ll_threshold=adjusted_threshold,
        max_delta_ll=max_delta_ll,
        max_normalized_delta_ll=max_norm_delta_ll,
        passes_base_threshold=passes_base,
        passes_adjusted_threshold=passes_adjusted,
        suppressed_false_positive=suppressed_false_positive,
        safety_report=safety_report,
    )


def discover_scenarios(directory: Path, limit: int | None) -> list[Path]:
    paths = sorted(directory.glob("*.json"))
    if limit is not None:
        paths = paths[:limit]
    return paths


def main() -> None:
    args = parse_args()
    scenario_paths = discover_scenarios(args.scenario_dir, args.limit)

    inference_mode = (
        InferenceMode.SCREEN_AND_REFINE
        if args.mode == "screen-and-refine"
        else InferenceMode.FULL_STOCHASTIC
    )

    results: list[dict[str, Any]] = []
    for path in scenario_paths:
        data = load_scenario(path)
        result = evaluate_scenario(data, inference_mode=inference_mode)
        payload = result.to_dict()
        payload["scenario_path"] = str(path)
        results.append(payload)
        print(
            f"[baseline-adequacy] scenario={result.scenario} "
            f"warning={result.baseline_warning_level} "
            f"maxΔLL={result.max_delta_ll:.2f} "
            f"adjusted={result.adjusted_delta_ll_threshold:.2f} "
            f"passes_adjusted={result.passes_adjusted_threshold}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh, indent=2)
    print(f"Wrote baseline adequacy summary to {args.output}")


if __name__ == "__main__":
    main()
