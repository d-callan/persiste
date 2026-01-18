"""
Power/identifiability sweep across the generated dataset grid.

For each dataset:
1. Run multiple inference replicates with different seeds.
2. Record ΔLL statistics and whether the run recovers the injected signal.
3. Aggregate summary metrics for downstream visualization.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from statistics import mean
from typing import Any

from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints

FEATURE_NAMES = ["reuse_count", "depth_change", "size_change", "symmetry_score"]
BASE_DELTA_LL_THRESHOLD = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run power/identifiability sweep over power grid datasets."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("src/persiste/plugins/assembly/validation/results/power_grid"),
        help="Directory containing power grid datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "src/persiste/plugins/assembly/validation/results/"
            "power_envelope_results.json"
        ),
        help="Path to write aggregated sweep results.",
    )
    parser.add_argument(
        "--n-replicates",
        type=int,
        default=5,
        help="Number of inference replicates per dataset (default: 5).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["screen-and-refine", "full-stochastic"],
        default="screen-and-refine",
        help="Inference mode for sweep.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of datasets (for smoke testing).",
    )
    parser.add_argument(
        "--skip-safety-checks",
        action="store_true",
        help="Skip Tier 1 safety checks during inference.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def run_replicate(
    data: dict[str, Any],
    *,
    replicate_seed: int,
    inference_mode: InferenceMode,
    skip_safety_checks: bool,
) -> tuple[float, float, float]:
    observed_compounds = set(data["observed_compounds"])
    primitives = data["primitives"]
    baseline_kwargs = data.get("inference_baseline", {})
    observation_records = data.get("observation_records")

    max_depth = data.get("config", {}).get("max_depth")
    if max_depth is None:
        max_depth = data.get("max_depth")

    result = fit_assembly_constraints(
        observed_compounds=observed_compounds,
        primitives=primitives,
        mode=inference_mode,
        feature_names=FEATURE_NAMES,
        seed=replicate_seed,
        skip_safety_checks=skip_safety_checks,
        observation_records=observation_records,
        max_depth=max_depth,
        **baseline_kwargs,
    )

    screening = result.get("screening_results", [])
    max_delta_ll = 0.0
    max_norm_delta_ll = 0.0
    if screening:
        max_delta_ll = max(r.get("delta_ll", 0.0) for r in screening)
        max_norm_delta_ll = max(r.get("normalized_delta_ll", 0.0) for r in screening)

    # Also check stochastic ΔLL if available
    stochastic_delta_ll = result.get("stochastic_delta_ll", 0.0)
    if stochastic_delta_ll > max_delta_ll:
        max_delta_ll = stochastic_delta_ll
        max_norm_delta_ll = stochastic_delta_ll  # Use raw ΔLL as normalized for now

    safety_report = result.get("safety") or {}
    adjusted_threshold = float(
        safety_report.get("adjusted_delta_ll_threshold", BASE_DELTA_LL_THRESHOLD)
    )

    return max_delta_ll, max_norm_delta_ll, adjusted_threshold


def evaluate_dataset(
    path: Path,
    *,
    inference_mode: InferenceMode,
    n_replicates: int,
    skip_safety_checks: bool,
) -> dict[str, Any]:
    data = load_dataset(path)
    config = data.get("config", {})
    theta_mode = data.get("theta_mode", "null")

    replicate_results = []
    base_seed = int(config.get("seed", 0))

    for rep in range(n_replicates):
        replicate_seed = base_seed + rep
        try:
            max_delta_ll, max_norm_delta_ll, adjusted_threshold = run_replicate(
                data,
                replicate_seed=replicate_seed,
                inference_mode=inference_mode,
                skip_safety_checks=skip_safety_checks,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            replicate_results.append(
                {
                    "success": False,
                    "error": str(exc),
                    "max_delta_ll": 0.0,
                    "max_normalized_delta_ll": 0.0,
                    "adjusted_threshold": BASE_DELTA_LL_THRESHOLD,
                }
            )
            continue

        passes_threshold = max_delta_ll >= adjusted_threshold
        replicate_results.append(
            {
                "success": True,
                "max_delta_ll": max_delta_ll,
                "max_normalized_delta_ll": max_norm_delta_ll,
                "adjusted_threshold": adjusted_threshold,
                "passes_threshold": passes_threshold,
            }
        )

    successes = [r for r in replicate_results if r["success"]]
    if successes:
        recovery_rate = sum(r["passes_threshold"] for r in successes) / len(successes)
        avg_delta_ll = mean(r["max_delta_ll"] for r in successes)
    else:
        recovery_rate = 0.0
        avg_delta_ll = 0.0

    false_positive_rate = 0.0
    true_positive_rate = 0.0
    if theta_mode == "signal":
        true_positive_rate = recovery_rate
    else:
        false_positive_rate = recovery_rate

    return {
        "dataset": path.name,
        "config": config,
        "theta_mode": theta_mode,
        "n_replicates": len(replicate_results),
        "n_successful_replicates": len(successes),
        "recovery_rate": recovery_rate,
        "avg_delta_ll": avg_delta_ll,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
        "replicates": replicate_results,
    }


def main() -> None:
    args = parse_args()
    inference_mode = (
        InferenceMode.SCREEN_AND_REFINE
        if args.mode == "screen-and-refine"
        else InferenceMode.FULL_STOCHASTIC
    )

    dataset_paths = sorted(args.dataset_dir.glob("*.json"))
    if args.limit is not None:
        dataset_paths = dataset_paths[: args.limit]

    summaries: list[dict[str, Any]] = []
    for idx, path in enumerate(dataset_paths, start=1):
        summary = evaluate_dataset(
            path,
            inference_mode=inference_mode,
            n_replicates=args.n_replicates,
            skip_safety_checks=args.skip_safety_checks,
        )
        summaries.append(summary)
        print(
            f"[power-sweep {idx}/{len(dataset_paths)}] {path.name} "
            f"θ_mode={summary['theta_mode']} "
            f"recovery={summary['recovery_rate']:.2f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"results": summaries}, fh, indent=2)
    print(f"Wrote power sweep summary to {args.output}")


if __name__ == "__main__":
    main()
