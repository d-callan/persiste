"""Run assembly robustness benchmark on generated dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints

DEFAULT_FEATURES = [
    "reuse_count",
    "depth_change",
    "size_change",
    "symmetry_score",
    "diversity_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run assembly benchmark inference")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/assembly_benchmark"),
        help="Directory containing primitives.json, observations.json, config.json",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["screen-only", "screen-and-refine", "full-stochastic"],
        default="screen-and-refine",
        help="Inference mode to run (default: screen-and-refine)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=300,
        help="Number of stochastic samples for refinement (default: 300)",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=25.0,
        help="Maximum simulation time (default: 25.0)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum assembly depth (default: 6)",
    )
    parser.add_argument(
        "--screen-budget",
        type=int,
        default=100,
        help="Screening budget if using adaptive screening (default: 100)",
    )
    parser.add_argument(
        "--screen-topk",
        type=int,
        default=10,
        help="Top-K refinements for screening (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="RNG seed (default: 2026)",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_dataset(dataset_dir: Path) -> dict[str, Any]:
    config = load_json(dataset_dir / "config.json")
    primitives_doc = load_json(dataset_dir / config["primitives"])
    observations_doc = load_json(dataset_dir / config["observations"])

    primitives = [p["name"] for p in primitives_doc["primitives"]]

    detected_compounds = config.get("detected_compounds")
    if not detected_compounds:
        detected_compounds = [
            record.get("compound_id", f"state_{record['state_id']}")
            for record in observations_doc.get("records", [])
            if record.get("detected")
        ]

    return {
        "config": config,
        "primitives": primitives,
        "observed_compounds": set(detected_compounds),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(args.dataset_dir)
    config = dataset["config"]

    mode_key = args.mode.replace("-", "_").upper()
    mode = InferenceMode[mode_key]

    observed_compounds = dataset["observed_compounds"]
    primitives = dataset["primitives"]

    screen_budget = args.screen_budget or config.get("screening", {}).get("budget", 100)
    screen_topk = args.screen_topk or config.get("screening", {}).get("top_k", 10)

    start = time.perf_counter()
    result = fit_assembly_constraints(
        observed_compounds=observed_compounds,
        primitives=primitives,
        mode=mode,
        feature_names=DEFAULT_FEATURES,
        screen_budget=screen_budget,
        screen_topk=screen_topk,
        screen_refine_radius=config.get("screening", {}).get("refine_radius", 0.5),
        n_samples=args.n_samples,
        t_max=args.t_max,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    duration = time.perf_counter() - start

    return {
        "result": result,
        "duration_sec": duration,
        "n_observed": len(observed_compounds),
        "n_primitives": len(primitives),
    }


def main() -> None:
    args = parse_args()
    benchmark = run_benchmark(args)

    print("=== Assembly Benchmark ===")
    print(f"Mode: {args.mode}")
    print(f"Observed compounds: {benchmark['n_observed']}")
    print(f"Primitives: {benchmark['n_primitives']}")
    print(f"Runtime: {benchmark['duration_sec']:.2f}s")
    print(f"Estimated θ: {benchmark['result']['theta_hat']}")
    if benchmark["result"].get("cache_stats"):
        print(f"Cache stats: {benchmark['result']['cache_stats']}")
    print("Screening results (top 3):")
    for row in benchmark["result"].get("screening_results", [])[:3]:
        print(
            f"  rank={row['rank']:>2} passed={row['passed']} "
            f"ΔLL={row['delta_ll']:.2f} norm={row['normalized_delta_ll']:.2f} θ={row['theta']}"
        )


if __name__ == "__main__":
    main()
