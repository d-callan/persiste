"""Generate synthetic datasets with controlled baseline misspecification.

These scenarios are used by the baseline adequacy validation harness to ensure
Tier-1 safety checks properly escalate when the baseline is wrong.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

import persiste_rust

SCENARIOS = ("correct", "mild", "severe")

TRUE_BASELINE = {
    "kappa": 1.0,
    "join_exponent": -0.5,
    "split_exponent": 0.3,
    "decay_rate": 0.01,
}

TRUE_THETA = {
    "reuse_count": 0.4,
    "depth_change": -0.2,
    "size_change": 0.15,
    "symmetry_score": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline adequacy validation scenarios"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "src/persiste/plugins/assembly/validation/results/baseline_scenarios"
        ),
        help="Directory to write scenario JSON files",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=[*SCENARIOS, "all"],
        default="all",
        help="Which scenario to generate (default: all three)",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=200,
        help="Number of Gillespie trajectories per scenario",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum assembly depth for simulation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Base RNG seed; incremented per-scenario to keep outputs unique",
    )
    parser.add_argument(
        "--n-primitives",
        type=int,
        default=10,
        help="Number of primitive building blocks to synthesize",
    )
    parser.add_argument(
        "--initial-states",
        type=int,
        default=6,
        help="Number of initial states to seed trajectories with",
    )
    return parser.parse_args()


def generate_baseline_scenario(
    scenario: Literal["correct", "mild", "severe"],
    *,
    n_primitives: int,
    max_depth: int,
    n_trajectories: int,
    n_initial_states: int,
    seed: int,
) -> dict[str, Any]:
    """Generate a dataset dictionary for the requested baseline scenario."""

    rng = np.random.default_rng(seed)
    primitives = [chr(ord("A") + i) for i in range(n_primitives)]
    initial_states = sample_initial_states(rng, primitives, n_initial_states)

    theta_true = TRUE_THETA
    paths = run_simulation_batches(
        primitives=primitives,
        initial_states=initial_states,
        theta=theta_true,
        n_total=n_trajectories,
        max_depth=max_depth,
        rng=rng,
    )

    observation_records, observed_compounds, duration_summary = build_observations(
        paths, rng=rng, detection_prob=0.8, false_positive=0.03
    )

    inference_baseline = infer_baseline_for_scenario(scenario)
    error_magnitude = compute_baseline_error(TRUE_BASELINE, inference_baseline)

    dataset = {
        "scenario": scenario,
        "seed": seed,
        "true_baseline": TRUE_BASELINE,
        "inference_baseline": inference_baseline,
        "baseline_error_magnitude": error_magnitude,
        "true_theta": theta_true,
        "primitives": primitives,
        "observed_compounds": sorted(observed_compounds),
        "observation_records": observation_records,
        "duration_summary": duration_summary,
        "n_observed": len(observed_compounds),
        "n_trajectories": n_trajectories,
        "max_depth": max_depth,
        "n_primitives": n_primitives,
        "simulation": {
            "theta_features": list(theta_true.keys()),
            "initial_states": initial_states,
        },
    }
    return dataset


def sample_initial_states(
    rng: np.random.Generator,
    primitives: Sequence[str],
    n_states: int,
) -> list[list[str]]:
    states: list[list[str]] = []
    for _ in range(n_states):
        depth = int(rng.integers(1, 3))
        states.append(list(rng.choice(primitives, size=depth, replace=True)))
    return states


def run_simulation_batches(
    *,
    primitives: Sequence[str],
    initial_states: Sequence[Sequence[str]],
    theta: dict[str, float],
    n_total: int,
    max_depth: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    baseline = TRUE_BASELINE
    n_batches = len(initial_states)
    batch_size = int(np.ceil(n_total / n_batches))
    all_paths: list[dict[str, Any]] = []

    for batch_idx, initial_parts in enumerate(initial_states):
        remaining = n_total - len(all_paths)
        if remaining <= 0:
            break
        samples = batch_size if batch_idx < n_batches - 1 else remaining
        seed = int(rng.integers(0, 2**31 - 1))
        result = persiste_rust.simulate_assembly_trajectories(
            primitives=primitives,
            initial_parts=initial_parts,
            theta=theta,
            n_samples=samples,
            t_max=25.0,
            burn_in=5.0,
            max_depth=max_depth,
            seed=seed,
            kappa=baseline["kappa"],
            join_exponent=baseline["join_exponent"],
            split_exponent=baseline["split_exponent"],
            decay_rate=baseline["decay_rate"],
        )
        all_paths.extend(result)

    return all_paths


def build_observations(
    paths: Sequence[dict[str, Any]],
    *,
    rng: np.random.Generator,
    detection_prob: float,
    false_positive: float,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    counts = Counter(int(path["final_state_id"]) for path in paths)
    total = max(len(paths), 1)
    observed_compounds: set[str] = set()
    records: list[dict[str, Any]] = []
    duration_bins = [0.0, 5.0, 10.0, 20.0, float("inf")]
    n_duration_bins = len(duration_bins) - 1
    per_state_duration: dict[int, dict[str, Any]] = {}
    global_hist = [0] * n_duration_bins
    global_sum = 0.0
    global_sq_sum = 0.0
    global_depth_sum = 0.0
    global_depth_sq_sum = 0.0
    global_depth_hist: dict[int, int] = {}

    def get_bin_index(value: float) -> int:
        for idx in range(n_duration_bins):
            if duration_bins[idx] <= value < duration_bins[idx + 1]:
                return idx
        return n_duration_bins - 1

    for path in paths:
        state_id = int(path["final_state_id"])
        duration = float(path.get("duration", 0.0))
        stats = per_state_duration.setdefault(
            state_id,
            {
                "count": 0,
                "hist": [0] * n_duration_bins,
                "sum": 0.0,
                "sq_sum": 0.0,
                "depth_sum": 0.0,
                "depth_sq_sum": 0.0,
                "depth_max": 0,
            },
        )
        stats["count"] += 1
        stats["sum"] += duration
        stats["sq_sum"] += duration * duration
        stats["depth_sum"] += int(path.get("max_depth_reached", 0))
        stats["depth_sq_sum"] += int(path.get("max_depth_reached", 0)) ** 2
        stats["depth_max"] = max(stats["depth_max"], int(path.get("max_depth_reached", 0)))
        bin_idx = get_bin_index(duration)
        stats["hist"][bin_idx] += 1
        global_hist[bin_idx] += 1
        global_sum += duration
        global_sq_sum += duration * duration
        depth_val = int(path.get("max_depth_reached", 0))
        global_depth_sum += depth_val
        global_depth_sq_sum += depth_val * depth_val
        global_depth_hist[depth_val] = global_depth_hist.get(depth_val, 0) + 1

    for state_id, count in counts.items():
        freq = count / total
        detect_probability = min(1.0, freq * detection_prob + false_positive)
        detected = bool(rng.random() < detect_probability)
        duration_info = per_state_duration.get(
            state_id,
            {
                "count": 0,
                "hist": [0] * n_duration_bins,
                "sum": 0.0,
                "sq_sum": 0.0,
                "depth_sum": 0.0,
                "depth_sq_sum": 0.0,
                "depth_max": 0,
            },
        )
        duration_count = max(duration_info["count"], 1)
        mean_duration = duration_info["sum"] / duration_count
        variance_duration = (
            duration_info["sq_sum"] / duration_count - mean_duration**2
        )
        mean_depth = duration_info["depth_sum"] / duration_count
        variance_depth = (
            duration_info["depth_sq_sum"] / duration_count - mean_depth**2
        )
        record = {
            "state_id": state_id,
            "compound_id": f"state_{state_id}",
            "count": count,
            "presence_estimate": freq,
            "detect_probability": detect_probability,
            "detected": detected,
            "duration_histogram": duration_info["hist"],
            "mean_duration": mean_duration,
            "duration_variance": variance_duration,
            "mean_max_depth": mean_depth,
            "max_depth_variance": variance_depth,
            "max_depth_observed": duration_info["depth_max"],
        }
        records.append(record)
        if detected:
            observed_compounds.add(record["compound_id"])

    # Inject a few hard false positives to keep safety honest
    n_false = max(1, len(records) // 12)
    for _ in range(n_false):
        fake_id = int(rng.integers(50_000, 90_000))
        record = {
            "state_id": fake_id,
            "compound_id": f"state_{fake_id}",
            "count": 0,
            "presence_estimate": 0.0,
            "detect_probability": false_positive,
            "detected": bool(rng.random() < false_positive),
        }
        records.append(record)
        if record["detected"]:
            observed_compounds.add(record["compound_id"])

    global_count = max(len(paths), 1)
    global_mean = global_sum / global_count
    global_variance = global_sq_sum / global_count - global_mean**2
    depth_mean = global_depth_sum / global_count
    depth_variance = global_depth_sq_sum / global_count - depth_mean**2
    max_depth_bins = (
        list(range(max(global_depth_hist.keys(), default=0) + 1))
        if global_depth_hist
        else [0]
    )
    max_depth_hist = [global_depth_hist.get(bin_val, 0) for bin_val in max_depth_bins]
    duration_summary = {
        "bin_edges": duration_bins,
        "histogram": global_hist,
        "mean_duration": global_mean,
        "variance_duration": global_variance,
        "max_depth_bins": max_depth_bins,
        "max_depth_histogram": max_depth_hist,
        "mean_max_depth": depth_mean,
        "variance_max_depth": depth_variance,
    }

    return records, observed_compounds, duration_summary


def infer_baseline_for_scenario(
    scenario: Literal["correct", "mild", "severe"]
) -> dict[str, float]:
    perturbed = dict(TRUE_BASELINE)
    if scenario == "mild":
        perturbed["join_exponent"] += 0.2
        perturbed["decay_rate"] *= 1.2
    elif scenario == "severe":
        perturbed["join_exponent"] += 0.5
        perturbed["decay_rate"] *= 1.6
        perturbed["kappa"] *= 0.7
    return perturbed


def compute_baseline_error(
    true_baseline: dict[str, float], inference_baseline: dict[str, float]
) -> float:
    diffs = []
    for key, true_value in true_baseline.items():
        inf_value = inference_baseline.get(key, true_value)
        denom = abs(true_value) if abs(true_value) > 1e-6 else 1.0
        diffs.append(((inf_value - true_value) / denom) ** 2)
    return float(np.sqrt(np.mean(diffs)))


def write_dataset(dataset: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2)


def main() -> None:
    args = parse_args()
    scenarios = (
        SCENARIOS
        if args.scenario == "all"
        else (args.scenario,)  # type: ignore[arg-type]
    )

    for idx, scenario in enumerate(scenarios):
        seed = args.seed + idx
        dataset = generate_baseline_scenario(
            scenario,  # type: ignore[arg-type]
            n_primitives=args.n_primitives,
            max_depth=args.max_depth,
            n_trajectories=args.n_trajectories,
            n_initial_states=args.initial_states,
            seed=seed,
        )
        output_path = args.output_dir / f"{scenario}_seed{seed}.json"
        write_dataset(dataset, output_path)
        print(
            f"[baseline] scenario={scenario} seed={seed} "
            f"n_observed={dataset['n_observed']} -> {output_path}"
        )


if __name__ == "__main__":
    main()
