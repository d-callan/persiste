"""Generate dataset with Symmetry Break A: depth-gated reuse modifier.

This script creates a controlled test of non-stationarity by enabling
depth-dependent reuse rates. The key identifiability prediction:

    ΔLL > 0 if and only if observations condition on depth.

Design:
- D* = 4 (depth threshold)
- θ_depth = 0.3 (small, realistic modifier)
- Reuse rate multiplier = exp(0.3) ≈ 1.35 when max_depth >= 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import persiste_rust

# Symmetry Break A parameters
DEPTH_GATE_THRESHOLD = 4
DEPTH_GATE_THETA = 0.3  # Small, realistic value

# True baseline (same as before)
TRUE_BASELINE = {
    "kappa": 1.0,
    "join_exponent": -0.5,
    "split_exponent": 0.3,
    "decay_rate": 0.01,
}

# Signal theta (same as targeted runs)
SIGNAL_THETA = {
    "reuse_count": 0.5,
    "depth_change": -0.3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate depth-gated dataset (Symmetry Break A)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "src/persiste/plugins/assembly/validation/results/"
            "depth_gated_pr8_depth7_traj200_signal.json"
        ),
        help="Output path for dataset.",
    )
    parser.add_argument(
        "--n-primitives",
        type=int,
        default=8,
        help="Number of primitives.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=7,
        help="Maximum assembly depth.",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=200,
        help="Number of trajectories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20000,
        help="RNG seed.",
    )
    return parser.parse_args()


def run_simulation(
    primitives: list[str],
    initial_states: list[list[str]],
    theta: dict[str, float],
    n_total: int,
    max_depth: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Run simulation with depth-gated reuse enabled."""
    all_paths = []
    batch_size = 50
    n_batches = (n_total + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        n_samples = min(batch_size, n_total - len(all_paths))
        initial_parts = rng.choice(initial_states, size=1)[0]
        seed = int(rng.integers(0, 2**32))

        result = persiste_rust.simulate_assembly_trajectories(
            primitives=primitives,
            initial_parts=initial_parts,
            theta=theta,
            n_samples=n_samples,
            t_max=30.0,
            burn_in=10.0,
            max_depth=max_depth,
            seed=seed,
            kappa=TRUE_BASELINE["kappa"],
            join_exponent=TRUE_BASELINE["join_exponent"],
            split_exponent=TRUE_BASELINE["split_exponent"],
            decay_rate=TRUE_BASELINE["decay_rate"],
            depth_gate_threshold=DEPTH_GATE_THRESHOLD,
            depth_gate_theta=DEPTH_GATE_THETA,
        )
        all_paths.extend(result)

    return all_paths


def build_observations(
    paths: list[dict[str, Any]], *, rng: np.random.Generator
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    """Build observations with depth conditioning."""
    from collections import Counter

    duration_bins = [0.0, 3.0, 6.0, 10.0, float("inf")]
    n_bins = len(duration_bins) - 1

    def bin_index(value: float) -> int:
        for idx in range(n_bins):
            if duration_bins[idx] <= value < duration_bins[idx + 1]:
                return idx
        return n_bins - 1

    counts = Counter(int(path["final_state_id"]) for path in paths)
    total = max(len(paths), 1)
    observed_compounds: set[str] = set()
    records: list[dict[str, Any]] = []

    per_state_stats: dict[int, dict[str, Any]] = {}
    global_hist = [0] * n_bins
    global_sum = 0.0
    global_sq_sum = 0.0
    global_depth_hist: dict[int, int] = {}
    global_depth_sum = 0.0
    global_depth_sq_sum = 0.0
    global_reuse_sum = 0
    global_reuse_sq_sum = 0

    for path in paths:
        state_id = int(path["final_state_id"])
        duration = float(path.get("duration", 0.0))
        max_depth = int(path.get("max_depth_reached", 0))
        reuse_count = int(path.get("reuse_count", 0))
        stats = per_state_stats.setdefault(
            state_id,
            {
                "count": 0,
                "hist": [0] * n_bins,
                "sum": 0.0,
                "sq_sum": 0.0,
                "depth_sum": 0.0,
                "depth_sq_sum": 0.0,
                "depth_max": 0,
                "reuse_sum": 0,
                "reuse_sq_sum": 0,
            },
        )
        stats["count"] += 1
        stats["sum"] += duration
        stats["sq_sum"] += duration * duration
        stats["depth_sum"] += max_depth
        stats["depth_sq_sum"] += max_depth * max_depth
        stats["depth_max"] = max(stats["depth_max"], max_depth)
        stats["reuse_sum"] += reuse_count
        stats["reuse_sq_sum"] += reuse_count * reuse_count
        idx = bin_index(duration)
        stats["hist"][idx] += 1
        global_hist[idx] += 1
        global_sum += duration
        global_sq_sum += duration * duration
        global_depth_sum += max_depth
        global_depth_sq_sum += max_depth * max_depth
        global_depth_hist[max_depth] = global_depth_hist.get(max_depth, 0) + 1
        global_reuse_sum += reuse_count
        global_reuse_sq_sum += reuse_count * reuse_count

    for state_id, count in counts.items():
        freq = count / total
        detect_p = min(1.0, 0.1 + freq * 0.8)
        detected = bool(rng.random() < detect_p)
        duration_info = per_state_stats.get(
            state_id,
            {
                "count": 0,
                "hist": [0] * n_bins,
                "sum": 0.0,
                "sq_sum": 0.0,
                "depth_sum": 0.0,
                "depth_sq_sum": 0.0,
                "depth_max": 0,
                "reuse_sum": 0,
                "reuse_sq_sum": 0,
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
        mean_reuse = duration_info["reuse_sum"] / duration_count
        variance_reuse = (
            duration_info["reuse_sq_sum"] / duration_count - mean_reuse**2
        )

        # Frequency observation (Poisson)
        lam = max(freq * len(paths), 0.5)
        frequency = int(rng.poisson(lam))
        if frequency > 0:
            detected = True

        record = {
            "state_id": state_id,
            "compound_id": f"state_{state_id}",
            "presence_estimate": freq,
            "detect_probability": detect_p,
            "detected": detected,
            "frequency": frequency,
            "duration_histogram": duration_info["hist"],
            "mean_duration": mean_duration,
            "duration_variance": variance_duration,
            "mean_max_depth": mean_depth,
            "max_depth_variance": variance_depth,
            "max_depth_observed": duration_info["depth_max"],
            "mean_reuse_count": mean_reuse,
            "reuse_count_variance": variance_reuse,
        }
        records.append(record)
        if detected:
            observed_compounds.add(record["compound_id"])

    global_count = max(len(paths), 1)
    global_mean = global_sum / global_count
    global_variance = global_sq_sum / global_count - global_mean**2
    depth_mean = global_depth_sum / global_count
    depth_variance = global_depth_sq_sum / global_count - depth_mean**2
    reuse_mean = global_reuse_sum / max(len(paths), 1)
    reuse_variance = global_reuse_sq_sum / max(len(paths), 1) - reuse_mean**2
    max_depth_bins = (
        list(range(max(global_depth_hist.keys(), default=0) + 1))
        if global_depth_hist
        else [0]
    )
    max_depth_hist = [global_depth_hist.get(b, 0) for b in max_depth_bins]

    duration_summary = {
        "bin_edges": duration_bins,
        "histogram": global_hist,
        "mean_duration": global_mean,
        "variance_duration": global_variance,
        "max_depth_bins": max_depth_bins,
        "max_depth_histogram": max_depth_hist,
        "mean_max_depth": depth_mean,
        "variance_max_depth": depth_variance,
        "mean_reuse_count": reuse_mean,
        "variance_reuse_count": reuse_variance,
    }

    return records, observed_compounds, duration_summary


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    primitives = [chr(ord("A") + i) for i in range(args.n_primitives)]
    initial_states = [[p] for p in primitives[:2]]

    print("Generating depth-gated dataset with Symmetry Break A:")
    print(f"  D* = {DEPTH_GATE_THRESHOLD}, θ_depth = {DEPTH_GATE_THETA}")
    print(f"  Primitives: {args.n_primitives}, Depth: {args.max_depth}")
    print(f"  Trajectories: {args.n_trajectories}")

    paths = run_simulation(
        primitives=primitives,
        initial_states=initial_states,
        theta=SIGNAL_THETA,
        n_total=args.n_trajectories,
        max_depth=args.max_depth,
        rng=rng,
    )

    observation_records, observed_compounds, duration_summary = build_observations(
        paths, rng=rng
    )

    dataset = {
        "config": {
            "n_primitives": args.n_primitives,
            "max_depth": args.max_depth,
            "n_trajectories": args.n_trajectories,
            "obs_type": "frequency",
            "theta_mode": "signal",
            "seed": args.seed,
            "symmetry_break": "depth_gated_reuse",
            "depth_gate_threshold": DEPTH_GATE_THRESHOLD,
            "depth_gate_theta": DEPTH_GATE_THETA,
        },
        "primitives": primitives,
        "initial_states": initial_states,
        "observed_compounds": sorted(observed_compounds),
        "observation_records": observation_records,
        "duration_summary": duration_summary,
        "true_theta": SIGNAL_THETA,
        "theta_mode": "signal",
        "true_baseline": TRUE_BASELINE,
        "inference_baseline": TRUE_BASELINE,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2)

    print(f"\nWrote dataset to {args.output}")
    print(f"  Observed compounds: {len(observed_compounds)}")
    print(f"  Mean max depth: {duration_summary['mean_max_depth']:.2f}")
    print(f"  Mean reuse count: {duration_summary['mean_reuse_count']:.1f}")


if __name__ == "__main__":
    main()
