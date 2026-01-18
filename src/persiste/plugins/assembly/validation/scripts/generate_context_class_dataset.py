"""Generate dataset with Symmetry Break B: context-dependent reuse classes.

This script enables class-aware reuse modifiers in the simulator while keeping
the inference layer unchanged. It is intended for power-envelope and
identifiability studies that isolate the effect of class-conditioned reuse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import persiste_rust

# Baseline physics (same as other benchmarks)
TRUE_BASELINE = {
    "kappa": 1.0,
    "join_exponent": -0.5,
    "split_exponent": 0.3,
    "decay_rate": 0.01,
}

# Signal θ reused from other datasets (still stationary at constraint layer)
SIGNAL_THETA = {
    "reuse_count": 0.5,
    "depth_change": -0.3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate context-class dataset (Symmetry Break B)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "src/persiste/plugins/assembly/validation/results/"
            "context_class_pr8_depth7_traj200_signal.json"
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
        default=21000,
        help="RNG seed.",
    )
    parser.add_argument(
        "--same-class-theta",
        type=float,
        default=0.35,
        help="Log-scale reuse bonus when source/target share the same class.",
    )
    parser.add_argument(
        "--cross-class-theta",
        type=float,
        default=-0.25,
        help="Log-scale reuse penalty when classes differ.",
    )
    return parser.parse_args()


def split_classes(primitives: list[str]) -> dict[str, str]:
    """Assign primitives to two latent classes (founder / derived)."""
    midpoint = max(len(primitives) // 2, 1)
    class_map: dict[str, str] = {}
    for idx, primitive in enumerate(primitives):
        class_map[primitive] = "founder" if idx < midpoint else "derived"
    return class_map


def run_simulation(
    *,
    primitives: list[str],
    initial_states: list[list[str]],
    theta: dict[str, float],
    n_total: int,
    max_depth: int,
    rng: np.random.Generator,
    context_class_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run simulation with context-class reuse enabled."""
    all_paths: list[dict[str, Any]] = []
    batch_size = 50
    n_batches = (n_total + batch_size - 1) // batch_size

    for _ in range(n_batches):
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
            context_class_config=context_class_config,
        )
        all_paths.extend(result)

    return all_paths


def build_observations(
    paths: list[dict[str, Any]], *, rng: np.random.Generator
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    """Build observations with class-conditioned reuse statistics."""
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
    global_reuse_sum = 0.0
    global_reuse_sq_sum = 0.0
    global_same_sum = 0.0
    global_same_sq_sum = 0.0
    global_cross_sum = 0.0
    global_cross_sq_sum = 0.0

    for path in paths:
        state_id = int(path["final_state_id"])
        duration = float(path.get("duration", 0.0))
        reuse_count = float(path.get("reuse_count", 0))
        feature_counts = path.get("feature_counts", {}) or {}
        same_reuse = float(feature_counts.get("reuse_same_class", 0))
        cross_reuse = float(feature_counts.get("reuse_cross_class", 0))

        stats = per_state_stats.setdefault(
            state_id,
            {
                "count": 0,
                "hist": [0] * n_bins,
                "sum": 0.0,
                "sq_sum": 0.0,
                "reuse_sum": 0.0,
                "reuse_sq_sum": 0.0,
                "same_sum": 0.0,
                "same_sq_sum": 0.0,
                "cross_sum": 0.0,
                "cross_sq_sum": 0.0,
            },
        )
        stats["count"] += 1
        stats["sum"] += duration
        stats["sq_sum"] += duration * duration
        stats["reuse_sum"] += reuse_count
        stats["reuse_sq_sum"] += reuse_count * reuse_count
        stats["same_sum"] += same_reuse
        stats["same_sq_sum"] += same_reuse * same_reuse
        stats["cross_sum"] += cross_reuse
        stats["cross_sq_sum"] += cross_reuse * cross_reuse
        idx = bin_index(duration)
        stats["hist"][idx] += 1

        global_hist[idx] += 1
        global_sum += duration
        global_sq_sum += duration * duration
        global_reuse_sum += reuse_count
        global_reuse_sq_sum += reuse_count * reuse_count
        global_same_sum += same_reuse
        global_same_sq_sum += same_reuse * same_reuse
        global_cross_sum += cross_reuse
        global_cross_sq_sum += cross_reuse * cross_reuse

    for state_id, count in counts.items():
        freq = count / total
        detect_p = min(1.0, 0.1 + freq * 0.8)
        detected = bool(rng.random() < detect_p)
        stats = per_state_stats.get(
            state_id,
            {
                "count": 0,
                "hist": [0] * n_bins,
                "sum": 0.0,
                "sq_sum": 0.0,
                "reuse_sum": 0.0,
                "reuse_sq_sum": 0.0,
                "same_sum": 0.0,
                "same_sq_sum": 0.0,
                "cross_sum": 0.0,
                "cross_sq_sum": 0.0,
            },
        )
        duration_count = max(stats["count"], 1)
        mean_duration = stats["sum"] / duration_count
        variance_duration = stats["sq_sum"] / duration_count - mean_duration**2
        mean_reuse = stats["reuse_sum"] / duration_count
        variance_reuse = stats["reuse_sq_sum"] / duration_count - mean_reuse**2
        mean_same = stats["same_sum"] / duration_count
        variance_same = stats["same_sq_sum"] / duration_count - mean_same**2
        mean_cross = stats["cross_sum"] / duration_count
        variance_cross = stats["cross_sq_sum"] / duration_count - mean_cross**2

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
            "duration_histogram": stats["hist"],
            "mean_duration": mean_duration,
            "duration_variance": variance_duration,
            "mean_reuse_count": mean_reuse,
            "reuse_count_variance": variance_reuse,
            "mean_same_class_reuse": mean_same,
            "same_class_reuse_variance": variance_same,
            "mean_cross_class_reuse": mean_cross,
            "cross_class_reuse_variance": variance_cross,
        }
        records.append(record)
        if detected:
            observed_compounds.add(record["compound_id"])

    global_count = max(len(paths), 1)
    global_mean = global_sum / global_count
    global_variance = global_sq_sum / global_count - global_mean**2
    reuse_mean = global_reuse_sum / global_count
    reuse_variance = global_reuse_sq_sum / global_count - reuse_mean**2
    same_mean = global_same_sum / global_count
    same_variance = global_same_sq_sum / global_count - same_mean**2
    cross_mean = global_cross_sum / global_count
    cross_variance = global_cross_sq_sum / global_count - cross_mean**2

    duration_summary = {
        "bin_edges": duration_bins,
        "histogram": global_hist,
        "mean_duration": global_mean,
        "variance_duration": global_variance,
        "mean_reuse_count": reuse_mean,
        "variance_reuse_count": reuse_variance,
        "mean_same_class_reuse": same_mean,
        "variance_same_class_reuse": same_variance,
        "mean_cross_class_reuse": cross_mean,
        "variance_cross_class_reuse": cross_variance,
    }

    return records, observed_compounds, duration_summary


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    primitives = [chr(ord("A") + i) for i in range(args.n_primitives)]
    initial_states = [[p] for p in primitives[:2]]
    primitive_classes = split_classes(primitives)
    context_class_config = {
        "primitive_classes": primitive_classes,
        "same_class_theta": args.same_class_theta,
        "cross_class_theta": args.cross_class_theta,
    }

    print("Generating context-class dataset (Symmetry Break B):")
    print(f"  same_class_theta = {args.same_class_theta}")
    print(f"  cross_class_theta = {args.cross_class_theta}")
    print(f"  Primitives: {args.n_primitives}, Depth: {args.max_depth}")
    print(f"  Trajectories: {args.n_trajectories}")

    paths = run_simulation(
        primitives=primitives,
        initial_states=initial_states,
        theta=SIGNAL_THETA,
        n_total=args.n_trajectories,
        max_depth=args.max_depth,
        rng=rng,
        context_class_config=context_class_config,
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
            "symmetry_break": "context_class_reuse",
            "context_class_config": context_class_config,
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
    print(f"  Mean reuse count: {duration_summary['mean_reuse_count']:.2f}")
    print(
        f"  Mean same-class reuse: {duration_summary['mean_same_class_reuse']:.2f}"
    )
    print(
        f"  Mean cross-class reuse: {duration_summary['mean_cross_class_reuse']:.2f}"
    )


if __name__ == "__main__":
    main()
