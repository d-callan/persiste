"""Generate dataset with Symmetry Break C: founder bias / exchangeability break.

This generator enables rank-aware reuse bonuses (founder preference) inside the
simulator so we can probe identifiability without changing the inference layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import persiste_rust

# Shared baseline config
TRUE_BASELINE = {
    "kappa": 1.0,
    "join_exponent": -0.5,
    "split_exponent": 0.3,
    "decay_rate": 0.01,
}

# Keep constraint θ identical to other targeted datasets
SIGNAL_THETA = {
    "reuse_count": 0.5,
    "depth_change": -0.3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate founder-bias dataset (Symmetry Break C)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "src/persiste/plugins/assembly/validation/results/"
            "founder_bias_pr8_depth7_traj200_signal.json"
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
        default=22000,
        help="RNG seed.",
    )
    parser.add_argument(
        "--founder-rank-threshold",
        type=int,
        default=2,
        help="Rank threshold for 'founder' bonus (1 = earliest state only).",
    )
    parser.add_argument(
        "--founder-bonus-theta",
        type=float,
        default=0.4,
        help="Log-scale reuse bonus for founder-ranked states.",
    )
    parser.add_argument(
        "--late-penalty-theta",
        type=float,
        default=-0.2,
        help="Log-scale penalty for derived/late states.",
    )
    return parser.parse_args()


def run_simulation(
    *,
    primitives: list[str],
    initial_states: list[list[str]],
    theta: dict[str, float],
    n_total: int,
    max_depth: int,
    rng: np.random.Generator,
    founder_bias_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run simulation with founder bias enabled."""
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
            founder_bias_config=founder_bias_config,
        )
        all_paths.extend(result)

    return all_paths


def build_observations(
    paths: list[dict[str, Any]], *, rng: np.random.Generator
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    """Build observations capturing founder vs. derived reuse."""
    from collections import Counter

    counts = Counter(int(path["final_state_id"]) for path in paths)
    total = max(len(paths), 1)
    observed_compounds: set[str] = set()
    records: list[dict[str, Any]] = []

    per_state_stats: dict[int, dict[str, Any]] = {}
    global_duration_sum = 0.0
    global_duration_sq_sum = 0.0
    global_founder_reuse = 0.0
    global_founder_sq = 0.0
    global_derived_reuse = 0.0
    global_derived_sq = 0.0
    global_rank_sum = 0.0
    global_rank_sq = 0.0

    for path in paths:
        state_id = int(path["final_state_id"])
        duration = float(path.get("duration", 0.0))
        feature_counts = path.get("feature_counts", {}) or {}
        founder_reuse = float(feature_counts.get("founder_reuse", 0))
        derived_reuse = float(feature_counts.get("derived_reuse", 0))
        founder_rank = float(path.get("founder_rank", 0))

        stats = per_state_stats.setdefault(
            state_id,
            {
                "count": 0,
                "duration_sum": 0.0,
                "duration_sq_sum": 0.0,
                "founder_sum": 0.0,
                "founder_sq_sum": 0.0,
                "derived_sum": 0.0,
                "derived_sq_sum": 0.0,
                "rank_sum": 0.0,
                "rank_sq_sum": 0.0,
            },
        )
        stats["count"] += 1
        stats["duration_sum"] += duration
        stats["duration_sq_sum"] += duration * duration
        stats["founder_sum"] += founder_reuse
        stats["founder_sq_sum"] += founder_reuse * founder_reuse
        stats["derived_sum"] += derived_reuse
        stats["derived_sq_sum"] += derived_reuse * derived_reuse
        stats["rank_sum"] += founder_rank
        stats["rank_sq_sum"] += founder_rank * founder_rank

        global_duration_sum += duration
        global_duration_sq_sum += duration * duration
        global_founder_reuse += founder_reuse
        global_founder_sq += founder_reuse * founder_reuse
        global_derived_reuse += derived_reuse
        global_derived_sq += derived_reuse * derived_reuse
        global_rank_sum += founder_rank
        global_rank_sq += founder_rank * founder_rank

    for state_id, count in counts.items():
        freq = count / total
        detect_p = min(1.0, 0.1 + freq * 0.8)
        detected = bool(rng.random() < detect_p)
        stats = per_state_stats.get(
            state_id,
            {
                "count": 0,
                "duration_sum": 0.0,
                "duration_sq_sum": 0.0,
                "founder_sum": 0.0,
                "founder_sq_sum": 0.0,
                "derived_sum": 0.0,
                "derived_sq_sum": 0.0,
                "rank_sum": 0.0,
                "rank_sq_sum": 0.0,
            },
        )
        denom = max(stats["count"], 1)
        mean_duration = stats["duration_sum"] / denom
        var_duration = stats["duration_sq_sum"] / denom - mean_duration**2
        mean_founder = stats["founder_sum"] / denom
        var_founder = stats["founder_sq_sum"] / denom - mean_founder**2
        mean_derived = stats["derived_sum"] / denom
        var_derived = stats["derived_sq_sum"] / denom - mean_derived**2
        mean_rank = stats["rank_sum"] / denom
        var_rank = stats["rank_sq_sum"] / denom - mean_rank**2

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
            "mean_duration": mean_duration,
            "duration_variance": var_duration,
            "mean_founder_reuse": mean_founder,
            "founder_reuse_variance": var_founder,
            "mean_derived_reuse": mean_derived,
            "derived_reuse_variance": var_derived,
            "mean_founder_rank": mean_rank,
            "founder_rank_variance": var_rank,
        }
        records.append(record)
        if detected:
            observed_compounds.add(record["compound_id"])

    global_count = max(len(paths), 1)
    mean_duration = global_duration_sum / global_count
    var_duration = global_duration_sq_sum / global_count - mean_duration**2
    mean_founder = global_founder_reuse / global_count
    var_founder = global_founder_sq / global_count - mean_founder**2
    mean_derived = global_derived_reuse / global_count
    var_derived = global_derived_sq / global_count - mean_derived**2
    mean_rank = global_rank_sum / global_count
    var_rank = global_rank_sq / global_count - mean_rank**2

    duration_summary = {
        "mean_duration": mean_duration,
        "variance_duration": var_duration,
        "mean_founder_reuse": mean_founder,
        "variance_founder_reuse": var_founder,
        "mean_derived_reuse": mean_derived,
        "variance_derived_reuse": var_derived,
        "mean_founder_rank": mean_rank,
        "variance_founder_rank": var_rank,
    }

    return records, observed_compounds, duration_summary


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    primitives = [chr(ord("A") + i) for i in range(args.n_primitives)]
    initial_states = [[p] for p in primitives[:2]]
    founder_bias_config = {
        "founder_rank_threshold": args.founder_rank_threshold,
        "founder_bonus_theta": args.founder_bonus_theta,
        "late_penalty_theta": args.late_penalty_theta,
    }

    print("Generating founder-bias dataset (Symmetry Break C):")
    print(f"  founder_rank_threshold = {args.founder_rank_threshold}")
    print(f"  founder_bonus_theta = {args.founder_bonus_theta}")
    print(f"  late_penalty_theta = {args.late_penalty_theta}")
    print(f"  Primitives: {args.n_primitives}, Depth: {args.max_depth}")
    print(f"  Trajectories: {args.n_trajectories}")

    paths = run_simulation(
        primitives=primitives,
        initial_states=initial_states,
        theta=SIGNAL_THETA,
        n_total=args.n_trajectories,
        max_depth=args.max_depth,
        rng=rng,
        founder_bias_config=founder_bias_config,
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
            "symmetry_break": "founder_bias",
            "founder_bias_config": founder_bias_config,
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
    print(f"  Mean founder reuse: {duration_summary['mean_founder_reuse']:.2f}")
    print(f"  Mean derived reuse: {duration_summary['mean_derived_reuse']:.2f}")
    print(f"  Mean founder rank: {duration_summary['mean_founder_rank']:.2f}")


if __name__ == "__main__":
    main()
