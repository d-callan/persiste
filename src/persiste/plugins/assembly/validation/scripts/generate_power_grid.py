"""Generate power/identifiability grid datasets with controllable signal strength."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import persiste_rust

DEFAULT_PRIMITIVES = [4, 6, 8]
DEFAULT_DEPTHS = [3, 5, 7]
DEFAULT_TRAJECTORIES = [100, 200, 400]
DEFAULT_OBS_TYPES = ["presence", "frequency"]
DEFAULT_THETA_MODES = ["null", "signal"]

TRUE_BASELINE = {
    "kappa": 1.0,
    "join_exponent": -0.5,
    "split_exponent": 0.3,
    "decay_rate": 0.01,
}

DEFAULT_SIGNAL_THETA = {
    "reuse_count": 0.6,
    "depth_change": -0.35,
    "size_change": 0.25,
    "symmetry_score": 0.15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate power-envelope datasets across primitive/depth/trajectory grid."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/persiste/plugins/assembly/validation/results/power_grid"),
        help="Directory to write datasets (default: validation/results/power_grid).",
    )
    parser.add_argument(
        "--primitives",
        type=int,
        nargs="+",
        default=DEFAULT_PRIMITIVES,
        help="List of primitive counts to sweep.",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=DEFAULT_DEPTHS,
        help="List of max depths to sweep.",
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        nargs="+",
        default=DEFAULT_TRAJECTORIES,
        help="Trajectory counts per dataset.",
    )
    parser.add_argument(
        "--obs-types",
        type=str,
        nargs="+",
        choices=["presence", "frequency"],
        default=DEFAULT_OBS_TYPES,
        help="Observation regimes to include.",
    )
    parser.add_argument(
        "--theta-modes",
        type=str,
        nargs="+",
        choices=["null", "signal"],
        default=DEFAULT_THETA_MODES,
        help="Theta modes to simulate (null vs signal).",
    )
    parser.add_argument(
        "--signal-theta",
        type=str,
        default=None,
        help=(
            "Override true signal theta as comma-separated key=value pairs "
            "(e.g., 'reuse_count=1.2,depth_change=-0.8')."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10_000,
        help="Base RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of datasets (for smoke testing).",
    )
    return parser.parse_args()


def parse_signal_theta(theta_str: str | None) -> dict[str, float] | None:
    if theta_str is None:
        return None
    overrides: dict[str, float] = {}
    for pair in theta_str.split(","):
        if not pair:
            continue
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            continue
        overrides[key] = float(value)
    return overrides if overrides else None


def generate_dataset(
    n_primitives: int,
    max_depth: int,
    n_trajectories: int,
    obs_type: str,
    theta_mode: str,
    seed: int,
    signal_theta_override: dict[str, float] | None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    primitives = [chr(ord("A") + i) for i in range(n_primitives)]
    initial_states = sample_initial_states(rng, primitives, n_states=min(6, n_primitives))

    theta_true = signal_theta_override or DEFAULT_SIGNAL_THETA
    if theta_mode != "signal":
        theta_true = {}
    paths = run_simulation(
        primitives=primitives,
        initial_states=initial_states,
        theta=theta_true,
        n_total=n_trajectories,
        max_depth=max_depth,
        rng=rng,
    )

    observation_records, observed_compounds, duration_summary = build_observations(
        paths, obs_type=obs_type, rng=rng
    )

    dataset = {
        "config": {
            "n_primitives": n_primitives,
            "max_depth": max_depth,
            "n_trajectories": n_trajectories,
            "obs_type": obs_type,
            "theta_mode": theta_mode,
            "seed": seed,
        },
        "primitives": primitives,
        "initial_states": initial_states,
        "observed_compounds": sorted(observed_compounds),
        "observation_records": observation_records,
        "duration_summary": duration_summary,
        "true_theta": theta_true,
        "theta_mode": theta_mode,
        "true_baseline": TRUE_BASELINE,
        "inference_baseline": TRUE_BASELINE,
    }
    return dataset


def sample_initial_states(
    rng: np.random.Generator, primitives: list[str], n_states: int
) -> list[list[str]]:
    states: list[list[str]] = []
    for _ in range(n_states):
        depth = int(rng.integers(1, 3))
        states.append(list(rng.choice(primitives, size=depth, replace=True)))
    return states


def run_simulation(
    *,
    primitives: list[str],
    initial_states: list[list[str]],
    theta: dict[str, float],
    n_total: int,
    max_depth: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    all_paths: list[dict[str, Any]] = []
    n_batches = len(initial_states)
    batch_size = int(np.ceil(n_total / max(n_batches, 1)))

    for idx, initial_parts in enumerate(initial_states):
        remaining = n_total - len(all_paths)
        if remaining <= 0:
            break
        samples = batch_size if idx < n_batches - 1 else remaining
        seed = int(rng.integers(0, 2**31 - 1))
        result = persiste_rust.simulate_assembly_trajectories(
            primitives=primitives,
            initial_parts=initial_parts,
            theta=theta,
            n_samples=samples,
            t_max=30.0,
            burn_in=10.0,
            max_depth=max_depth,
            seed=seed,
            kappa=TRUE_BASELINE["kappa"],
            join_exponent=TRUE_BASELINE["join_exponent"],
            split_exponent=TRUE_BASELINE["split_exponent"],
            decay_rate=TRUE_BASELINE["decay_rate"],
        )
        all_paths.extend(result)

    return all_paths


def build_observations(
    paths: list[dict[str, Any]], *, obs_type: str, rng: np.random.Generator
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
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
        record = {
            "state_id": state_id,
            "compound_id": f"state_{state_id}",
            "presence_estimate": freq,
            "detect_probability": detect_p,
            "detected": detected,
            "frequency": None,
            "duration_histogram": duration_info["hist"],
            "mean_duration": mean_duration,
            "duration_variance": variance_duration,
            "mean_max_depth": mean_depth,
            "max_depth_variance": variance_depth,
            "max_depth_observed": duration_info["depth_max"],
            "mean_reuse_count": mean_reuse,
            "reuse_count_variance": variance_reuse,
        }

        if obs_type == "frequency":
            lam = max(freq * len(paths), 0.5)
            record["frequency"] = int(rng.poisson(lam))
            if record["frequency"] > 0:
                detected = True
                record["detected"] = True

        if detected:
            observed_compounds.add(record["compound_id"])
        records.append(record)

    global_count = max(len(paths), 1)
    global_mean = global_sum / global_count
    global_variance = global_sq_sum / global_count - global_mean**2
    max_depth_bins = (
        list(range(max(global_depth_hist.keys(), default=0) + 1))
        if global_depth_hist
        else [0]
    )
    max_depth_hist = [global_depth_hist.get(b, 0) for b in max_depth_bins]
    global_count = max(len(paths), 1)
    global_mean = global_sum / global_count
    global_variance = global_sq_sum / global_count - global_mean**2
    depth_mean = global_depth_sum / global_count
    depth_variance = global_depth_sq_sum / global_count - depth_mean**2
    reuse_mean = global_reuse_sum / max(len(paths), 1)
    reuse_variance = global_reuse_sq_sum / max(len(paths), 1) - reuse_mean**2
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


def write_dataset(dataset: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2)


def main() -> None:
    args = parse_args()
    signal_theta_override = parse_signal_theta(args.signal_theta)

    combos = list(
        product(
            args.primitives,
            args.depths,
            args.trajectories,
            args.obs_types,
            args.theta_modes,
        )
    )

    if args.limit is not None:
        combos = combos[: args.limit]

    for idx, (n_p, depth, traj, obs, theta_mode) in enumerate(combos):
        seed = args.seed + idx
        dataset = generate_dataset(
            n_primitives=n_p,
            max_depth=depth,
            n_trajectories=traj,
            obs_type=obs,
            theta_mode=theta_mode,
            seed=seed,
            signal_theta_override=signal_theta_override if theta_mode == "signal" else None,
        )
        filename = (
            f"pr{n_p}_depth{depth}_traj{traj}_{obs}_{theta_mode}_seed{seed}.json"
        )
        output_path = args.output_dir / filename
        write_dataset(dataset, output_path)
        print(
            f"[power-grid] primitives={n_p} depth={depth} traj={traj} "
            f"obs={obs} theta={theta_mode} -> {output_path}"
        )


if __name__ == "__main__":
    main()
