"""Generate assembly benchmark dataset for robustness/perf testing.

This script synthesizes primitives, runs the Rust Gillespie simulator, caches
latent trajectories, and emits observation data with missingness + false positives.

Usage:
    python scripts/assembly/generate_benchmark_dataset.py --output-dir data/assembly_benchmark
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import persiste_rust

PRIMITIVE_TYPES = ["hydro", "lipid", "metal", "amine", "silicate"]
PRIMITIVE_MOTIFS = ["helix", "ring", "branch", "cluster", "ladder"]
FEATURE_NAMES = [
    "reuse_bonus",
    "depth_penalty",
    "motif_bonus",
    "size_change",
    "symmetry_bias",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate assembly benchmark dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/assembly_benchmark"),
        help="Directory to write dataset artifacts (default: data/assembly_benchmark)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducibility (default: 2026)",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=300,
        help="Number of Gillespie trajectories to simulate (default: 300)",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=25.0,
        help="Maximum simulation time horizon (default: 25.0)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum assembly depth for simulator (default: 6)",
    )
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=0.25,
        help="Fraction of states to drop from observations (default: 0.25)",
    )
    return parser.parse_args()


def generate_primitives(rng: np.random.Generator, n_primitives: int = 18) -> list[dict[str, Any]]:
    primitives = []
    letters = [chr(ord("A") + i) for i in range(n_primitives)]
    for name in letters:
        primitive = {
            "name": name,
            "size": int(rng.integers(1, 5)),
            "type": str(rng.choice(PRIMITIVE_TYPES)),
            "motifs": [],
        }
        if rng.random() < 0.35:
            motif = str(rng.choice(PRIMITIVE_MOTIFS))
            primitive["motifs"].append(motif)
        primitives.append(primitive)
    return primitives


def sample_initial_states(
    rng: np.random.Generator,
    primitive_names: list[str],
    n_states: int = 5,
) -> list[list[str]]:
    states: list[list[str]] = []
    for _ in range(n_states):
        depth = int(rng.integers(1, 3))  # depth 1-2
        parts = list(rng.choice(primitive_names, size=depth, replace=True))
        states.append(parts)
    return states


def run_simulation_batches(
    primitives: list[dict[str, Any]],
    initial_states: list[list[str]],
    theta_ref: dict[str, float],
    n_total: int,
    t_max: float,
    max_depth: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    primitive_names = [p["name"] for p in primitives]
    baseline = {
        "kappa": 1.0,
        "join_exponent": -0.5,
        "split_exponent": 0.3,
        "decay_rate": 0.01,
    }
    n_batches = len(initial_states)
    batch_size = int(np.ceil(n_total / n_batches))
    all_paths: list[dict[str, Any]] = []

    for batch_idx, initial_parts in enumerate(initial_states):
        samples = batch_size if batch_idx < n_batches - 1 else n_total - len(all_paths)
        if samples <= 0:
            break
        seed = int(rng.integers(0, 2**31 - 1))
        result = persiste_rust.simulate_assembly_trajectories(
            primitives=primitive_names,
            initial_parts=initial_parts,
            theta=theta_ref,
            n_samples=samples,
            t_max=t_max,
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
    paths: list[dict[str, Any]],
    rng: np.random.Generator,
    detect_prob: float = 0.7,
    false_positive: float = 0.05,
    missing_rate: float = 0.25,
) -> list[dict[str, Any]]:
    state_counts = Counter(path["final_state_id"] for path in paths)
    total = len(paths)
    observations: list[dict[str, Any]] = []

    for state_id, count in state_counts.items():
        presence = count / max(total, 1)
        detect_p = presence * detect_prob + (1 - presence) * false_positive
        detected = bool(rng.random() < detect_p)

        frequency = None
        if rng.random() < 0.4:
            lam = max(presence * 8.0, 0.2)
            frequency = int(rng.poisson(lam))

        record = {
            "state_id": int(state_id),
            "compound_id": f"state_{int(state_id)}",
            "presence_estimate": presence,
            "detect_probability": detect_p,
            "detected": detected,
            "frequency": frequency,
        }
        observations.append(record)

    # Apply missingness
    observations = [obs for obs in observations if rng.random() > missing_rate]

    # Add a few pure false positives (states never seen)
    for _ in range(max(1, len(observations) // 10)):
        fake_state = int(rng.integers(10_000, 99_999))
        observations.append(
            {
                "state_id": fake_state,
                "compound_id": f"state_{fake_state}",
                "presence_estimate": 0.0,
                "detect_probability": false_positive,
                "detected": rng.random() < false_positive,
                "frequency": None,
            }
        )

    return observations


def build_constraint_config(rng: np.random.Generator) -> dict[str, float]:
    # Choose 5 features with varied strengths (one strong motif bonus)
    weights = {
        "reuse_bonus": round(float(rng.normal(0.8, 0.2)), 3),
        "depth_penalty": round(float(rng.normal(-0.4, 0.1)), 3),
        "motif_bonus": 2.0,
        "size_change": round(float(rng.normal(0.3, 0.05)), 3),
        "symmetry_bias": round(float(rng.normal(0.2, 0.05)), 3),
    }
    return weights


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    primitives = generate_primitives(rng)
    primitive_names = [p["name"] for p in primitives]
    initial_states = sample_initial_states(rng, primitive_names)

    theta_ref: dict[str, float] = {}
    theta_strong = build_constraint_config(rng)

    paths = run_simulation_batches(
        primitives=primitives,
        initial_states=initial_states,
        theta_ref=theta_ref,
        n_total=args.n_trajectories,
        t_max=args.t_max,
        max_depth=args.max_depth,
        rng=rng,
    )

    observations = build_observations(
        paths,
        rng,
        detect_prob=0.7,
        false_positive=0.05,
        missing_rate=args.missing_rate,
    )
    detected_compounds = sorted(
        {obs["compound_id"] for obs in observations if obs["detected"]}
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    primitives_path = output_dir / "primitives.json"
    latent_cache_path = output_dir / "latent_cache.json"
    observations_path = output_dir / "observations.json"
    config_path = output_dir / "config.json"

    write_json(primitives_path, {"primitives": primitives})
    write_json(
        latent_cache_path,
        {
            "theta_ref": theta_ref,
            "paths": paths,
        },
    )
    write_json(
        observations_path,
        {
            "detect_prob": 0.7,
            "false_positive": 0.05,
            "missing_rate": args.missing_rate,
            "records": observations,
        },
    )
    write_json(
        config_path,
        {
            "n_trajectories": args.n_trajectories,
            "t_max": args.t_max,
            "max_depth": args.max_depth,
            "primitives": primitives_path.name,
            "latent_cache": latent_cache_path.name,
            "observations": observations_path.name,
            "theta_reference": theta_ref,
            "theta_strong": theta_strong,
            "detected_compounds": detected_compounds,
            "screening": {
                "mode": "auto",
                "budget": 100,
                "top_k": 10,
                "refine_radius": 0.5,
            },
            "cache": {
                "trust_radius": 1.0,
                "ess_threshold": 0.3,
            },
        },
    )

    print(f"Wrote primitives to {primitives_path}")
    print(f"Wrote latent cache to {latent_cache_path} ({len(paths)} paths)")
    print(f"Wrote observations to {observations_path} (n={len(observations)})")
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
