#!/usr/bin/env python3
"""
Stage I: Feature-level null tests.

Validates that individual features have mean ≈ 0, stable variance, and no drift
when data are generated from the baseline (θ = 0).

Usage:
    python run_feature_nulls.py --n-seeds 5 --n-samples 50 --output results.csv
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import persiste_rust

from persiste.plugins.assembly.features.assembly_features import AssemblyFeatureExtractor


def extract_features_from_paths(
    paths: list[dict[str, Any]],
    feature_extractor: AssemblyFeatureExtractor,
) -> dict[str, list[float]]:
    """Extract feature values from simulated paths."""
    feature_values = {}

    for path in paths:
        feature_counts = path["feature_counts"]
        for feature_name, count in feature_counts.items():
            if feature_name not in feature_values:
                feature_values[feature_name] = []
            feature_values[feature_name].append(float(count))

    return feature_values


def run_null_test(
    n_seeds: int,
    n_samples: int,
    t_max: float,
    burn_in: float,
    max_depth: int,
    primitives: list[str],
    initial_parts: list[str],
) -> dict[str, Any]:
    """Run feature-level null tests across seeds."""
    results = {}

    for seed in range(n_seeds):
        # Simulate at θ = 0 (baseline only)
        paths = persiste_rust.simulate_assembly_trajectories(
            primitives=primitives,
            initial_parts=initial_parts,
            theta={},  # null
            n_samples=n_samples,
            t_max=t_max,
            burn_in=burn_in,
            max_depth=max_depth,
            seed=seed,
        )

        # Extract features
        feature_values = extract_features_from_paths(paths, AssemblyFeatureExtractor())

        # Compute statistics per feature
        for feature_name, values in feature_values.items():
            if feature_name not in results:
                results[feature_name] = {
                    "means": [],
                    "stds": [],
                    "mins": [],
                    "maxs": [],
                }

            results[feature_name]["means"].append(np.mean(values))
            results[feature_name]["stds"].append(np.std(values))
            results[feature_name]["mins"].append(np.min(values))
            results[feature_name]["maxs"].append(np.max(values))

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage I: Feature-level null tests")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--n-samples", type=int, default=50, help="Trajectories per seed")
    parser.add_argument("--t-max", type=float, default=10.0, help="Simulation time")
    parser.add_argument("--burn-in", type=float, default=2.0, help="Burn-in time")
    parser.add_argument("--max-depth", type=int, default=4, help="Max assembly depth")
    parser.add_argument("--primitives", type=str, default="A,B", help="Comma-separated primitives")
    parser.add_argument("--output", type=str, default="feature_nulls.json", help="Output file")

    args = parser.parse_args()

    primitives = args.primitives.split(",")
    initial_parts = [primitives[0]]

    print("Running feature-level null tests...")
    print(f"  Seeds: {args.n_seeds}")
    print(f"  Samples per seed: {args.n_samples}")
    print(f"  Primitives: {primitives}")

    results = run_null_test(
        n_seeds=args.n_seeds,
        n_samples=args.n_samples,
        t_max=args.t_max,
        burn_in=args.burn_in,
        max_depth=args.max_depth,
        primitives=primitives,
        initial_parts=initial_parts,
    )

    # Summarize and save
    summary = {}
    for feature_name, stats in results.items():
        summary[feature_name] = {
            "mean_of_means": float(np.mean(stats["means"])),
            "std_of_means": float(np.std(stats["means"])),
            "mean_of_stds": float(np.mean(stats["stds"])),
            "overall_min": float(np.min(stats["mins"])),
            "overall_max": float(np.max(stats["maxs"])),
        }

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print("\nFeature Summary:")
    for feature_name, stats in summary.items():
        print(f"  {feature_name}:")
        print(f"    Mean of means: {stats['mean_of_means']:.6f}")
        print(f"    Std of means: {stats['std_of_means']:.6f}")
        print(f"    Mean of stds: {stats['mean_of_stds']:.6f}")


if __name__ == "__main__":
    main()
