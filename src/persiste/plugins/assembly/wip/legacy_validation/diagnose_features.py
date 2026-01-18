#!/usr/bin/env python3
"""
Diagnostic: Check feature extraction on simulated data.

Simulates trajectories with different theta values and extracts features
to verify that theta actually affects the feature statistics.

Usage:
    python diagnose_features.py --theta-values 0.0,0.5,1.0,1.5 --n-samples 100 --output features_diag.csv
"""

import argparse
import csv
import json
from pathlib import Path

import persiste_rust

from persiste.plugins.assembly.features.assembly_features import (
    AssemblyFeatureExtractor,
)


def extract_features_from_paths(paths) -> dict:
    """Extract aggregate feature statistics from simulated paths."""
    if isinstance(paths, str):
        paths = json.loads(paths)

    feature_counts = {
        "reuse_count": [],
        "depth_change": [],
        "size_change": [],
        "transition_join": [],
        "transition_split": [],
        "transition_decay": [],
    }

    for path in paths:
        features = path.get("feature_counts", {})
        for key in feature_counts:
            if key in features:
                feature_counts[key].append(features[key])

    # Compute statistics
    stats = {}
    for key, values in feature_counts.items():
        if values:
            stats[key] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
        else:
            stats[key] = {"mean": 0, "min": 0, "max": 0, "count": 0}

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose feature extraction on simulated data"
    )
    parser.add_argument(
        "--theta-values",
        type=str,
        default="0.0,0.5,1.0,1.5",
        help="Comma-separated theta_reuse values",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of trajectories per theta",
    )
    parser.add_argument(
        "--primitives",
        type=str,
        default="A,B",
        help="Comma-separated primitives",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="features_diag.csv",
        help="Output CSV",
    )

    args = parser.parse_args()

    theta_values = [float(x) for x in args.theta_values.split(",")]
    primitives = args.primitives.split(",")

    print("Diagnosing feature extraction...")
    print(f"  Theta values: {theta_values}")
    print(f"  Samples per theta: {args.n_samples}")
    print(f"  Primitives: {primitives}")

    results = []

    for theta_reuse in theta_values:
        print(f"\n  Simulating with θ_reuse = {theta_reuse}...")

        theta_true = {"reuse_count": theta_reuse}

        # Simulate trajectories
        paths_json = persiste_rust.simulate_assembly_trajectories(
            primitives=primitives,
            initial_parts=[primitives[0]],
            theta=theta_true,
            n_samples=args.n_samples,
            t_max=20.0,
            burn_in=5.0,
            max_depth=9,
            seed=42,
        )

        # Extract features
        feature_stats = extract_features_from_paths(paths_json)

        print(f"    Feature statistics:")
        for feature_name, stats in feature_stats.items():
            print(
                f"      {feature_name}: mean={stats['mean']:.2f}, "
                f"min={stats['min']}, max={stats['max']}"
            )

        # Store results
        for feature_name, stats in feature_stats.items():
            row = {
                "theta_reuse": theta_reuse,
                "feature": feature_name,
                "mean": stats["mean"],
                "min": stats["min"],
                "max": stats["max"],
                "count": stats["count"],
            }
            results.append(row)

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["theta_reuse", "feature", "mean", "min", "max", "count"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\nFeature Extraction Diagnostic Summary:")
    print("Checking if features change with theta...")

    for feature_name in [
        "reuse_count",
        "depth_change",
        "size_change",
        "transition_join",
    ]:
        feature_results = [r for r in results if r["feature"] == feature_name]
        if feature_results:
            means = [r["mean"] for r in feature_results]
            print(f"\n  {feature_name}:")
            for theta_reuse, mean in zip(theta_values, means):
                print(f"    θ={theta_reuse}: mean={mean:.2f}")

            # Check if means differ
            if len(set(round(m, 2) for m in means)) > 1:
                print(f"    ✓ Feature varies with theta")
            else:
                print(f"    ✗ Feature DOES NOT vary with theta (potential bug!)")


if __name__ == "__main__":
    main()
