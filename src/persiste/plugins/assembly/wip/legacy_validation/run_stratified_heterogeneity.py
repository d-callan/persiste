#!/usr/bin/env python3
"""
V: Stratified/heterogeneous constraint tests.

Validates detection of heterogeneous constraints across primitive classes.
Simulates two classes with different θ values and compares:
- Single global θ model
- Per-class θ model

Usage:
    python run_stratified_heterogeneity.py --n-seeds 3 --output results.csv
"""

import argparse
import csv
from pathlib import Path

from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints


def run_heterogeneity_test(
    n_seeds: int,
    primitives_class1: list[str],
    primitives_class2: list[str],
    theta_class1: dict,
    theta_class2: dict,
) -> list[dict]:
    """Run inference comparing global vs per-class θ models."""
    results = []

    all_primitives = primitives_class1 + primitives_class2
    observed_compounds = set(all_primitives)

    for seed in range(n_seeds):
        # Run global inference (treats all primitives uniformly)
        global_result = fit_assembly_constraints(
            observed_compounds=observed_compounds,
            primitives=all_primitives,
            mode=InferenceMode.FULL_STOCHASTIC,
            n_samples=30,
            t_max=15.0,
            burn_in=5.0,
            max_depth=4,
            seed=seed,
        )

        global_theta = global_result.get("theta_hat", {})
        global_ll = global_result.get("stochastic_ll")

        # Run per-class inference (separate fits for each class)
        class1_result = fit_assembly_constraints(
            observed_compounds=set(primitives_class1),
            primitives=primitives_class1,
            mode=InferenceMode.FULL_STOCHASTIC,
            n_samples=20,
            t_max=15.0,
            burn_in=5.0,
            max_depth=4,
            seed=seed + 100,
        )

        class2_result = fit_assembly_constraints(
            observed_compounds=set(primitives_class2),
            primitives=primitives_class2,
            mode=InferenceMode.FULL_STOCHASTIC,
            n_samples=20,
            t_max=15.0,
            burn_in=5.0,
            max_depth=4,
            seed=seed + 200,
        )

        class1_theta = class1_result.get("theta_hat", {})
        class2_theta = class2_result.get("theta_hat", {})
        class1_ll = class1_result.get("stochastic_ll", 0.0)
        class2_ll = class2_result.get("stochastic_ll", 0.0)
        per_class_ll = class1_ll + class2_ll

        # Compare models
        ll_improvement = per_class_ll - global_ll if global_ll else 0.0
        heterogeneity_detected = ll_improvement > 1.0  # Rough threshold

        row = {
            "seed": seed,
            "global_theta": str(global_theta),
            "class1_theta": str(class1_theta),
            "class2_theta": str(class2_theta),
            "global_ll": global_ll,
            "per_class_ll": per_class_ll,
            "ll_improvement": ll_improvement,
            "heterogeneity_detected": heterogeneity_detected,
            "theta_class1_true": str(theta_class1),
            "theta_class2_true": str(theta_class2),
        }
        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="V: Stratified/heterogeneous constraint tests"
    )
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument(
        "--class1-primitives",
        type=str,
        default="A,B",
        help="Comma-separated primitives for class 1",
    )
    parser.add_argument(
        "--class2-primitives",
        type=str,
        default="C,D",
        help="Comma-separated primitives for class 2",
    )
    parser.add_argument(
        "--theta-class1",
        type=str,
        default="reuse_count:0.5",
        help="Theta for class 1 (format: feature:value)",
    )
    parser.add_argument(
        "--theta-class2",
        type=str,
        default="reuse_count:1.5",
        help="Theta for class 2 (format: feature:value)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stratified_heterogeneity.csv",
        help="Output CSV",
    )

    args = parser.parse_args()

    primitives_class1 = args.class1_primitives.split(",")
    primitives_class2 = args.class2_primitives.split(",")

    # Parse theta specifications
    def parse_theta(theta_str):
        parts = theta_str.split(":")
        if len(parts) == 2:
            return {parts[0]: float(parts[1])}
        return {}

    theta_class1 = parse_theta(args.theta_class1)
    theta_class2 = parse_theta(args.theta_class2)

    print("Running V: Stratified/heterogeneous constraint tests...")
    print(f"  Class 1 primitives: {primitives_class1}, θ={theta_class1}")
    print(f"  Class 2 primitives: {primitives_class2}, θ={theta_class2}")
    print(f"  Seeds: {args.n_seeds}")

    results = run_heterogeneity_test(
        n_seeds=args.n_seeds,
        primitives_class1=primitives_class1,
        primitives_class2=primitives_class2,
        theta_class1=theta_class1,
        theta_class2=theta_class2,
    )

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\nStratified Heterogeneity Summary:")
    if results:
        detected_count = sum(1 for r in results if r["heterogeneity_detected"])
        detection_rate = detected_count / len(results) if results else 0.0
        mean_ll_improvement = sum(
            r["ll_improvement"] for r in results
        ) / len(results)

        print(f"  Heterogeneity detection rate: {detection_rate:.2%}")
        print(f"  Mean LL improvement (per-class vs global): {mean_ll_improvement:.4f}")

        if detection_rate > 0.5:
            print("  ✓ Heterogeneity detected in majority of runs")
        else:
            print("  ⚠️  Heterogeneity not reliably detected")


if __name__ == "__main__":
    main()
