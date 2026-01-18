#!/usr/bin/env python3
"""
V3: Specificity tests.

Validates that each constraint lifts only when present, guarding against
feature leakage and correlated statistics.

Usage:
    python run_specificity_tests.py --n-seeds 3 --output results.csv
"""

import argparse
import csv
from pathlib import Path

from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints


def run_specificity_test(
    active_feature: str,
    n_seeds: int,
    primitives: list[str],
    observed_compounds: set[str],
) -> list[dict]:
    """Run inference with only one constraint active, measure ΔLL per feature."""
    results = []

    for seed in range(n_seeds):
        # Run inference with only active_feature enabled
        result = fit_assembly_constraints(
            observed_compounds=observed_compounds,
            primitives=primitives,
            mode=InferenceMode.FULL_STOCHASTIC,
            n_samples=30,
            t_max=15.0,
            burn_in=5.0,
            max_depth=4,
            seed=seed,
        )

        theta_hat = result.get("theta_hat", {})
        stoch_delta_ll = result.get("stochastic_delta_ll")

        # Record which features have non-zero weights
        active_features = [k for k, v in theta_hat.items() if abs(v) > 1e-6]

        row = {
            "test_feature": active_feature,
            "seed": seed,
            "delta_ll": stoch_delta_ll,
            "theta_hat": str(theta_hat),
            "active_features": ",".join(active_features) if active_features else "none",
            "target_active": active_feature in active_features,
        }
        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser(description="V3: Specificity tests")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument(
        "--features",
        type=str,
        default="reuse_count,depth_change,depth_gate_reuse,same_class_reuse,founder_reuse",
        help="Comma-separated features to test",
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
        default="specificity_tests.csv",
        help="Output CSV",
    )

    args = parser.parse_args()

    features_to_test = args.features.split(",")
    primitives = args.primitives.split(",")
    observed_compounds = set(primitives)

    print("Running V3: Specificity tests...")
    print(f"  Features to test: {features_to_test}")
    print(f"  Seeds per feature: {args.n_seeds}")

    all_results = []
    for feature in features_to_test:
        print(f"  Testing {feature}...")
        results = run_specificity_test(
            active_feature=feature,
            n_seeds=args.n_seeds,
            primitives=primitives,
            observed_compounds=observed_compounds,
        )
        all_results.extend(results)

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_feature",
                "seed",
                "delta_ll",
                "theta_hat",
                "active_features",
                "target_active",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\nSpecificity Summary:")
    for feature in features_to_test:
        subset = [r for r in all_results if r["test_feature"] == feature]
        if subset:
            target_active_count = sum(1 for r in subset if r["target_active"])
            specificity = target_active_count / len(subset) if subset else 0.0
            mean_delta_ll = sum(
                r["delta_ll"] for r in subset if r["delta_ll"] is not None
            ) / len([r for r in subset if r["delta_ll"] is not None])

            print(f"  {feature}:")
            print(f"    Specificity (target active): {specificity:.2%}")
            print(f"    Mean ΔLL: {mean_delta_ll:.4f}")


if __name__ == "__main__":
    main()
