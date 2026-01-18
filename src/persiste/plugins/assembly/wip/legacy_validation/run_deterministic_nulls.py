#!/usr/bin/env python3
"""
Stage II: Deterministic null sanity checks.

Validates that deterministic screening alone (without stochastic refinement)
reports ΔLL ≈ 0 on null data, catching scaling mismatches before stochastic runs.

Usage:
    python run_deterministic_nulls.py --n-seeds 5 --output results.csv
"""

import argparse
import csv
from pathlib import Path

from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints


def run_deterministic_nulls(
    n_seeds: int,
    primitives: list[str],
    observed_compounds: set[str],
) -> list[dict]:
    """Run deterministic screening on null data across seeds."""
    results = []

    for seed in range(n_seeds):
        result = fit_assembly_constraints(
            observed_compounds=observed_compounds,
            primitives=primitives,
            mode=InferenceMode.SCREEN_ONLY,
            screen_budget=15,
            seed=seed,
        )

        # Extract screening results
        screening_results = result.get("screening_results", [])
        if screening_results:
            top_result = screening_results[0]
            row = {
                "seed": seed,
                "n_candidates": len(screening_results),
                "top_theta": str(top_result["theta"]),
                "top_delta_ll": top_result["delta_ll"],
                "top_normalized_delta_ll": top_result["normalized_delta_ll"],
                "top_passed": top_result["passed"],
                "deterministic_delta_ll": result.get("deterministic_delta_ll"),
            }
        else:
            row = {
                "seed": seed,
                "n_candidates": 0,
                "top_theta": "{}",
                "top_delta_ll": None,
                "top_normalized_delta_ll": None,
                "top_passed": False,
                "deterministic_delta_ll": result.get("deterministic_delta_ll"),
            }
        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stage II: Deterministic null sanity checks"
    )
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--primitives", type=str, default="A,B", help="Comma-separated primitives")
    parser.add_argument("--output", type=str, default="deterministic_nulls.csv", help="Output CSV")

    args = parser.parse_args()

    primitives = args.primitives.split(",")
    observed_compounds = set(primitives)

    print("Running Stage II: Deterministic null sanity checks...")
    print(f"  Seeds: {args.n_seeds}")
    print(f"  Primitives: {primitives}")

    results = run_deterministic_nulls(
        n_seeds=args.n_seeds,
        primitives=primitives,
        observed_compounds=observed_compounds,
    )

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "n_candidates",
                "top_theta",
                "top_delta_ll",
                "top_normalized_delta_ll",
                "top_passed",
                "deterministic_delta_ll",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\nDeterministic Null Sanity Check Summary:")
    delta_lls = [r["deterministic_delta_ll"] for r in results if r["deterministic_delta_ll"] is not None]
    norm_delta_lls = [
        r["top_normalized_delta_ll"]
        for r in results
        if r["top_normalized_delta_ll"] is not None
    ]

    if delta_lls:
        print(f"  Deterministic ΔLL: mean={sum(delta_lls)/len(delta_lls):.4f}, max_abs={max(abs(x) for x in delta_lls):.4f}")
    if norm_delta_lls:
        print(f"  Top normalized ΔLL: mean={sum(norm_delta_lls)/len(norm_delta_lls):.4f}, max={max(norm_delta_lls):.4f}")

    # Check for failures
    failures = [r for r in results if r["top_passed"]]
    if failures:
        print(f"\n⚠️  WARNING: {len(failures)} seeds had top candidate marked as 'passed'")
        print("  This may indicate threshold escalation on null data.")


if __name__ == "__main__":
    main()
