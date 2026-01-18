#!/usr/bin/env python3
"""
V1: Null calibration sweep.

Validates that ΔLL ≈ 0 when θ = 0 across deterministic and stochastic inference modes.

Usage:
    python run_null_calibration.py --n-seeds 3 --output null_calib.csv
"""

import argparse
import csv
from pathlib import Path

from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints


def run_null_calibration(
    n_seeds: int,
    primitives: list[str],
    observed_compounds: set[str],
) -> list[dict]:
    """Run null calibration across seeds and inference modes."""
    results = []

    modes = [
        InferenceMode.SCREEN_ONLY,
        InferenceMode.FULL_STOCHASTIC,
        InferenceMode.SCREEN_AND_REFINE,
    ]

    for seed in range(n_seeds):
        for mode in modes:
            kwargs = {
                "observed_compounds": observed_compounds,
                "primitives": primitives,
                "mode": mode,
                "seed": seed,
            }

            # Mode-specific parameters
            if mode == InferenceMode.SCREEN_ONLY:
                kwargs["screen_budget"] = 10
            elif mode == InferenceMode.FULL_STOCHASTIC:
                kwargs["n_samples"] = 30
            elif mode == InferenceMode.SCREEN_AND_REFINE:
                kwargs["screen_budget"] = 10
                kwargs["n_samples"] = 30

            result = fit_assembly_constraints(**kwargs)

            row = {
                "seed": seed,
                "mode": mode.value,
                "deterministic_delta_ll": result.get("deterministic_delta_ll"),
                "stochastic_delta_ll": result.get("stochastic_delta_ll"),
                "theta_hat": str(result.get("theta_hat", {})),
            }
            results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser(description="V1: Null calibration sweep")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--primitives", type=str, default="A,B", help="Comma-separated primitives")
    parser.add_argument("--output", type=str, default="null_calibration.csv", help="Output CSV")

    args = parser.parse_args()

    primitives = args.primitives.split(",")
    observed_compounds = set(primitives)

    print("Running null calibration sweep...")
    print(f"  Seeds: {args.n_seeds}")
    print(f"  Primitives: {primitives}")

    results = run_null_calibration(
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
            fieldnames=["seed", "mode", "deterministic_delta_ll", "stochastic_delta_ll", "theta_hat"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\nNull Calibration Summary:")
    for mode in ["screen-only", "full-stochastic", "screen-and-refine"]:
        mode_results = [r for r in results if r["mode"] == mode]
        if mode_results:
            det_lls = [r["deterministic_delta_ll"] for r in mode_results if r["deterministic_delta_ll"] is not None]
            stoch_lls = [r["stochastic_delta_ll"] for r in mode_results if r["stochastic_delta_ll"] is not None]

            print(f"  {mode}:")
            if det_lls:
                print(f"    Deterministic ΔLL: mean={sum(det_lls)/len(det_lls):.4f}, max={max(abs(x) for x in det_lls):.4f}")
            if stoch_lls:
                print(f"    Stochastic ΔLL: mean={sum(stoch_lls)/len(stoch_lls):.4f}, max={max(abs(x) for x in stoch_lls):.4f}")


if __name__ == "__main__":
    main()
