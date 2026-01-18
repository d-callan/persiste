#!/usr/bin/env python3
"""
Reuse-only overdrive experiment.

Isolates reuse_count constraint (no depth gates, no founder bias, no context classes).
Maps the detectability boundary: at what (θ, N, depth) does ΔLL reliably go positive?

Usage:
    python run_reuse_overdrive.py --theta-values 0.5,1.0,1.5,2.0 --n-samples 200,500,1000 --max-depths 7,9,11 --output overdrive.csv
"""

import argparse
import csv
from pathlib import Path

import persiste_rust

from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints


def run_reuse_overdrive_point(
    theta_reuse: float,
    n_samples: int,
    max_depth: int,
    primitives: list[str],
    observed_compounds: set[str],
    seed: int,
) -> dict:
    """Run inference at a single reuse-only overdrive point."""
    # Simulate data with only reuse_count active
    theta_true = {"reuse_count": theta_reuse}

    persiste_rust.simulate_assembly_trajectories(
        primitives=primitives,
        initial_parts=[primitives[0]],
        theta=theta_true,
        n_samples=n_samples,
        t_max=20.0,
        burn_in=5.0,
        max_depth=max_depth,
        seed=seed,
    )

    # Run inference with only reuse_count enabled
    result = fit_assembly_constraints(
        observed_compounds=observed_compounds,
        primitives=primitives,
        mode=InferenceMode.FULL_STOCHASTIC,
        feature_names=["reuse_count"],  # Only reuse_count, no gates
        n_samples=n_samples,
        t_max=20.0,
        burn_in=5.0,
        max_depth=max_depth,
        seed=seed + 1000,
    )

    theta_hat = result.get("theta_hat", {})
    stoch_ll = result.get("stochastic_ll")
    stoch_delta_ll = result.get("stochastic_delta_ll")
    cache_stats = result.get("cache_stats", {})

    # Compute recovery and bias
    theta_hat_reuse = theta_hat.get("reuse_count", 0.0)
    recovered = abs(theta_hat_reuse - theta_reuse) < 0.3  # Recovery threshold
    bias = theta_hat_reuse - theta_reuse
    ess = cache_stats.get("ess_ratio", 1.0) if cache_stats else 1.0

    # Flag if ΔLL is positive (signal detected)
    signal_detected = stoch_delta_ll > 0.5 if stoch_delta_ll is not None else False

    return {
        "theta_true": theta_reuse,
        "theta_hat": theta_hat_reuse,
        "delta_ll": stoch_delta_ll,
        "absolute_ll": stoch_ll,
        "recovered": recovered,
        "bias": bias,
        "ess": ess,
        "signal_detected": signal_detected,
        "n_paths": cache_stats.get("n_paths", n_samples) if cache_stats else n_samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reuse-only overdrive: map detectability boundary"
    )
    parser.add_argument(
        "--theta-values",
        type=str,
        default="0.5,1.0,1.5,2.0",
        help="Comma-separated theta_reuse values to test",
    )
    parser.add_argument(
        "--n-samples",
        type=str,
        default="200,500,1000",
        help="Comma-separated trajectory counts",
    )
    parser.add_argument(
        "--max-depths",
        type=str,
        default="7,9,11",
        help="Comma-separated max depths",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=2,
        help="Number of random seeds per grid point",
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
        default="reuse_overdrive.csv",
        help="Output CSV",
    )

    args = parser.parse_args()

    theta_values = [float(x) for x in args.theta_values.split(",")]
    n_samples_list = [int(x) for x in args.n_samples.split(",")]
    max_depth_list = [int(x) for x in args.max_depths.split(",")]
    primitives = args.primitives.split(",")
    observed_compounds = set(primitives)

    print("Running reuse-only overdrive experiment...")
    print(f"  θ_reuse values: {theta_values}")
    print(f"  Trajectories: {n_samples_list}")
    print(f"  Max depths: {max_depth_list}")
    print(f"  Seeds per point: {args.n_seeds}")

    results = []
    total_points = len(theta_values) * len(n_samples_list) * len(max_depth_list) * args.n_seeds
    completed = 0

    for theta_reuse in theta_values:
        for n_samples in n_samples_list:
            for max_depth in max_depth_list:
                for seed in range(args.n_seeds):
                    completed += 1
                    print(
                        f"  [{completed}/{total_points}] θ={theta_reuse}, n={n_samples}, "
                        f"depth={max_depth}, seed={seed}"
                    )

                    try:
                        row_data = run_reuse_overdrive_point(
                            theta_reuse=theta_reuse,
                            n_samples=n_samples,
                            max_depth=max_depth,
                            primitives=primitives,
                            observed_compounds=observed_compounds,
                            seed=seed,
                        )

                        row = {
                            "theta_reuse": theta_reuse,
                            "n_samples": n_samples,
                            "max_depth": max_depth,
                            "seed": seed,
                            **row_data,
                        }
                        results.append(row)
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        results.append({
                            "theta_reuse": theta_reuse,
                            "n_samples": n_samples,
                            "max_depth": max_depth,
                            "seed": seed,
                            "error": str(e),
                        })

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
    print("\nReuse-Only Overdrive Summary:")
    print("Detectability boundary (ΔLL > 0.5):")

    for theta_reuse in theta_values:
        for n_samples in n_samples_list:
            for max_depth in max_depth_list:
                subset = [
                    r
                    for r in results
                    if r.get("theta_reuse") == theta_reuse
                    and r.get("n_samples") == n_samples
                    and r.get("max_depth") == max_depth
                    and "error" not in r
                ]
                if subset:
                    detected = sum(1 for r in subset if r.get("signal_detected"))
                    detection_rate = detected / len(subset) if subset else 0.0
                    recovered = sum(1 for r in subset if r.get("recovered"))
                    recovery_rate = recovered / len(subset) if subset else 0.0
                    biases = [r.get("bias", 0.0) for r in subset if "bias" in r]
                    mean_bias = sum(biases) / len(biases) if biases else 0.0
                    delta_lls = [r.get("delta_ll") for r in subset if r.get("delta_ll") is not None]
                    mean_delta_ll = sum(delta_lls) / len(delta_lls) if delta_lls else 0.0

                    print(f"  θ={theta_reuse}, n={n_samples}, depth={max_depth}:")
                    print(f"    Signal detection rate (ΔLL>0.5): {detection_rate:.2%}")
                    print(f"    Recovery rate: {recovery_rate:.2%}")
                    print(f"    Mean ΔLL: {mean_delta_ll:.4f}")
                    print(f"    Mean bias: {mean_bias:.4f}")

    # Identify detectability boundary
    print("\nDetectability Boundary Analysis:")
    for n_samples in n_samples_list:
        for max_depth in max_depth_list:
            subset_all = [
                r
                for r in results
                if r.get("n_samples") == n_samples
                and r.get("max_depth") == max_depth
                and "error" not in r
            ]
            if subset_all:
                # Find minimum theta where detection rate > 50%
                detection_by_theta = {}
                for theta_reuse in theta_values:
                    subset = [r for r in subset_all if r.get("theta_reuse") == theta_reuse]
                    if subset:
                        detected = sum(1 for r in subset if r.get("signal_detected"))
                        detection_by_theta[theta_reuse] = detected / len(subset)

                min_detectable_theta = None
                for theta_reuse in sorted(detection_by_theta.keys()):
                    if detection_by_theta[theta_reuse] > 0.5:
                        min_detectable_theta = theta_reuse
                        break

                if min_detectable_theta:
                    print(
                        f"  n={n_samples}, depth={max_depth}: "
                        f"min detectable θ ≈ {min_detectable_theta}"
                    )
                else:
                    print(
                        f"  n={n_samples}, depth={max_depth}: "
                        f"no detection at tested θ values"
                    )


if __name__ == "__main__":
    main()
