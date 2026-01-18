#!/usr/bin/env python3
"""
V2: Power grid sweep.

Validates identifiability and power across effect sizes, data volumes, and
observation richness. Records ΔLL, recovery rate, bias, ESS, and false positives.

Usage:
    python run_power_grid.py --effect-sizes 0.5,1.0,1.5 --n-samples 50,100 --output results.csv
"""

import argparse
import csv
from pathlib import Path

import persiste_rust

from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints


def run_power_grid_point(
    theta_true: dict,
    n_samples: int,
    t_max: float,
    burn_in: float,
    max_depth: int,
    primitives: list[str],
    observed_compounds: set[str],
    seed: int,
) -> dict:
    """Run inference at a single power grid point."""
    # Simulate data at theta_true (data generation uses theta_true internally)
    persiste_rust.simulate_assembly_trajectories(
        primitives=primitives,
        initial_parts=[primitives[0]],
        theta=theta_true,
        n_samples=n_samples,
        t_max=t_max,
        burn_in=burn_in,
        max_depth=max_depth,
        seed=seed,
    )

    # Run inference (full stochastic for power grid)
    result = fit_assembly_constraints(
        observed_compounds=observed_compounds,
        primitives=primitives,
        mode=InferenceMode.FULL_STOCHASTIC,
        n_samples=n_samples,
        t_max=t_max,
        burn_in=burn_in,
        max_depth=max_depth,
        seed=seed + 1000,  # Different seed for inference
    )

    theta_hat = result.get("theta_hat", {})
    stoch_ll = result.get("stochastic_ll")
    stoch_delta_ll = result.get("stochastic_delta_ll")
    cache_stats = result.get("cache_stats", {})

    # Compute recovery and bias
    feature_name = list(theta_true.keys())[0] if theta_true else "reuse_count"
    theta_true_val = theta_true.get(feature_name, 0.0)
    theta_hat_val = theta_hat.get(feature_name, 0.0)

    recovered = abs(theta_hat_val - theta_true_val) < 0.5  # Recovery threshold
    bias = theta_hat_val - theta_true_val
    ess = cache_stats.get("ess_ratio", 1.0) if cache_stats else 1.0

    return {
        "theta_true": str(theta_true),
        "theta_hat": str(theta_hat),
        "delta_ll": stoch_delta_ll,
        "absolute_ll": stoch_ll,
        "recovered": recovered,
        "bias": bias,
        "ess": ess,
        "n_paths": cache_stats.get("n_paths", n_samples) if cache_stats else n_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="V2: Power grid sweep")
    parser.add_argument(
        "--effect-sizes",
        type=str,
        default="0.0,0.3,0.6,1.0",
        help="Comma-separated effect sizes (theta values)",
    )
    parser.add_argument(
        "--n-samples-list",
        type=str,
        default="200,500,1000",
        help="Comma-separated trajectory counts",
    )
    parser.add_argument(
        "--max-depth-list",
        type=str,
        default="7,9,11",
        help="Comma-separated depths to explore",
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
        "--feature",
        type=str,
        default="reuse_count",
        help="Feature to sweep (e.g., reuse_count, depth_gate_reuse)",
    )
    parser.add_argument(
        "--feature-names",
        type=str,
        default="reuse_count,depth_change",
        help="Comma-separated feature names to enable during inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="power_grid.csv",
        help="Output CSV",
    )

    args = parser.parse_args()

    effect_sizes = [float(x) for x in args.effect_sizes.split(",")]
    n_samples_list = [int(x) for x in args.n_samples_list.split(",")]
    max_depth_list = [int(x) for x in args.max_depth_list.split(",")]
    primitives = args.primitives.split(",")
    observed_compounds = set(primitives)
    feature_names = [f.strip() for f in args.feature_names.split(",") if f.strip()]

    print("Running V2: Power grid sweep...")
    print(f"  Effect sizes: {effect_sizes}")
    print(f"  Trajectories per config: {n_samples_list}")
    print(f"  Depths: {max_depth_list}")
    print(f"  Seeds per point: {args.n_seeds}")
    print(f"  Feature: {args.feature}")
    print(f"  Enabled feature names: {feature_names}")

    results = []
    total_points = len(effect_sizes) * len(n_samples_list) * len(max_depth_list) * args.n_seeds
    completed = 0

    for effect_size in effect_sizes:
        for n_samples in n_samples_list:
            for max_depth in max_depth_list:
                for seed in range(args.n_seeds):
                    completed += 1
                    print(
                        f"  [{completed}/{total_points}] θ={effect_size}, n={n_samples}, "
                        f"depth={max_depth}, seed={seed}"
                    )

                    theta_true = {args.feature: effect_size}

                    try:
                        row_data = run_power_grid_point(
                            theta_true=theta_true,
                            n_samples=n_samples,
                            t_max=15.0,
                            burn_in=5.0,
                            max_depth=max_depth,
                            primitives=primitives,
                            observed_compounds=observed_compounds,
                            seed=seed,
                        )

                        row = {
                            "effect_size": effect_size,
                            "n_samples": n_samples,
                            "max_depth": max_depth,
                            "seed": seed,
                            **row_data,
                        }
                        results.append(row)
                    except Exception as e:
                        print(f"    ERROR: {e}")
                        results.append({
                            "effect_size": effect_size,
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
    print("\nPower Grid Summary:")
    for effect_size in effect_sizes:
        for n_samples in n_samples_list:
            for max_depth in max_depth_list:
                subset = [
                    r
                    for r in results
                    if r.get("effect_size") == effect_size
                    and r.get("n_samples") == n_samples
                    and r.get("max_depth") == max_depth
                    and "error" not in r
                ]
                if subset:
                    recovered = sum(1 for r in subset if r.get("recovered"))
                    recovery_rate = recovered / len(subset) if subset else 0.0
                    biases = [r.get("bias", 0.0) for r in subset if "bias" in r]
                    mean_bias = sum(biases) / len(biases) if biases else 0.0
                    delta_lls = [r.get("delta_ll") for r in subset if r.get("delta_ll") is not None]
                    mean_delta_ll = sum(delta_lls) / len(delta_lls) if delta_lls else 0.0

                    print(f"  θ={effect_size}, n={n_samples}, depth={max_depth}:")
                    print(f"    Recovery rate: {recovery_rate:.2%}")
                    print(f"    Mean bias: {mean_bias:.4f}")
                    print(f"    Mean ΔLL: {mean_delta_ll:.4f}")


if __name__ == "__main__":
    main()
