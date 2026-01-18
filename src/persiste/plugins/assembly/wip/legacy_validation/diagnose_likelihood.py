#!/usr/bin/env python3
"""
Diagnostic: Check likelihood surface.

Simulates data with theta_true, then evaluates ΔLL at various theta values
to see if the likelihood surface has signal or is flat.

Usage:
    python diagnose_likelihood.py --theta-true 1.0 --theta-test-values 0.0,0.5,1.0,1.5,2.0 --n-samples 100
"""

import argparse

import persiste_rust

from persiste.plugins.assembly.cli import fit_assembly_constraints, InferenceMode


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose likelihood surface"
    )
    parser.add_argument(
        "--theta-true",
        type=float,
        default=1.0,
        help="True theta_reuse for data generation",
    )
    parser.add_argument(
        "--theta-test-values",
        type=str,
        default="0.0,0.5,1.0,1.5,2.0",
        help="Comma-separated theta values to evaluate",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of trajectories",
    )
    parser.add_argument(
        "--primitives",
        type=str,
        default="A,B",
        help="Comma-separated primitives",
    )

    args = parser.parse_args()

    theta_test_values = [float(x) for x in args.theta_test_values.split(",")]
    primitives = args.primitives.split(",")
    observed_compounds = set(primitives)

    print("Diagnosing likelihood surface...")
    print(f"  True theta_reuse: {args.theta_true}")
    print(f"  Test theta values: {theta_test_values}")
    print(f"  Samples: {args.n_samples}")

    # Generate data with theta_true
    print(f"\nGenerating {args.n_samples} trajectories with θ_true={args.theta_true}...")
    theta_true = {"reuse_count": args.theta_true}
    persiste_rust.simulate_assembly_trajectories(
        primitives=primitives,
        initial_parts=[primitives[0]],
        theta=theta_true,
        n_samples=args.n_samples,
        t_max=20.0,
        burn_in=5.0,
        max_depth=9,
        seed=42,
    )

    print("\nEvaluating likelihood at different theta values:")
    print("(Note: This uses the cached data from the simulator)")

    results = []
    for theta_test in theta_test_values:
        print(f"\n  θ_test = {theta_test}:")

        # Run inference at this theta (using cached data)
        result = fit_assembly_constraints(
            observed_compounds=observed_compounds,
            primitives=primitives,
            mode=InferenceMode.FULL_STOCHASTIC,
            feature_names=["reuse_count"],
            n_samples=args.n_samples,
            t_max=20.0,
            burn_in=5.0,
            max_depth=9,
            seed=42 + int(theta_test * 100),
        )

        stoch_ll = result.get("stochastic_ll")
        stoch_delta_ll = result.get("stochastic_delta_ll")
        theta_hat = result.get("theta_hat", {})

        print(f"    Absolute LL: {stoch_ll:.4f}")
        print(f"    ΔLL (vs null): {stoch_delta_ll:.4f}")
        print(f"    θ̂: {theta_hat}")

        results.append({
            "theta_test": theta_test,
            "absolute_ll": stoch_ll,
            "delta_ll": stoch_delta_ll,
            "theta_hat": theta_hat,
        })

    # Summary
    print("\n" + "=" * 60)
    print("Likelihood Surface Summary:")
    print("=" * 60)

    delta_lls = [r["delta_ll"] for r in results if r["delta_ll"] is not None]
    if delta_lls:
        max_delta_ll = max(delta_lls)
        min_delta_ll = min(delta_lls)
        range_delta_ll = max_delta_ll - min_delta_ll

        print(f"ΔLL range: {min_delta_ll:.4f} to {max_delta_ll:.4f}")
        print(f"ΔLL span: {range_delta_ll:.4f}")

        if range_delta_ll < 0.01:
            print("⚠️  FLAT SURFACE: ΔLL barely varies across theta values!")
            print("   This suggests the likelihood is not sensitive to theta.")
        else:
            print("✓ Surface has structure: ΔLL varies across theta values")

        # Find peak
        peak_idx = delta_lls.index(max_delta_ll)
        peak_theta = theta_test_values[peak_idx]
        print(f"\nPeak ΔLL at θ ≈ {peak_theta} (ΔLL = {max_delta_ll:.4f})")
        print(f"True θ = {args.theta_true}")

        if abs(peak_theta - args.theta_true) < 0.2:
            print("✓ Peak is near true theta (good!)")
        else:
            print("✗ Peak is far from true theta (potential issue)")


if __name__ == "__main__":
    main()
