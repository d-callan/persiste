"""
Recipe: Standard Assembly Analysis

Provides the official, supported workflow for inferring assembly constraints
from observed compounds. This is the recommended starting point for most users.
"""

import logging

from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_standard_analysis(
    observed_compounds: set[str],
    primitives: list[str],
    feature_names: list[str] | None = None,
    mode: InferenceMode = InferenceMode.FULL_STOCHASTIC,
    seed: int = 42
) -> dict:
    """
    Standard analysis workflow for Assembly Theory.

    This recipe:
    1. Fits the requested constraint features (default: reuse_count, depth_change).
    2. Runs Tier 1 safety checks (baseline sanity, identifiability).
    3. Returns a consolidated result dictionary with theta estimates and safety report.

    Args:
        observed_compounds: Set of identifiers for compounds observed in the system.
        primitives: List of primitive building blocks (e.g., ['A', 'B']).
        feature_names: List of features to infer (e.g., ['reuse_count']).
        mode: Inference mode (FULL_STOCHASTIC recommended for final results).
        seed: RNG seed for reproducibility.

    Returns:
        Result dictionary containing 'theta_hat', 'safety', and likelihood statistics.
    """
    if feature_names is None:
        feature_names = ["reuse_count", "depth_change"]

    logger.info(f"Starting standard assembly analysis with features: {feature_names}")

    result = fit_assembly_constraints(
        observed_compounds=observed_compounds,
        primitives=primitives,
        feature_names=feature_names,
        mode=mode,
        seed=seed
    )

    # Simple interpretation of results
    theta = result.get("theta_hat", {})
    safety = result.get("safety", {})

    print("\n" + "="*40)
    print("ASSEMBLY ANALYSIS SUMMARY")
    print("="*40)
    print(f"Inference Mode: {mode.value}")
    print("\nFitted Parameters (Theta):")
    for feat, val in theta.items():
        print(f"  {feat:20s}: {val:.3f}")

    print("\nSafety Report:")
    print(f"  Overall Status: {safety.get('overall_status', 'UNKNOWN').upper()}")
    if safety.get("overall_safe"):
        print("  ✓ Results are statistically reliable.")
    else:
        print("  ⚠ WARNING: Potential reliability issues detected.")
        for msg in safety.get("recommendations", []):
            print(f"    - {msg}")
    print("="*40 + "\n")

    return result

if __name__ == "__main__":
    # Quick test/demo of the recipe
    primitives = ["A", "B"]
    # Simulated observations: A, B, and a few complex ones
    observed = {"A", "B", "state_1", "state_2"}

    run_standard_analysis(observed, primitives)
