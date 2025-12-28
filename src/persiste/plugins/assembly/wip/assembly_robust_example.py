"""
Safe-by-Default Constraint Inference Example

Demonstrates the robust inference API with all safeguards:
- Joint baseline + constraint inference
- Automatic null testing
- L2 regularization
- Profile diagnostics
- Baseline sensitivity
- Cross-validation
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.baselines.baseline_family import SimpleBaselineFamily, FixedBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.observation.presence_model import FrequencyWeightedPresenceModel
from persiste.plugins.assembly.inference.robust_inference import RobustConstraintInference


def generate_test_data(
    primitives: list,
    theta_true: dict,
    baseline: AssemblyBaseline,
    n_samples: int = 100,
    seed: int = 42,
) -> dict:
    """Generate synthetic test data."""
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    constraint = AssemblyConstraint(feature_weights=theta_true)
    simulator = GillespieSimulator(graph, baseline, constraint, rng=np.random.default_rng(seed))
    
    observed_counts = {}
    for _ in range(n_samples):
        traj = simulator.simulate(
            AssemblyState.from_parts([primitives[0]], depth=0),
            t_max=50.0,
            burn_in=25.0
        )
        final_state = traj.final_state()
        for part in final_state.get_parts_list():
            if np.random.rand() < 0.9:
                observed_counts[part] = observed_counts.get(part, 0) + 1
    
    return observed_counts


def example_1_basic_usage():
    """Example 1: Basic usage with all safeguards."""
    print("=" * 80)
    print("Example 1: Basic Safe-by-Default Inference")
    print("=" * 80)
    
    # Setup
    primitives = ['A', 'B', 'C', 'D', 'E']
    theta_true = {'reuse_count': 1.5, 'depth_change': -0.5}
    baseline_true = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    
    # Generate data
    print("\n[Setup] Generating synthetic data...")
    print(f"  True constraints: {theta_true}")
    observed_counts = generate_test_data(primitives, theta_true, baseline_true, n_samples=100)
    print(f"  Observed: {len(observed_counts)} compounds, {sum(observed_counts.values())} detections")
    
    # Inference with joint baseline estimation
    print("\n[Inference] Using SimpleBaselineFamily (infers join_exponent)...")
    
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline_family = SimpleBaselineFamily(
        parameter='join_exponent',
        initial_value=-0.5,
        prior_std=0.2,  # Regularizes baseline
    )
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    
    inference = RobustConstraintInference(
        graph=graph,
        baseline_family=baseline_family,
        obs_model=obs_model,
        regularization=0.1,  # L2 penalty on constraints
    )
    
    # Fit with automatic null testing and diagnostics
    result = inference.fit_with_null(
        observed_counts,
        constraint_features=['reuse_count', 'depth_change'],
        profile_diagnostics=True,
        verbose=True,
    )
    
    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(result.get_recommendation())
    print()


def example_2_baseline_sensitivity():
    """Example 2: Baseline sensitivity analysis."""
    print("\n" + "=" * 80)
    print("Example 2: Baseline Sensitivity Analysis")
    print("=" * 80)
    
    # Setup
    primitives = ['A', 'B', 'C', 'D', 'E']
    theta_true = {'reuse_count': 1.5, 'depth_change': -0.5}
    baseline_true = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    
    # Generate data
    print("\n[Setup] Generating synthetic data...")
    observed_counts = generate_test_data(primitives, theta_true, baseline_true, n_samples=100)
    
    # Inference
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline_family = SimpleBaselineFamily(parameter='join_exponent', initial_value=-0.5)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    
    inference = RobustConstraintInference(
        graph=graph,
        baseline_family=baseline_family,
        obs_model=obs_model,
        regularization=0.1,
    )
    
    # Test sensitivity to baseline variations
    baseline_variations = [
        SimpleBaselineFamily(parameter='join_exponent', initial_value=-0.4),
        SimpleBaselineFamily(parameter='join_exponent', initial_value=-0.5),
        SimpleBaselineFamily(parameter='join_exponent', initial_value=-0.6),
        SimpleBaselineFamily(parameter='join_exponent', initial_value=-0.7),
    ]
    
    stability = inference.baseline_sensitivity_analysis(
        observed_counts,
        baseline_variations,
        constraint_features=['reuse_count', 'depth_change'],
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    for feature, std in stability.items():
        if std < 0.2:
            print(f"  {feature}: STABLE (±{std:.3f})")
        elif std < 0.5:
            print(f"  {feature}: MODERATE (±{std:.3f})")
        else:
            print(f"  {feature}: UNSTABLE (±{std:.3f}) - baseline matters!")
    print()


def example_3_cross_validation():
    """Example 3: Cross-validation."""
    print("\n" + "=" * 80)
    print("Example 3: Cross-Validation")
    print("=" * 80)
    
    # Setup
    primitives = ['A', 'B', 'C', 'D', 'E']
    theta_true = {'reuse_count': 1.5, 'depth_change': -0.5}
    baseline_true = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    
    # Generate data
    print("\n[Setup] Generating synthetic data...")
    observed_counts = generate_test_data(primitives, theta_true, baseline_true, n_samples=150)
    
    # Inference
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline_family = SimpleBaselineFamily(parameter='join_exponent', initial_value=-0.5)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    
    inference = RobustConstraintInference(
        graph=graph,
        baseline_family=baseline_family,
        obs_model=obs_model,
        regularization=0.1,
    )
    
    # Cross-validate
    cv_score = inference.cross_validate(
        observed_counts,
        constraint_features=['reuse_count', 'depth_change'],
        k=5,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if cv_score > -100:
        print(f"  ✓ Good generalization (CV score = {cv_score:.1f})")
    else:
        print(f"  ⚠ Poor generalization (CV score = {cv_score:.1f})")
        print(f"  → May be overfitting to training data")
    print()


def example_4_null_model():
    """Example 4: Null model (no constraints) - should not find spurious constraints."""
    print("\n" + "=" * 80)
    print("Example 4: Null Model Test (No Constraints)")
    print("=" * 80)
    
    # Setup
    primitives = ['A', 'B', 'C', 'D', 'E']
    theta_true = {}  # NULL MODEL
    baseline_true = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    
    # Generate data
    print("\n[Setup] Generating data from NULL model (no constraints)...")
    observed_counts = generate_test_data(primitives, theta_true, baseline_true, n_samples=100)
    
    # Inference with joint baseline
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline_family = SimpleBaselineFamily(parameter='join_exponent', initial_value=-0.5)
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    
    inference = RobustConstraintInference(
        graph=graph,
        baseline_family=baseline_family,
        obs_model=obs_model,
        regularization=0.1,  # Regularization helps suppress false positives
    )
    
    result = inference.fit_with_null(
        observed_counts,
        constraint_features=['reuse_count', 'depth_change'],
        profile_diagnostics=True,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("EXPECTED BEHAVIOR")
    print("=" * 60)
    print("  • Δ LL should be small (<5)")
    print("  • Evidence should be 'none' or 'weak'")
    print("  • Estimates should be near zero")
    print("\n" + "=" * 60)
    print("ACTUAL RESULT")
    print("=" * 60)
    print(f"  • Δ LL = {result.delta_ll:.2f}")
    print(f"  • Evidence = {result.evidence}")
    print(f"  • Estimates = {result.estimate}")
    
    if result.evidence in ['none', 'weak'] and result.delta_ll < 5:
        print("\n  ✓ CORRECT: No false positive detected")
    else:
        print("\n  ⚠ WARNING: Possible false positive")
        print("  → Increase regularization or use more conservative threshold")
    print()


def example_5_fixed_baseline():
    """Example 5: Fixed baseline (traditional approach)."""
    print("\n" + "=" * 80)
    print("Example 5: Fixed Baseline (Traditional Approach)")
    print("=" * 80)
    
    # Setup
    primitives = ['A', 'B', 'C', 'D', 'E']
    theta_true = {'reuse_count': 1.5, 'depth_change': -0.5}
    baseline_true = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    
    # Generate data
    print("\n[Setup] Generating synthetic data...")
    observed_counts = generate_test_data(primitives, theta_true, baseline_true, n_samples=100)
    
    # Inference with FIXED baseline (no joint estimation)
    print("\n[Inference] Using FixedBaseline (traditional approach)...")
    
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline_family = FixedBaseline(baseline_true)  # No baseline inference
    obs_model = FrequencyWeightedPresenceModel(detection_prob=0.9, false_positive_rate=0.5)
    
    inference = RobustConstraintInference(
        graph=graph,
        baseline_family=baseline_family,
        obs_model=obs_model,
        regularization=0.1,
    )
    
    result = inference.fit_with_null(
        observed_counts,
        constraint_features=['reuse_count', 'depth_change'],
        profile_diagnostics=True,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("NOTE")
    print("=" * 60)
    print("  Fixed baseline is faster but less robust.")
    print("  Use only when baseline is well-validated.")
    print("  Joint inference (SimpleBaselineFamily) is recommended.")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("ROBUST CONSTRAINT INFERENCE EXAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate the safe-by-default API:")
    print("  • Joint baseline + constraint inference")
    print("  • Automatic null testing")
    print("  • L2 regularization")
    print("  • Profile diagnostics")
    print("  • Baseline sensitivity")
    print("  • Cross-validation")
    print()
    
    # Run examples
    example_1_basic_usage()
    example_2_baseline_sensitivity()
    example_3_cross_validation()
    example_4_null_model()
    example_5_fixed_baseline()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nRecommended workflow:")
    print("  1. Use SimpleBaselineFamily (infer 1 baseline parameter)")
    print("  2. Use regularization=0.1 (soft sparsity)")
    print("  3. Call fit_with_null() for automatic diagnostics")
    print("  4. Check result.evidence and result.warnings")
    print("  5. Validate with baseline_sensitivity_analysis()")
    print("  6. Validate with cross_validate()")
    print("\nThis prevents false positives and ensures robust inference.")
    print("=" * 80)


if __name__ == '__main__':
    main()
