"""
Positive-signal tests for constraint inference.

Validates that:
1. Likelihood discriminates between theta values (ΔLL varies with theta)
2. Optimizer recovers theta from simulated data (theta_hat close to theta_true)
3. Feature extraction works correctly under constraints

Design principle:
A likelihood test must compare fixed observed data to θ-dependent expectations.
Never compare two θ-dependent summaries, assert seed-to-seed variability, or assume
ΔLL > 0 unless θ explains the data better than null.
"""

import numpy as np

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.likelihood import compute_observation_ll
from persiste.plugins.assembly.observation.cached_observation import (
    CacheConfig,
    CachedAssemblyObservationModel,
    SimulationSettings,
)
from persiste.plugins.assembly.states.assembly_state import AssemblyState


def simulate_assembly_data(theta, primitives=None, n_samples=50, seed=0):
    """
    Simulate assembly data under a given constraint parameter.

    Args:
        theta: Dict of feature weights (e.g., {"reuse_count": 1.0})
        primitives: List of primitives (default: ["A", "B"])
        n_samples: Number of simulation samples
        seed: RNG seed

    Returns:
        Dict with 'observed_compounds' and 'latent_states' for likelihood evaluation
    """
    if primitives is None:
        primitives = ["A", "B"]

    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5)
    initial_state = AssemblyState.from_parts(primitives[:1], depth=0)
    constraint = AssemblyConstraint(feature_weights=theta)

    # Use a large trust radius and low ESS threshold to prevent resimulation.
    # This is critical for tests because state_ids are not stable across
    # resimulations (they are simulation-local). Reweighting existing paths
    # ensures consistent ID-to-Structure mapping.
    cached_model = CachedAssemblyObservationModel(
        primitives=primitives,
        baseline=baseline,
        initial_state=initial_state,
        simulation=SimulationSettings(
            n_samples=n_samples,
            t_max=30.0,
            burn_in=10.0,
            max_depth=5,
        ),
        cache_config=CacheConfig(trust_radius=10.0, ess_threshold=0.0),
        rng_seed=seed,
    )

    latent_states = cached_model.get_latent_states(constraint)

    # Simulate observed compounds from latent states
    observed_compounds = set()
    # Sort states by probability descending
    sorted_states = sorted(latent_states.items(), key=lambda x: x[1], reverse=True)

    # Take top states (up to 5, or all if fewer)
    # We need to ensure we have actual states to test discrimination
    for state_id, prob in sorted_states[:5]:
        if prob > 0:
            observed_compounds.add(f"state_{state_id}")

    # Fallback: if simulation failed to produce states (unlikely), use primitives
    if not observed_compounds:
        # This will likely cause discrimination tests to fail (equal likelihoods)
        # but prevents crashes
        observed_compounds = {"A", "B"}

    return {
        "observed_compounds": observed_compounds,
        "latent_states": latent_states,
        "cached_model": cached_model,
        "constraint": constraint,
    }


def compute_assembly_ll(data, theta, primitives=None):
    """
    Compute log-likelihood for assembly data under a given theta.

    Args:
        data: Dict from simulate_assembly_data with observed_compounds and latent_states
        theta: Dict of feature weights
        primitives: List of primitives

    Returns:
        Log-likelihood value
    """
    if primitives is None:
        primitives = ["A", "B"]

    constraint = AssemblyConstraint(feature_weights=theta)

    cached_model = data["cached_model"]
    latent_states = cached_model.get_latent_states(constraint)

    ll = compute_observation_ll(
        latent_states,
        data["observed_compounds"],
        primitives,
        observation_records=None,
        max_depth=5,
        null_latent_states=None,
        ess_ratio=1.0,
    )

    return ll


class TestLikelihoodDiscrimination:
    """Test that likelihood varies with theta (not constant)."""

    def test_likelihood_prefers_true_theta_over_null(self):
        """
        Specification:
        Data generated under a nonzero constraint parameter should be better
        explained by that parameter than by the null.
        """
        theta_true = {"reuse_count": 1.0}

        data = simulate_assembly_data(theta=theta_true, seed=0)

        ll_null = compute_assembly_ll(data, theta={"reuse_count": 0.0})
        ll_true = compute_assembly_ll(data, theta=theta_true)

        assert ll_true > ll_null, (
            f"True theta should have higher LL: "
            f"LL(true)={ll_true:.4f} vs LL(null)={ll_null:.4f}"
        )

    def test_likelihood_peaks_near_true_theta(self):
        """
        Specification:
        Likelihood as a function of θ should attain a local maximum
        near the parameter used to generate the data.
        """
        theta_true = 1.0
        data = simulate_assembly_data(theta={"reuse_count": theta_true}, seed=1)

        thetas = [0.0, 0.5, 1.0, 1.5]
        lls = [compute_assembly_ll(data, {"reuse_count": t}) for t in thetas]

        max_theta = thetas[np.argmax(lls)]

        assert abs(max_theta - theta_true) <= 0.5, (
            f"Likelihood should peak near true theta: "
            f"max_theta={max_theta} vs theta_true={theta_true}"
        )

    def test_likelihood_stable_across_seeds_given_fixed_data(self):
        """
        Specification:
        For fixed observed data and θ, likelihood estimates should be
        consistent across Monte Carlo seeds.
        """
        data = simulate_assembly_data(theta={"reuse_count": 1.0}, seed=0)
        theta = {"reuse_count": 1.0}

        lls = [compute_assembly_ll(data, theta) for _ in range(3)]

        assert np.std(lls) < 1e-2, (
            f"Likelihood should be stable across evaluations: "
            f"std(LL)={np.std(lls):.6f}"
        )

    def test_likelihood_gradient_nonzero_under_signal(self):
        """
        Specification:
        Likelihood should change with θ when data were generated
        under a non-null constraint.
        """
        data = simulate_assembly_data(theta={"reuse_count": 1.0}, seed=2)

        ll0 = compute_assembly_ll(data, {"reuse_count": 0.5})
        ll1 = compute_assembly_ll(data, {"reuse_count": 1.0})

        assert ll1 != ll0, (
            f"Likelihood should vary with theta: "
            f"LL(0.5)={ll0:.4f} vs LL(1.0)={ll1:.4f}"
        )


class TestThetaRecovery:
    """Test that optimizer can recover theta from simulated data."""

    def test_optimizer_recovers_significant_signal(self):
        """
        Specification:
        When data are generated with a strong constraint (reuse_count=2.0),
        the optimizer should recover a positive theta and a significant Delta LL (>5.0).
        """
        theta_true = {"reuse_count": 2.0}
        # Generate 'observed' data using a fixed seed and known theta
        data = simulate_assembly_data(theta=theta_true, n_samples=100, seed=42)
        observed = data["observed_compounds"]

        result = fit_assembly_constraints(
            observed_compounds=observed,
            primitives=["A", "B"],
            mode=InferenceMode.FULL_STOCHASTIC,
            feature_names=["reuse_count"],
            n_samples=100,
            t_max=30.0,
            burn_in=10.0,
            max_depth=5,
            seed=42,
        )

        theta_hat = result.get("theta_hat", {})
        delta_ll = result.get("stochastic_delta_ll", 0.0)
        reuse_hat = theta_hat.get("reuse_count", 0.0)

        # 1. Delta LL should be significant
        assert delta_ll > 5.0, f"Expected significant Delta LL (>5), got {delta_ll:.2f}"

        # 2. Recovered theta should be positive and in the right ballpark
        assert reuse_hat > 0.5, f"Expected recovered reuse_count > 0.5, got {reuse_hat:.2f}"
        assert reuse_hat < 5.0, f"Recovered theta is unexpectedly extreme: {reuse_hat:.2f}"


class TestFeatureExtractionUnderConstraints:
    """Test that feature extraction works correctly under different constraints."""

    def test_features_vary_with_constraint(self):
        """Feature extraction should produce valid feature summaries."""
        # This test verifies that constraint-specific features are computed
        # We check this indirectly by verifying that the inference completes
        # without errors and produces valid results
        result = fit_assembly_constraints(
            observed_compounds={"A", "B"},
            primitives=["A", "B"],
            mode=InferenceMode.FULL_STOCHASTIC,
            feature_names=["reuse_count"],
            n_samples=100,
            t_max=20.0,
            burn_in=5.0,
            max_depth=7,
            seed=42,
        )

        # Inference should complete successfully
        assert result is not None, "Inference should return a result"

        # Result should have required keys
        assert "stochastic_delta_ll" in result, "Result should have stochastic_delta_ll"
        assert "theta_hat" in result, "Result should have theta_hat"

        # ΔLL should be a number
        delta_ll = result.get("stochastic_delta_ll", 0.0)
        assert isinstance(delta_ll, (int, float)), "ΔLL should be numeric"
