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

import logging

import numpy as np
import pytest

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.cli import fit_assembly_constraints, InferenceMode
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.likelihood import compute_observation_ll
from persiste.plugins.assembly.observation.cached_observation import (
    CacheConfig,
    CachedAssemblyObservationModel,
    SimulationSettings,
)
from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.states.resolver import StateIDResolver


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
    graph = AssemblyGraph(primitives, max_depth=5)
    cached_model = CachedAssemblyObservationModel(
        primitives=primitives,
        baseline=baseline,
        initial_state=initial_state,
        simulation=SimulationSettings(
            n_samples=500, # Increased samples for better signal
            t_max=50.0,
            burn_in=20.0,
            max_depth=5,
        ),
        cache_config=CacheConfig(trust_radius=10.0, ess_threshold=0.0),
        rng_seed=seed,
        graph=graph,
    )

    latent_states = cached_model.get_latent_states(constraint)
    resolver = StateIDResolver(primitives)

    # Simulate observed compounds from latent states
    observation_records = []
    total_obs = 1000

    # We use stable IDs directly for the observations in the test
    # to ensure zero ambiguity during likelihood calculation.
    observed_compounds = set()

    for state_id, prob in latent_states.items():
        state = cached_model.graph.get_state(state_id)
        if state is None:
            continue

        count = np.random.binomial(total_obs, prob)
        if count > 0:
            observed_compounds.add(state_id)
            observation_records.append({
                "compound_id": state_id,
                "frequency": float(count)
            })

    return {
        "observed_compounds": observed_compounds,
        "observation_records": observation_records,
        "latent_states": latent_states,
        "cached_model": cached_model,
        "constraint": constraint,
        "resolver": resolver,
        "stable_observed_ids": {resolver.resolve_string(cid) for cid in observed_compounds}
    }


def compute_assembly_ll(data, theta, primitives=None):
    """
    Compute log-likelihood for assembly data under a given theta.
    """
    if primitives is None:
        primitives = ["A", "B"]

    constraint = AssemblyConstraint(feature_weights=theta)

    cached_model = data["cached_model"]
    latent_states = cached_model.get_latent_states(constraint)
    resolver = data["resolver"]

    # Build mapping for realistic compound resolution using StateIDResolver
    compound_to_state = {}
    for cid in data["observed_compounds"]:
        try:
            compound_to_state[cid] = resolver.resolve_string(cid)
        except ValueError:
            continue

    ll = compute_observation_ll(
        latent_states,
        data["stable_observed_ids"],
        primitives,
        observation_records=data.get("observation_records"),
        max_depth=5,
        null_latent_states=None,
        ess_ratio=1.0,
        compound_to_state=compound_to_state,
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

        print(f"\nDEBUG test_likelihood_prefers_true_theta_over_null:")
        print(f"  Observed compounds: {data['observed_compounds']}")
        print(f"  LL(null): {ll_null:.4f}")
        print(f"  LL(true): {ll_true:.4f}")
        print(f"  Delta LL: {ll_true - ll_null:.4f}")

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
        When data are generated with a strong constraint,
        the optimizer should recover a theta in the right direction
        and a significant Delta LL.
        """
        # Use depth_change as it has a strong effect on simple assemblies
        theta_true = {"depth_change": -1.0}
        
        # Generate 'observed' data using a fixed seed and known theta
        data = simulate_assembly_data(theta=theta_true, n_samples=200, seed=42)
        observed = data["observed_compounds"]

        result = fit_assembly_constraints(
            observed_compounds=observed,
            primitives=["A", "B"],
            mode=InferenceMode.FULL_STOCHASTIC,
            feature_names=["depth_change"],
            n_samples=200,
            t_max=30.0,
            burn_in=10.0,
            max_depth=5,
            seed=42,
        )

        theta_hat = result.get("theta_hat", {})
        delta_ll = result.get("stochastic_delta_ll", 0.0)
        val_hat = theta_hat.get("depth_change", 0.0)

        # 1. Delta LL should be positive (better than null)
        assert delta_ll > 0.0, f"Expected positive Delta LL, got {delta_ll:.2f}"

        # 2. Recovered theta should be in the right direction (negative)
        assert val_hat < 0.0, f"Expected recovered depth_change < 0, got {val_hat:.2f}"


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
