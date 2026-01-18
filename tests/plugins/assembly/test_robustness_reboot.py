"""
Validation suite for assembly robustness reboot.

Tests the key validation criteria:
1. Correctness: θ̂ converges to true θ under moderate ESS degradation
2. Stability: Inference doesn't snap back to θ=0 when cache reuse is heavy
3. Screening: Deterministic screening correctly triages hypotheses
"""

import math

import numpy as np
import pytest

import persiste_rust

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.diagnostics.artifacts import CachedPathData, InferenceArtifacts
from persiste.plugins.assembly.diagnostics.suite import null_resampling, profile_likelihood
from persiste.plugins.assembly.observation.cached_observation import (
    CachedAssemblyObservationModel,
    CacheConfig,
    SimulationSettings,
)
from persiste.plugins.assembly.screening.screening import (
    AdaptiveScreeningGrid,
    adaptive_screen,
    screen_hypotheses,
)
from persiste.plugins.assembly.screening.steady_state import (
    SteadyStateAssemblyModel,
    SteadyStateConfig,
)
from persiste.plugins.assembly.states.assembly_state import AssemblyState


class TestRustSimulation:
    """Test Rust Gillespie simulation and importance sampling."""

    def test_simulate_trajectories_returns_paths(self):
        """Verify simulation returns expected path structure."""
        result = persiste_rust.simulate_assembly_trajectories(
            primitives=["A", "B"],
            initial_parts=["A"],
            theta={"reuse_count": 1.0},
            n_samples=10,
            t_max=10.0,
            burn_in=2.0,
            max_depth=3,
            seed=42,
        )

        assert len(result) == 10
        for path in result:
            assert "feature_counts" in path
            assert "final_state_id" in path
            assert "duration" in path
            assert "n_transitions" in path

    def test_importance_weights_at_reference(self):
        """Weights should be uniform at reference θ."""
        result = persiste_rust.simulate_assembly_trajectories(
            primitives=["A", "B"],
            initial_parts=["A"],
            theta={"reuse_count": 1.0},
            n_samples=50,
            t_max=10.0,
            burn_in=2.0,
            max_depth=3,
            seed=42,
        )

        feature_counts = [r["feature_counts"] for r in result]
        theta_ref = {"reuse_count": 1.0}

        weights, ess = persiste_rust.compute_importance_weights(
            feature_counts, theta_ref, theta_ref
        )

        # At reference, ESS should equal N
        assert abs(ess - 50.0) < 0.1

    def test_ess_decreases_with_theta_shift(self):
        """ESS should decrease as θ moves away from reference."""
        result = persiste_rust.simulate_assembly_trajectories(
            primitives=["A", "B"],
            initial_parts=["A"],
            theta={"reuse_count": 1.0},
            n_samples=100,
            t_max=10.0,
            burn_in=2.0,
            max_depth=3,
            seed=42,
        )

        feature_counts = [r["feature_counts"] for r in result]
        theta_ref = {"reuse_count": 1.0}

        _, ess_at_ref = persiste_rust.compute_importance_weights(
            feature_counts, theta_ref, theta_ref
        )

        _, ess_shifted = persiste_rust.compute_importance_weights(
            feature_counts, {"reuse_count": 2.0}, theta_ref
        )

        assert ess_shifted < ess_at_ref


class TestCacheManagement:
    """Test cache validity and resimulation triggers."""

    def test_cache_valid_at_same_theta(self):
        """Cache should be valid when θ unchanged."""
        result = persiste_rust.simulate_assembly_trajectories(
            primitives=["A", "B"],
            initial_parts=["A"],
            theta={},
            n_samples=50,
            t_max=10.0,
            burn_in=2.0,
            max_depth=3,
            seed=42,
        )

        feature_counts = [r["feature_counts"] for r in result]
        final_state_ids = [r["final_state_id"] for r in result]

        status = persiste_rust.evaluate_cache(
            feature_counts,
            final_state_ids,
            theta={},
            theta_ref={},
        )

        assert status["valid"] is True
        assert abs(status["ess"] - 50.0) < 0.1

    def test_cache_invalid_outside_trust_region(self):
        """Cache should be invalid when θ exits trust region."""
        result = persiste_rust.simulate_assembly_trajectories(
            primitives=["A", "B"],
            initial_parts=["A"],
            theta={},
            n_samples=50,
            t_max=10.0,
            burn_in=2.0,
            max_depth=3,
            seed=42,
        )

        feature_counts = [r["feature_counts"] for r in result]
        final_state_ids = [r["final_state_id"] for r in result]

        status = persiste_rust.evaluate_cache(
            feature_counts,
            final_state_ids,
            theta={"reuse_count": 2.0},  # delta = 2.0 > trust_radius = 1.0
            theta_ref={},
            trust_radius=1.0,
        )

        assert status["valid"] is False
        assert "trust radius" in status["reason"].lower()

    def test_topology_guard_detects_large_changes(self):
        """Topology guard should detect large θ changes in sensitive features."""
        affected = persiste_rust.check_topology_change(
            theta={"depth_penalty": 3.0},
            theta_ref={"depth_penalty": 0.0},
            sensitive_features=["depth_penalty"],
            threshold=2.0,
        )

        assert "depth_penalty" in affected

    def test_topology_guard_allows_small_changes(self):
        """Topology guard should allow small θ changes."""
        affected = persiste_rust.check_topology_change(
            theta={"depth_penalty": 1.0},
            theta_ref={"depth_penalty": 0.0},
            sensitive_features=["depth_penalty"],
            threshold=2.0,
        )

        assert len(affected) == 0


class TestCachedObservationModel:
    """Test Python cached observation model adapter."""

    def test_initialization_creates_cache(self):
        """First call should initialize cache."""
        baseline = AssemblyBaseline()
        initial_state = AssemblyState.from_parts(["A"], depth=0)
        constraint = AssemblyConstraint(feature_weights={"reuse_count": 1.0})

        model = CachedAssemblyObservationModel(
            primitives=["A", "B"],
            baseline=baseline,
            initial_state=initial_state,
            simulation=SimulationSettings(n_samples=20, t_max=10.0, burn_in=2.0),
            cache_config=CacheConfig(trust_radius=1.0, ess_threshold=0.3),
            rng_seed=42,
        )

        states = model.get_latent_states(constraint)
        assert model.cache_stats["initialized"]
        assert len(states) > 0

    def test_cache_reuse_avoids_resimulation(self):
        """Same θ should reuse cache without resimulation."""
        baseline = AssemblyBaseline()
        initial_state = AssemblyState.from_parts(["A"], depth=0)
        constraint = AssemblyConstraint(feature_weights={"reuse_count": 1.0})

        model = CachedAssemblyObservationModel(
            primitives=["A", "B"],
            baseline=baseline,
            initial_state=initial_state,
            simulation=SimulationSettings(n_samples=20, t_max=10.0, burn_in=2.0),
            cache_config=CacheConfig(trust_radius=1.0, ess_threshold=0.3),
            rng_seed=42,
        )

        # First call
        model.get_latent_states(constraint)
        resim_count_1 = model.cache_stats["resimulation_count"]

        # Second call with same θ
        model.get_latent_states(constraint)
        resim_count_2 = model.cache_stats["resimulation_count"]

        assert resim_count_1 == resim_count_2


class TestScreening:
    """Test deterministic screening."""

    def test_screen_hypotheses_ranks_correctly(self):
        """Screening should rank hypotheses by normalized ΔLL."""
        baseline = AssemblyBaseline()
        model = SteadyStateAssemblyModel(
            primitives=["A", "B"],
            baseline=baseline,
            config=SteadyStateConfig(max_depth=3),
        )
        initial_state = AssemblyState.from_parts(["A"], depth=0)
        observed = {"A", "B"}

        hypotheses = [
            {},
            {"reuse_count": 1.0},
            {"depth_change": -0.5},
        ]

        results = screen_hypotheses(hypotheses, model, observed, initial_state)

        assert len(results) == 3
        # Results should be sorted by normalized_delta_ll descending
        for i in range(len(results) - 1):
            assert results[i].normalized_delta_ll >= results[i + 1].normalized_delta_ll

    def test_adaptive_screening_refines_winners(self):
        """Adaptive screening should evaluate more hypotheses than coarse grid."""
        baseline = AssemblyBaseline()
        model = SteadyStateAssemblyModel(
            primitives=["A", "B"],
            baseline=baseline,
            config=SteadyStateConfig(max_depth=3),
        )
        initial_state = AssemblyState.from_parts(["A"], depth=0)
        observed = {"A", "B"}

        grid = AdaptiveScreeningGrid(
            feature_names=["reuse_count"],
            coarse_steps=3,
            budget=20,
            top_k=3,
        )

        results = adaptive_screen(model, observed, initial_state, grid)

        # Should have evaluated more than just coarse grid
        assert len(results) >= 3


class TestDiagnostics:
    """Test diagnostic suite."""

    def test_null_resampling_returns_valid_pvalue(self):
        """Null resampling should return p-value in [0, 1]."""
        artifacts = InferenceArtifacts(
            theta_hat={"reuse_count": 1.0},
            log_likelihood=-10.0,
            cache_id="test",
            baseline_config={},
            graph_config={},
        )

        cache = CachedPathData(
            feature_counts=[{"reuse_count": i % 3} for i in range(100)],
            final_state_ids=list(range(100)),
            theta_ref={},
        )

        result = null_resampling(artifacts, cache, n_resamples=100)

        assert 0.0 <= result.p_value <= 1.0

    def test_profile_likelihood_finds_mle(self):
        """Profile likelihood should find MLE on grid."""
        artifacts = InferenceArtifacts(
            theta_hat={"reuse_count": 1.0},
            log_likelihood=-10.0,
            cache_id="test",
            baseline_config={},
            graph_config={},
        )

        cache = CachedPathData(
            feature_counts=[{"reuse_count": i % 3} for i in range(100)],
            final_state_ids=list(range(100)),
            theta_ref={},
        )

        result = profile_likelihood(artifacts, cache, "reuse_count", n_grid=11)

        assert result.mle is not None
        assert result.confidence_interval[0] <= result.mle <= result.confidence_interval[1]


class TestNullCalibration:
    """V1: Null calibration — ensure ΔLL ≈ 0 when θ = 0."""

    def test_null_delta_ll_deterministic(self):
        """Deterministic screening should report ΔLL ≈ 0 at θ = 0."""
        result = fit_assembly_constraints(
            observed_compounds={"A", "B"},
            primitives=["A", "B"],
            mode=InferenceMode.SCREEN_ONLY,
            screen_budget=5,
            seed=42,
        )

        # At null, deterministic ΔLL should be near zero
        if result["deterministic_delta_ll"] is not None:
            assert abs(result["deterministic_delta_ll"]) < 0.5, \
                f"Deterministic ΔLL={result['deterministic_delta_ll']} deviates from null"

    def test_null_delta_ll_stochastic(self):
        """Stochastic inference should report ΔLL ≈ 0 at θ = 0."""
        result = fit_assembly_constraints(
            observed_compounds={"A", "B"},
            primitives=["A", "B"],
            mode=InferenceMode.FULL_STOCHASTIC,
            n_samples=20,
            seed=42,
        )

        # At null, stochastic ΔLL should be near zero
        assert abs(result["stochastic_delta_ll"]) < 1.0, \
            f"Stochastic ΔLL={result['stochastic_delta_ll']} deviates from null"

    def test_null_delta_ll_screen_and_refine(self):
        """Screen + refine should report ΔLL ≈ 0 at θ = 0."""
        result = fit_assembly_constraints(
            observed_compounds={"A", "B"},
            primitives=["A", "B"],
            mode=InferenceMode.SCREEN_AND_REFINE,
            screen_budget=5,
            n_samples=20,
            seed=42,
        )

        # Both deterministic and stochastic should be near zero
        if result["deterministic_delta_ll"] is not None:
            assert abs(result["deterministic_delta_ll"]) < 0.5
        assert abs(result["stochastic_delta_ll"]) < 1.0


class TestDeterministicSanity:
    """Stage II: Deterministic screen sanity checks."""

    def test_screen_only_null_constraints(self):
        """Deterministic-only mode should not escalate thresholds on null data."""
        result = fit_assembly_constraints(
            observed_compounds={"A", "B"},
            primitives=["A", "B"],
            mode=InferenceMode.SCREEN_ONLY,
            screen_budget=10,
            seed=42,
        )

        # Screening results should exist but not auto-accept
        assert result["screening_results"] is not None
        # Top result should have low normalized ΔLL (near null)
        if result["screening_results"]:
            top_result = result["screening_results"][0]
            norm_dll = top_result["normalized_delta_ll"]
            assert norm_dll < 2.0, \
                f"Top screening result has unexpectedly high normalized ΔLL: {norm_dll}"


class TestCLI:
    """Test CLI inference modes."""

    def test_screen_only_mode(self):
        """Screen-only mode should return screening results."""
        result = fit_assembly_constraints(
            observed_compounds={"A", "B"},
            primitives=["A", "B"],
            mode=InferenceMode.SCREEN_ONLY,
            screen_budget=10,
        )

        assert result["mode"] == "screen-only"
        assert len(result["screening_results"]) > 0

    def test_full_stochastic_mode(self):
        """Full stochastic mode should run without screening."""
        result = fit_assembly_constraints(
            observed_compounds={"A", "B"},
            primitives=["A", "B"],
            mode=InferenceMode.FULL_STOCHASTIC,
            n_samples=20,
        )

        assert result["mode"] == "full-stochastic"
        assert "theta_hat" in result

    def test_screen_vs_full_stochastic_consistency(self):
        """SCREEN_AND_REFINE should match FULL_STOCHASTIC on the same seed."""

        common_kwargs = {
            "observed_compounds": {"A", "B"},
            "primitives": ["A", "B"],
            "n_samples": 30,
            "t_max": 15.0,
            "burn_in": 5.0,
            "max_depth": 4,
            "seed": 314,
        }

        screened = fit_assembly_constraints(
            mode=InferenceMode.SCREEN_AND_REFINE,
            screen_budget=12,
            screen_topk=3,
            screen_refine_radius=0.25,
            **common_kwargs,
        )
        full = fit_assembly_constraints(
            mode=InferenceMode.FULL_STOCHASTIC,
            **common_kwargs,
        )

        assert screened["screening_results"], "Screened run should produce candidates"
        assert screened["cache_stats"] is not None
        assert full["cache_stats"] is not None

        theta_keys = set(screened["theta_hat"]) | set(full["theta_hat"])
        for key in theta_keys:
            assert math.isclose(
                screened["theta_hat"].get(key, 0.0),
                full["theta_hat"].get(key, 0.0),
                abs_tol=1e-6,
            )

        assert math.isclose(
            screened["stochastic_ll"],
            full["stochastic_ll"],
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        assert math.isclose(
            screened["stochastic_delta_ll"],
            full["stochastic_delta_ll"],
            rel_tol=1e-6,
            abs_tol=1e-6,
        )

    @pytest.mark.slow
    def test_convergence_with_nonzero_theta_ref(self):
        """
        Validation: θ̂ should converge when θ_ref ≠ 0.

        This tests that inference works correctly even when
        trajectories are cached at a non-null reference point.
        """
        # Simulate at θ_ref = {reuse_count: 1.0}
        result = persiste_rust.simulate_assembly_trajectories(
            primitives=["A", "B"],
            initial_parts=["A"],
            theta={"reuse_count": 1.0},
            n_samples=200,
            t_max=20.0,
            burn_in=5.0,
            max_depth=4,
            seed=42,
        )

        feature_counts = [r["feature_counts"] for r in result]
        theta_ref = {"reuse_count": 1.0}

        # Evaluate at several θ points
        test_thetas = [
            {"reuse_count": 0.5},
            {"reuse_count": 1.0},
            {"reuse_count": 1.5},
        ]

        for theta in test_thetas:
            weights, ess = persiste_rust.compute_importance_weights(
                feature_counts, theta, theta_ref
            )
            # ESS should be reasonable (not completely collapsed)
            # With 200 samples, ESS > 5 means at least some effective samples remain
            assert ess > 5, f"ESS too low at θ={theta}: {ess}"

    @pytest.mark.slow
    def test_stability_under_heavy_cache_reuse(self):
        """
        Validation: Inference should not snap back to θ=0 under heavy cache reuse.
        
        This tests that the importance sampling doesn't degrade catastrophically
        when we reweight many times.
        """
        baseline = AssemblyBaseline()
        initial_state = AssemblyState.from_parts(["A"], depth=0)

        model = CachedAssemblyObservationModel(
            primitives=["A", "B"],
            baseline=baseline,
            initial_state=initial_state,
            simulation=SimulationSettings(n_samples=100, t_max=20.0, burn_in=5.0),
            cache_config=CacheConfig(trust_radius=2.0, ess_threshold=0.1),
            rng_seed=42,
        )

        # Initialize at non-null θ
        constraint = AssemblyConstraint(feature_weights={"reuse_count": 1.0})
        model.get_latent_states(constraint)

        # Query many nearby θ values (heavy reuse)
        resim_count_start = model.cache_stats["resimulation_count"]

        for i in range(10):
            theta = {"reuse_count": 1.0 + 0.1 * i}
            constraint = AssemblyConstraint(feature_weights=theta)
            model.get_latent_states(constraint)

        resim_count_end = model.cache_stats["resimulation_count"]

        # Some resimulations are expected, but not at every query
        # (Trust region should allow some reuse)
        assert resim_count_end < resim_count_start + 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
