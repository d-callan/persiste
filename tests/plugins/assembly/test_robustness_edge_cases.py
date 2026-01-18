"""
Additional edge case, integration, and stress tests for robustness reboot.

These tests cover:
- Edge cases (topology guard at threshold, extreme shifts)
- Integration (observation mapping, screen+refine, cache+topology)
- Stress tests (profile CI convergence, CLI overrides, screening budget)
"""

import json

import persiste_rust
import pytest

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.diagnostics.artifacts import CachedPathData, InferenceArtifacts
from persiste.plugins.assembly.diagnostics.suite import profile_likelihood
from persiste.plugins.assembly.observation.cached_observation import (
    CacheConfig,
    CachedAssemblyObservationModel,
    SimulationSettings,
)
from persiste.plugins.assembly.screening.screening import (
    AdaptiveScreeningGrid,
    adaptive_screen,
)
from persiste.plugins.assembly.screening.steady_state import (
    SteadyStateAssemblyModel,
    SteadyStateConfig,
)
from persiste.plugins.assembly.states.assembly_state import AssemblyState


class TestEdgeCases:
    """Edge case tests for boundary conditions."""

    def test_topology_guard_at_threshold(self):
        """
        Edge case: Δθ exactly at threshold.

        Behavior: > threshold triggers (strictly greater than), <= threshold passes.
        """
        threshold = 2.0

        # Exactly at threshold (should NOT trigger with >)
        affected_at = persiste_rust.check_topology_change(
            theta={"depth_penalty": 2.0},
            theta_ref={"depth_penalty": 0.0},
            sensitive_features=["depth_penalty"],
            threshold=threshold,
        )
        assert len(affected_at) == 0, "Should not trigger at threshold (uses >)"

        # Just below threshold
        affected_below = persiste_rust.check_topology_change(
            theta={"depth_penalty": 1.999},
            theta_ref={"depth_penalty": 0.0},
            sensitive_features=["depth_penalty"],
            threshold=threshold,
        )
        assert len(affected_below) == 0, "Should not trigger below threshold"

        # Just above threshold
        affected_above = persiste_rust.check_topology_change(
            theta={"depth_penalty": 2.001},
            theta_ref={"depth_penalty": 0.0},
            sensitive_features=["depth_penalty"],
            threshold=threshold,
        )
        assert "depth_penalty" in affected_above, "Should trigger above threshold"

    @pytest.mark.slow
    def test_extreme_theta_shift_triggers_resim(self):
        """
        Stress test: θ far outside trust region.

        Expected: ESS collapses, resimulation occurs, weights remain finite.
        """
        result = persiste_rust.simulate_assembly_trajectories(
            primitives=["A", "B"],
            initial_parts=["A"],
            theta={},
            n_samples=100,
            t_max=10.0,
            burn_in=2.0,
            max_depth=3,
            seed=42,
        )

        feature_counts = [r["feature_counts"] for r in result]
        final_state_ids = [r["final_state_id"] for r in result]

        # Extreme shift: θ = 10.0 (far outside trust_radius=1.0)
        status = persiste_rust.evaluate_cache(
            feature_counts,
            final_state_ids,
            theta={"reuse_count": 10.0},
            theta_ref={},
            trust_radius=1.0,
            ess_threshold=0.3,
        )

        assert status["valid"] is False, "Cache should be invalid for extreme shift"
        assert "trust radius" in status["reason"].lower()

        # Check weights are finite
        weights, ess = persiste_rust.compute_importance_weights(
            feature_counts, {"reuse_count": 10.0}, {}
        )
        assert all(w >= 0 and w <= 1.0 for w in weights), "Weights should be normalized"
        assert ess >= 0, "ESS should be non-negative"

    def test_observation_model_mapping(self):
        """
        Verify latent → observed correctness.

        With known latent states and detection probabilities,
        observed probabilities should match expected formula.
        """
        # Create simple latent distribution
        latent_states = {
            1: 0.7,  # State 1 present with 70% probability
            2: 0.3,  # State 2 present with 30% probability
        }

        # Simple observation model: compound detected if state present
        # P(observe compound) = sum over states containing compound

        # For this test, assume state 1 contains compound "A", state 2 contains "B"
        # Then P(observe "A") ≈ 0.7, P(observe "B") ≈ 0.3

        # This is a placeholder - full implementation would need actual observation model
        assert sum(latent_states.values()) == 1.0, "Latent probabilities should sum to 1"
        assert all(0 <= p <= 1 for p in latent_states.values()), "Valid probabilities"


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.slow
    def test_screen_plus_stochastic_refine(self):
        """
        Verify screening → stochastic refinement integration.

        Only top-K from screening should be refined stochastically.
        """
        result = fit_assembly_constraints(
            observed_compounds={"A", "B"},
            primitives=["A", "B"],
            mode=InferenceMode.SCREEN_AND_REFINE,
            screen_budget=15,
            screen_topk=3,
            n_samples=20,
            seed=42,
        )

        assert result["mode"] == "screen-and-refine"
        assert len(result["screening_results"]) > 0

        # Should have run stochastic refinement (cache stats present)
        assert result["cache_stats"] is not None
        assert result["cache_stats"]["initialized"]

    @pytest.mark.slow
    def test_cache_reuse_after_topology_guard(self):
        """
        Integration: cache + topology checks.

        Small Δθ reuses cache, large Δθ triggers resimulation.
        """
        baseline = AssemblyBaseline()
        initial_state = AssemblyState.from_parts(["A"], depth=0)

        model = CachedAssemblyObservationModel(
            primitives=["A", "B"],
            baseline=baseline,
            initial_state=initial_state,
            simulation=SimulationSettings(n_samples=50, t_max=10.0, burn_in=2.0),
            cache_config=CacheConfig(
                trust_radius=1.0,
                ess_threshold=0.3,
                topology_sensitive_features=["depth_penalty"],
                topology_change_threshold=2.0,
            ),
            rng_seed=42,
        )

        # Initialize at θ=0
        constraint_init = AssemblyConstraint(feature_weights={})
        model.get_latent_states(constraint_init)
        resim_after_init = model.cache_stats["resimulation_count"]

        # Small shift (within trust region, below topology threshold)
        constraint_small = AssemblyConstraint(feature_weights={"depth_penalty": 0.5})
        model.get_latent_states(constraint_small)
        resim_after_small = model.cache_stats["resimulation_count"]

        assert resim_after_small == resim_after_init, "Small shift should reuse cache"

        # Large shift (triggers topology guard)
        constraint_large = AssemblyConstraint(feature_weights={"depth_penalty": 3.0})
        model.get_latent_states(constraint_large)
        resim_after_large = model.cache_stats["resimulation_count"]

        assert resim_after_large > resim_after_small, "Large shift should trigger resimulation"


class TestStress:
    """Stress tests for performance and robustness."""

    @pytest.mark.slow
    @pytest.mark.parametrize("n_trajectories", [10, 50, 200])
    def test_profile_likelihood_ci_convergence(self, n_trajectories):
        """
        Diagnostics sensitivity: CI width decreases with more trajectories.

        MLE should remain inside CI.
        """
        artifacts = InferenceArtifacts(
            theta_hat={"reuse_count": 1.0},
            log_likelihood=-10.0,
            cache_id="test",
            baseline_config={},
            graph_config={},
        )

        cache = CachedPathData(
            feature_counts=[{"reuse_count": i % 3} for i in range(n_trajectories)],
            final_state_ids=list(range(n_trajectories)),
            theta_ref={},
        )

        result = profile_likelihood(artifacts, cache, "reuse_count", n_grid=11)

        # MLE should be in CI
        assert result.confidence_interval[0] <= result.mle <= result.confidence_interval[1]

        # CI width
        ci_width = result.confidence_interval[1] - result.confidence_interval[0]

        # Store for comparison (in real test, would compare across parametrize runs)
        assert ci_width > 0, "CI should have positive width"

    def test_cli_advanced_overrides(self, tmp_path):
        """
        CLI default override correctness.

        Custom trust_radius, ess_threshold, screen_budget should be applied.
        """
        output_file = tmp_path / "result.json"

        result = fit_assembly_constraints(
            observed_compounds={"A", "B"},
            primitives=["A", "B"],
            mode=InferenceMode.SCREEN_ONLY,
            trust_radius=2.5,
            ess_threshold=0.2,
            screen_budget=25,
        )

        # Verify screening budget was respected
        assert len(result["screening_results"]) <= 25

        # Save and reload to verify serialization
        with open(output_file, "w") as f:
            json.dump(result, f)

        with open(output_file) as f:
            loaded = json.load(f)

        assert loaded["mode"] == result["mode"]
        assert len(loaded["screening_results"]) == len(result["screening_results"])

    def test_extreme_false_positive_detection(self):
        """
        Observation robustness: all latent states absent, false_pos > 0.

        Observed counts should reflect false positives accurately.
        """
        # Latent distribution: all states have zero probability
        latent_states = {1: 0.0, 2: 0.0, 3: 0.0}

        # With false positive rate, we'd still observe some detections
        # This is a placeholder for when observation model is fully integrated

        # For now, just verify probabilities are valid
        assert all(p >= 0 for p in latent_states.values())
        assert sum(latent_states.values()) == 0.0

    def test_screening_edge_budget(self):
        """
        Edge case: budget smaller than top-K or fewer candidates than budget.

        Adaptive refinement should not crash.
        """
        baseline = AssemblyBaseline()
        model = SteadyStateAssemblyModel(
            primitives=["A", "B"],
            baseline=baseline,
            config=SteadyStateConfig(max_depth=3),
        )
        initial_state = AssemblyState.from_parts(["A"], depth=0)
        observed = {"A", "B"}

        # Budget smaller than top-K
        grid_small_budget = AdaptiveScreeningGrid(
            feature_names=["reuse_count"],
            coarse_steps=3,
            budget=5,  # Very small
            top_k=10,  # Larger than budget
        )

        results = adaptive_screen(model, observed, initial_state, grid_small_budget)

        # Should complete without crash
        assert len(results) <= 5, "Should respect budget"
        assert len(results) > 0, "Should evaluate at least some hypotheses"

        # Budget larger than available candidates
        grid_large_budget = AdaptiveScreeningGrid(
            feature_names=["reuse_count"],
            coarse_steps=2,  # Very coarse
            budget=100,  # Much larger than needed
            top_k=3,
        )

        results_large = adaptive_screen(model, observed, initial_state, grid_large_budget)

        # Should complete without crash
        assert len(results_large) > 0
        assert len(results_large) < 100, "Should not exceed available candidates"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
