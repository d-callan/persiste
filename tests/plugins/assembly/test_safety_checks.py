"""
Tests for Tier 1 safety checks.

Tests the safety/ module including baseline sanity, identifiability,
cache reliability, and consolidated safety report.
"""

import pytest

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.safety import (
    BaselineSanityResult,
    CacheReliabilityResult,
    IdentifiabilityResult,
    SafetyReport,
    check_baseline_sanity,
    check_cache_reliability,
    check_identifiability,
    run_safety_checks,
)
from persiste.plugins.assembly.screening.screening import ScreeningResult
from persiste.plugins.assembly.states.assembly_state import AssemblyState


class TestBaselineSanityCheck:
    """Tests for baseline sanity check."""

    @pytest.fixture
    def baseline(self):
        return AssemblyBaseline(
            kappa=1.0,
            join_exponent=-0.5,
            split_exponent=0.3,
            decay_rate=0.01,
        )

    @pytest.fixture
    def initial_state(self):
        return AssemblyState.from_parts(["A"], depth=0)

    def test_baseline_ok_with_matching_observations(self, baseline, initial_state):
        """Baseline should pass when observations match expectations."""
        # Observed compounds similar to what baseline would produce
        observed = {"A", "B"}
        primitives = ["A", "B"]

        result = check_baseline_sanity(
            observed_compounds=observed,
            primitives=primitives,
            baseline=baseline,
            initial_state=initial_state,
            n_samples=50,
            max_depth=3,
            seed=42,
        )

        assert isinstance(result, BaselineSanityResult)
        assert result.warning_level in ("none", "mild", "severe")
        assert result.delta_ll_multiplier >= 1.0
        assert "observed_summary" in result.to_dict()
        assert "expected_summary" in result.to_dict()

    def test_baseline_warning_with_divergent_observations(self, baseline, initial_state):
        """Baseline should warn when observations diverge significantly."""
        # Many more compounds than baseline would produce (simulated mismatch)
        observed = {f"compound_{i}" for i in range(100)}
        primitives = ["A", "B"]

        result = check_baseline_sanity(
            observed_compounds=observed,
            primitives=primitives,
            baseline=baseline,
            initial_state=initial_state,
            n_samples=50,
            max_depth=3,
            seed=42,
        )

        # 1. Should detect divergence
        msg = f"Expected high divergence, got {result.divergence_score:.2f}"
        assert result.divergence_score > 1.0, msg

        # 2. Multiplier should be elevated (mild=2.0 or severe=3.0)
        assert result.delta_ll_multiplier > 1.0, "Multiplier should be boosted for divergent data"
        assert result.warning_level in ("mild", "severe")
        assert not result.baseline_ok

    def test_multiplier_values(self, baseline, initial_state):
        """Test that multipliers are correctly assigned."""
        observed = {"A"}
        primitives = ["A", "B"]

        result = check_baseline_sanity(
            observed_compounds=observed,
            primitives=primitives,
            baseline=baseline,
            initial_state=initial_state,
            n_samples=20,
            max_depth=2,
            seed=42,
        )

        # Multiplier should be one of the defined values
        assert result.delta_ll_multiplier in (1.0, 2.0, 3.0)


class TestIdentifiabilityCheck:
    """Tests for identifiability screen."""

    def test_identifiable_with_good_screening(self):
        """Constraints should be identifiable with varied screening results."""
        screening_results = [
            ScreeningResult(
                theta={"reuse_count": 1.0},
                delta_ll=5.0,
                normalized_delta_ll=5.0,
                passed=True,
                rank=1,
            ),
            ScreeningResult(
                theta={"reuse_count": 0.5},
                delta_ll=3.0,
                normalized_delta_ll=3.0,
                passed=True,
                rank=2,
            ),
            ScreeningResult(
                theta={"reuse_count": 0.0},
                delta_ll=1.0,
                normalized_delta_ll=1.0,
                passed=False,
                rank=3,
            ),
        ]
        theta_hat = {"reuse_count": 1.0}

        result = check_identifiability(screening_results, theta_hat)

        assert isinstance(result, IdentifiabilityResult)
        assert result.status in ("ok", "flat", "collapse_to_null")
        assert "screening_variance" in result.evidence
        assert result.recommendation

    def test_collapse_to_null_detection(self):
        """Should detect collapse to null when θ̂ near zero and weak signal."""
        screening_results = [
            ScreeningResult(
                theta={"reuse_count": 0.0},
                delta_ll=0.5,
                normalized_delta_ll=0.5,
                passed=False,
                rank=1,
            ),
        ]
        theta_hat = {"reuse_count": 0.01}

        result = check_identifiability(screening_results, theta_hat)

        assert result.status == "collapse_to_null"
        assert not result.identifiable
        assert "do not interpret" in result.recommendation.lower()

    def test_flat_likelihood_detection(self):
        """Should detect flat likelihood when variance is low."""
        # All screening results have similar ΔLL
        screening_results = [
            ScreeningResult(
                theta={"reuse_count": 1.0},
                delta_ll=2.1,
                normalized_delta_ll=2.1,
                passed=True,
                rank=1,
            ),
            ScreeningResult(
                theta={"reuse_count": 0.5},
                delta_ll=2.0,
                normalized_delta_ll=2.0,
                passed=True,
                rank=2,
            ),
            ScreeningResult(
                theta={"reuse_count": 0.0},
                delta_ll=1.9,
                normalized_delta_ll=1.9,
                passed=False,
                rank=3,
            ),
        ]
        theta_hat = {"reuse_count": 1.0}

        result = check_identifiability(screening_results, theta_hat)

        # Low variance + low curvature = flat
        assert result.evidence["screening_variance"] < 1.0

    def test_empty_screening_results(self):
        """Should handle empty screening results gracefully."""
        result = check_identifiability([], {"reuse_count": 0.5})

        assert result.identifiable  # Conservative default
        assert "unknown" in result.message.lower()


class TestCacheReliabilityCheck:
    """Tests for cache reliability check."""

    def test_ok_with_good_ess(self):
        """Should be OK with high ESS."""
        cache_stats = {
            "ess_at_theta_hat": 80.0,
            "n_paths": 100,
            "resimulation_count": 0,
        }

        result = check_cache_reliability(cache_stats)

        assert isinstance(result, CacheReliabilityResult)
        assert result.status == "ok"
        assert result.inference_stable
        assert result.ess_ratio == 0.8

    def test_warning_with_low_ess(self):
        """Should warn with low ESS."""
        cache_stats = {
            "ess_at_theta_hat": 25.0,
            "n_paths": 100,
            "resimulation_count": 0,
        }

        result = check_cache_reliability(cache_stats, ess_threshold=0.3)

        assert result.status == "warning"
        assert not result.inference_stable

    def test_severe_with_very_low_ess(self):
        """Should be severe with very low ESS."""
        cache_stats = {
            "ess_at_theta_hat": 10.0,
            "n_paths": 100,
            "resimulation_count": 0,
        }

        result = check_cache_reliability(cache_stats)

        assert result.status == "severe"
        assert not result.inference_stable

    def test_warning_with_excessive_resimulations(self):
        """Should warn with many resimulations."""
        cache_stats = {
            "ess_at_theta_hat": 80.0,
            "n_paths": 100,
            "resimulation_count": 10,
        }

        result = check_cache_reliability(cache_stats)

        assert result.status == "warning"
        assert "resimulation" in result.message.lower()

    def test_handles_missing_keys(self):
        """Should handle missing cache_stats keys."""
        result = check_cache_reliability({})

        assert isinstance(result, CacheReliabilityResult)
        # Should use defaults


class TestSafetyReport:
    """Tests for consolidated safety report."""

    @pytest.fixture
    def baseline(self):
        return AssemblyBaseline(
            kappa=1.0,
            join_exponent=-0.5,
            split_exponent=0.3,
            decay_rate=0.01,
        )

    @pytest.fixture
    def initial_state(self):
        return AssemblyState.from_parts(["A"], depth=0)

    def test_run_safety_checks_returns_report(self, baseline, initial_state):
        """run_safety_checks should return a SafetyReport."""
        screening_results = [
            ScreeningResult(
                theta={"reuse_count": 1.0},
                delta_ll=5.0,
                normalized_delta_ll=5.0,
                passed=True,
                rank=1,
            ),
        ]

        report = run_safety_checks(
            observed_compounds={"A", "B"},
            primitives=["A", "B"],
            baseline=baseline,
            initial_state=initial_state,
            theta_hat={"reuse_count": 1.0},
            screening_results=screening_results,
            cache_stats={"ess_at_theta_hat": 80.0, "n_paths": 100},
            n_baseline_samples=20,
            max_depth=3,
            seed=42,
        )

        assert isinstance(report, SafetyReport)
        assert report.overall_status in ("ok", "warning", "unsafe")
        assert isinstance(report.baseline_check, BaselineSanityResult)
        assert isinstance(report.identifiability_check, IdentifiabilityResult)
        assert isinstance(report.cache_check, CacheReliabilityResult)

    def test_overall_status_ok_when_all_pass(self, baseline, initial_state):
        """Overall status should be OK when all checks pass."""
        screening_results = [
            ScreeningResult(
                theta={"reuse_count": 1.0},
                delta_ll=5.0,
                normalized_delta_ll=5.0,
                passed=True,
                rank=1,
            ),
            ScreeningResult(
                theta={"reuse_count": 0.5},
                delta_ll=3.0,
                normalized_delta_ll=3.0,
                passed=True,
                rank=2,
            ),
            ScreeningResult(
                theta={"reuse_count": 0.0},
                delta_ll=1.0,
                normalized_delta_ll=1.0,
                passed=False,
                rank=3,
            ),
        ]

        report = run_safety_checks(
            observed_compounds={"A", "B"},
            primitives=["A", "B"],
            baseline=baseline,
            initial_state=initial_state,
            theta_hat={"reuse_count": 1.0},
            screening_results=screening_results,
            cache_stats={"ess_at_theta_hat": 80.0, "n_paths": 100},
            n_baseline_samples=20,
            max_depth=3,
            seed=42,
        )

        # If all individual checks pass, overall should be ok
        if (
            report.baseline_check.baseline_ok
            and report.identifiability_check.identifiable
            and report.cache_check.inference_stable
        ):
            assert report.overall_status == "ok"
            assert report.overall_safe

    def test_to_dict_serialization(self, baseline, initial_state):
        """SafetyReport should serialize to dict."""
        report = run_safety_checks(
            observed_compounds={"A"},
            primitives=["A", "B"],
            baseline=baseline,
            initial_state=initial_state,
            theta_hat={},
            screening_results=[],
            cache_stats={},
            n_baseline_samples=10,
            max_depth=2,
            seed=42,
        )

        d = report.to_dict()

        assert "overall_status" in d
        assert "baseline_check" in d
        assert "identifiability_check" in d
        assert "cache_check" in d
        assert "warnings" in d
        assert "recommendations" in d
        assert "adjusted_delta_ll_threshold" in d

    def test_adjusted_threshold_uses_multiplier(self, baseline, initial_state):
        """Adjusted threshold should use baseline multiplier."""
        report = run_safety_checks(
            observed_compounds={"A"},
            primitives=["A", "B"],
            baseline=baseline,
            initial_state=initial_state,
            theta_hat={},
            screening_results=[],
            cache_stats={},
            n_baseline_samples=10,
            max_depth=2,
            seed=42,
        )

        # Threshold = base (2.0) * multiplier
        expected = 2.0 * report.baseline_check.delta_ll_multiplier
        assert report.adjusted_delta_ll_threshold == expected
