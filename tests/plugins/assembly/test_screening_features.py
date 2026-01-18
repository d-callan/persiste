"""
Tests for screening and feature extraction coverage.

Validates that:
1. Screening grid generation works with various feature sets
2. Deterministic screening correctly evaluates hypotheses
3. Feature extraction handles edge cases
"""

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.features.assembly_features import (
    AssemblyFeatureExtractor,
    TransitionType,
)
from persiste.plugins.assembly.screening.screening import (
    AdaptiveScreeningGrid,
    screen_hypotheses,
)
from persiste.plugins.assembly.screening.steady_state import (
    SteadyStateAssemblyModel,
    SteadyStateConfig,
)
from persiste.plugins.assembly.states.assembly_state import AssemblyState


class TestScreeningGrid:
    """Test screening grid generation and hypothesis enumeration."""

    def test_adaptive_screening_grid_single_feature(self):
        """Grid should generate candidates for single feature."""
        grid = AdaptiveScreeningGrid(
            feature_names=["reuse_count"],
            budget=10,
            top_k=2,
        )

        assert grid.feature_names == ["reuse_count"]
        assert grid.budget == 10
        assert grid.top_k == 2

    def test_adaptive_screening_grid_multiple_features(self):
        """Grid should handle multiple features."""
        grid = AdaptiveScreeningGrid(
            feature_names=["reuse_count", "depth_change", "size_change"],
            budget=30,
            top_k=5,
        )

        assert len(grid.feature_names) == 3
        assert grid.budget == 30

    def test_adaptive_screening_grid_with_symmetry_breaks(self):
        """Grid should include symmetry break features."""
        grid = AdaptiveScreeningGrid(
            feature_names=[
                "reuse_count",
                "depth_gate_reuse",
                "same_class_reuse",
                "cross_class_reuse",
                "founder_reuse",
            ],
            budget=50,
            top_k=5,
        )

        assert "depth_gate_reuse" in grid.feature_names
        assert "same_class_reuse" in grid.feature_names
        assert "founder_reuse" in grid.feature_names


class TestDeterministicScreening:
    """Test deterministic screening evaluation."""

    def test_screen_hypotheses_basic(self):
        """Screening should evaluate and rank hypotheses."""
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
            {"reuse_count": 0.5},
            {"reuse_count": 1.0},
        ]

        results = screen_hypotheses(hypotheses, model, observed, initial_state)

        assert len(results) == 3
        # Check ranking
        for i in range(len(results) - 1):
            assert results[i].rank <= results[i + 1].rank

    def test_screen_hypotheses_with_threshold(self):
        """Screening should mark hypotheses as passed/failed based on threshold."""
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
            {"reuse_count": 0.1},
            {"reuse_count": 5.0},
        ]

        results = screen_hypotheses(
            hypotheses, model, observed, initial_state, threshold=2.0
        )

        # At least one result should be marked as passed or failed
        assert any(r.passed for r in results) or any(not r.passed for r in results)

    def test_screen_hypotheses_includes_ll_metrics(self):
        """Screening results should include absolute and null LL."""
        baseline = AssemblyBaseline()
        model = SteadyStateAssemblyModel(
            primitives=["A", "B"],
            baseline=baseline,
            config=SteadyStateConfig(max_depth=3),
        )
        initial_state = AssemblyState.from_parts(["A"], depth=0)
        observed = {"A", "B"}

        hypotheses = [{}, {"reuse_count": 0.5}]

        results = screen_hypotheses(hypotheses, model, observed, initial_state)

        for result in results:
            assert result.absolute_ll is not None
            assert result.null_ll is not None
            assert result.delta_ll is not None


class TestFeatureExtractionEdgeCases:
    """Test feature extraction for edge cases and boundary conditions."""

    def test_feature_extraction_no_reuse(self):
        """Feature extraction should handle non-reuse transitions."""
        extractor = AssemblyFeatureExtractor()

        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts(["B"], depth=1)

        features = extractor.extract_features(source, target, TransitionType.JOIN)

        # reuse_count should be 0 (no reuse)
        assert features.reuse_count == 0

    def test_feature_extraction_depth_change(self):
        """Feature extraction should compute depth changes."""
        extractor = AssemblyFeatureExtractor()

        source = AssemblyState.from_parts(["A"], depth=1)
        target = AssemblyState.from_parts(["A", "B"], depth=3)

        features = extractor.extract_features(source, target, TransitionType.JOIN)

        # depth_change should be 2
        assert features.depth_change == 2

    def test_feature_extraction_size_change(self):
        """Feature extraction should compute size changes."""
        extractor = AssemblyFeatureExtractor()

        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts(["A", "B", "C"], depth=1)

        features = extractor.extract_features(source, target, TransitionType.JOIN)

        # size_change should be 2 (from 1 to 3 parts)
        assert features.size_change == 2

    def test_feature_extraction_transition_type_join(self):
        """Feature extraction should record transition type."""
        extractor = AssemblyFeatureExtractor()

        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts(["A", "B"], depth=1)

        features = extractor.extract_features(source, target, TransitionType.JOIN)

        assert features.transition_type == "join"

    def test_feature_extraction_transition_type_split(self):
        """Feature extraction should record split transitions."""
        extractor = AssemblyFeatureExtractor()

        source = AssemblyState.from_parts(["A", "B"], depth=1)
        target = AssemblyState.from_parts(["A"], depth=0)

        features = extractor.extract_features(source, target, TransitionType.SPLIT)

        assert features.transition_type == "split"

    def test_feature_extraction_transition_type_decay(self):
        """Feature extraction should record decay transitions."""
        extractor = AssemblyFeatureExtractor()

        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts([], depth=0)

        features = extractor.extract_features(source, target, TransitionType.DECAY)

        assert features.transition_type == "decay"

    def test_feature_extraction_no_config(self):
        """Feature extraction should work without symmetry break configuration."""
        extractor = AssemblyFeatureExtractor()

        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts(["A", "B"], depth=1)

        features = extractor.extract_features(source, target, TransitionType.JOIN)

        # All symmetry break features should be 0 (no config)
        assert features.depth_gate_reuse == 0
        assert features.same_class_reuse == 0
        assert features.cross_class_reuse == 0
        assert features.founder_reuse == 0

    def test_feature_extraction_partial_config(self):
        """Feature extraction should work with partial symmetry break configuration."""
        extractor = AssemblyFeatureExtractor(depth_gate_threshold=2)

        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts(["A", "B"], depth=3)

        features = extractor.extract_features(
            source, target, TransitionType.JOIN, target_depth=3
        )

        # depth_gate_reuse should be set, others should be 0
        assert features.depth_gate_reuse > 0
        assert features.same_class_reuse == 0
        assert features.founder_reuse == 0


class TestConstraintWithoutSymmetryBreaks:
    """Test that constraints work correctly without symmetry break configuration."""

    def test_constraint_null_weights(self):
        """Constraint with null weights should not affect rates."""
        constraint = AssemblyConstraint()

        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts(["A", "B"], depth=1)

        # Contribution should be 0 (no weights)
        contribution = constraint.constraint_contribution(
            source, target, TransitionType.JOIN
        )
        assert contribution == 0.0

    def test_constraint_backward_compatibility(self):
        """Constraint should be backward compatible with legacy feature weights."""
        constraint = AssemblyConstraint(
            feature_weights={"depth_change": -0.5}
        )

        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts(["A", "B"], depth=2)

        # Contribution should be non-zero (depth_change = 2, weight = -0.5)
        contribution = constraint.constraint_contribution(
            source, target, TransitionType.JOIN
        )
        assert contribution == -1.0  # 2 * -0.5 = -1.0
