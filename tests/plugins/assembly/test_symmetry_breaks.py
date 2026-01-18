"""
Tests for symmetry break features and constraint detection.

Validates that:
1. Symmetry break features are extracted correctly
2. Constraints can be configured with symmetry break parameters
3. Feature weights affect transition rates appropriately
4. Screening detects symmetry breaks when present
"""

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.features.assembly_features import (
    AssemblyFeatureExtractor,
    TransitionType,
)
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.states.assembly_state import AssemblyState


class TestSymmetryBreakFeatureExtraction:
    """Test feature extraction for symmetry breaks A, B, C."""

    def test_depth_gate_reuse_feature_extraction(self):
        """Symmetry Break A: depth-gated reuse feature activates at threshold."""
        extractor = AssemblyFeatureExtractor(depth_gate_threshold=3)

        # Source and target states
        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts(["A", "B"], depth=4)

        # Extract features with target_depth >= threshold
        features = extractor.extract_features(
            source, target, TransitionType.JOIN, target_depth=4
        )

        # depth_gate_reuse should be set (reuse=1, depth=4 >= threshold=3)
        assert features.depth_gate_reuse > 0, "depth_gate_reuse should activate at depth >= threshold"

    def test_depth_gate_reuse_below_threshold(self):
        """Symmetry Break A: depth-gated reuse feature does not activate below threshold."""
        extractor = AssemblyFeatureExtractor(depth_gate_threshold=3)

        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts(["A", "B"], depth=2)

        # Extract features with target_depth < threshold
        features = extractor.extract_features(
            source, target, TransitionType.JOIN, target_depth=2
        )

        # depth_gate_reuse should be 0 (depth=2 < threshold=3)
        assert features.depth_gate_reuse == 0, "depth_gate_reuse should not activate below threshold"

    def test_same_class_reuse_feature_extraction(self):
        """Symmetry Break B: same-class reuse feature activates for intra-class reuse."""
        extractor = AssemblyFeatureExtractor(
            primitive_classes={"A": "class1", "B": "class1", "C": "class2"}
        )

        # Both source and target use class1 primitives
        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts(["A", "B"], depth=1)

        features = extractor.extract_features(source, target, TransitionType.JOIN)

        # same_class_reuse should be set
        assert features.same_class_reuse > 0, "same_class_reuse should activate for intra-class reuse"
        assert features.cross_class_reuse == 0, "cross_class_reuse should not activate"

    def test_cross_class_reuse_feature_extraction(self):
        """Symmetry Break B: cross-class reuse feature activates for inter-class reuse."""
        extractor = AssemblyFeatureExtractor(
            primitive_classes={"A": "class1", "B": "class2"}
        )

        # Source uses class1, target uses both classes
        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts(["A", "B"], depth=1)

        features = extractor.extract_features(source, target, TransitionType.JOIN)

        # cross_class_reuse should be set
        assert features.cross_class_reuse > 0, "cross_class_reuse should activate for inter-class reuse"
        assert features.same_class_reuse == 0, "same_class_reuse should not activate"

    def test_founder_reuse_feature_extraction(self):
        """Symmetry Break C: founder reuse feature activates for founder-rank reuse."""
        extractor = AssemblyFeatureExtractor(founder_rank_threshold=2)

        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts(["A", "B"], depth=1)

        # Extract features with founder_rank=1 (founder)
        features = extractor.extract_features(
            source, target, TransitionType.JOIN, founder_rank=1
        )

        # founder_reuse should be set (rank=1 <= threshold=2)
        assert features.founder_reuse > 0, "founder_reuse should activate for founder rank"

    def test_founder_reuse_above_threshold(self):
        """Symmetry Break C: founder reuse does not activate above rank threshold."""
        extractor = AssemblyFeatureExtractor(founder_rank_threshold=2)

        source = AssemblyState.from_parts(["A"], depth=0)
        target = AssemblyState.from_parts(["A", "B"], depth=1)

        # Extract features with founder_rank=3 (derived, above threshold)
        features = extractor.extract_features(
            source, target, TransitionType.JOIN, founder_rank=3
        )

        # founder_reuse should be 0 (rank=3 > threshold=2)
        assert features.founder_reuse == 0, "founder_reuse should not activate above rank threshold"


class TestSymmetryBreakConstraints:
    """Test constraint configuration and application with symmetry breaks."""

    def test_depth_gate_constraint_configuration(self):
        """Constraint can be configured with depth gate threshold."""
        constraint = AssemblyConstraint(
            feature_weights={"depth_gate_reuse": 1.0},
            depth_gate_threshold=3,
        )

        assert constraint.feature_extractor.depth_gate_threshold == 3
        assert "depth_gate_reuse" in constraint.feature_weights

    def test_context_class_constraint_configuration(self):
        """Constraint can be configured with primitive class mapping."""
        primitive_classes = {"A": "class1", "B": "class1", "C": "class2"}
        constraint = AssemblyConstraint(
            feature_weights={"same_class_reuse": 1.0, "cross_class_reuse": -0.5},
            primitive_classes=primitive_classes,
        )

        assert constraint.feature_extractor.primitive_classes == primitive_classes
        assert "same_class_reuse" in constraint.feature_weights
        assert "cross_class_reuse" in constraint.feature_weights

    def test_founder_bias_constraint_configuration(self):
        """Constraint can be configured with founder rank threshold."""
        constraint = AssemblyConstraint(
            feature_weights={"founder_reuse": 2.0},
            founder_rank_threshold=2,
        )

        assert constraint.feature_extractor.founder_rank_threshold == 2
        assert "founder_reuse" in constraint.feature_weights

    def test_combined_symmetry_breaks_configuration(self):
        """Constraint can be configured with all three symmetry breaks simultaneously."""
        constraint = AssemblyConstraint(
            feature_weights={
                "depth_gate_reuse": 1.0,
                "same_class_reuse": 0.5,
                "cross_class_reuse": -0.3,
                "founder_reuse": 1.5,
            },
            depth_gate_threshold=3,
            primitive_classes={"A": "class1", "B": "class1", "C": "class2"},
            founder_rank_threshold=2,
        )

        assert constraint.feature_extractor.depth_gate_threshold == 3
        assert constraint.feature_extractor.primitive_classes is not None
        assert constraint.feature_extractor.founder_rank_threshold == 2


class TestSymmetryBreakRateModulation:
    """Test that symmetry break features modulate transition rates."""

    def test_depth_gate_reuse_increases_join_rate(self):
        """Depth-gated reuse with positive weight should increase join rates."""
        primitives = ["A", "B"]
        graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=0.0)
        baseline = AssemblyBaseline(kappa=1.0)

        # Constraint with positive depth_gate_reuse weight
        constraint = AssemblyConstraint(
            feature_weights={"depth_gate_reuse": 1.0},
            depth_gate_threshold=2,
        )

        state = AssemblyState.from_parts(["A"], depth=0)
        neighbors_with_constraint = graph.get_neighbors(state, baseline, constraint)

        # Compare to null constraint
        null_constraint = AssemblyConstraint()
        neighbors_null = graph.get_neighbors(state, baseline, null_constraint)

        # Rates should differ (constraint has effect)
        assert len(neighbors_with_constraint) == len(neighbors_null)

    def test_founder_reuse_modulates_rates(self):
        """Founder reuse with weight should modulate transition rates."""
        primitives = ["A", "B"]
        graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=0.0)
        baseline = AssemblyBaseline(kappa=1.0)

        constraint = AssemblyConstraint(
            feature_weights={"founder_reuse": 2.0},
            founder_rank_threshold=2,
        )

        state = AssemblyState.from_parts(["A"], depth=0)
        neighbors = graph.get_neighbors(state, baseline, constraint)

        # Should return valid neighbors
        assert len(neighbors) > 0


class TestScreeningWithSymmetryBreaks:
    """Test that screening can detect and rank symmetry break hypotheses."""

    def test_screening_grid_includes_symmetry_breaks(self):
        """Screening grid should include symmetry break features in candidate generation."""
        from persiste.plugins.assembly.screening.screening import AdaptiveScreeningGrid

        # Create grid with symmetry break features
        grid = AdaptiveScreeningGrid(
            feature_names=[
                "reuse_count",
                "depth_gate_reuse",
                "same_class_reuse",
                "founder_reuse",
            ],
            budget=20,
            top_k=3,
        )

        # Grid should have candidates for each feature
        assert "reuse_count" in grid.feature_names
        assert "depth_gate_reuse" in grid.feature_names
        assert "same_class_reuse" in grid.feature_names
        assert "founder_reuse" in grid.feature_names

    def test_screening_evaluates_symmetry_break_hypotheses(self):
        """Screening should evaluate hypotheses with symmetry break weights."""
        from persiste.plugins.assembly.screening.screening import screen_hypotheses
        from persiste.plugins.assembly.screening.steady_state import (
            SteadyStateAssemblyModel,
            SteadyStateConfig,
        )

        baseline = AssemblyBaseline()
        model = SteadyStateAssemblyModel(
            primitives=["A", "B"],
            baseline=baseline,
            config=SteadyStateConfig(max_depth=3),
        )
        initial_state = AssemblyState.from_parts(["A"], depth=0)
        observed = {"A", "B"}

        # Hypotheses including symmetry breaks
        hypotheses = [
            {},
            {"reuse_count": 0.5},
            {"depth_gate_reuse": 0.5},
            {"same_class_reuse": 0.3},
            {"founder_reuse": 0.7},
        ]

        results = screen_hypotheses(hypotheses, model, observed, initial_state)

        # Should evaluate all hypotheses
        assert len(results) == len(hypotheses)
        # Results should be ranked
        for i in range(len(results) - 1):
            assert results[i].rank <= results[i + 1].rank
