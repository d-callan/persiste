"""
Assembly feature extractors (Layer 1 - mechanics).

This module defines WHAT features can be computed from assembly transitions.
It does NOT define which features matter or what direction they push.

Features are hypothesis-neutral. Weights are hypotheses.

Key principle: Features are cheap, local, compositional, interpretable.
"""

from dataclasses import dataclass

from persiste.plugins.assembly.baselines.assembly_baseline import TransitionType
from persiste.plugins.assembly.states.assembly_state import AssemblyState


@dataclass
class TransitionFeatures:
    """
    Features extracted from an assembly transition.
    
    These are observables, not value judgments.
    A constraint model assigns weights to these features.
    
    Attributes:
        reuse_count: Number of times source appears in target's history
        depth_change: Change in assembly depth (target - source)
        size_change: Change in number of parts (target - source)
        motif_gained: Set of motifs gained in transition
        motif_lost: Set of motifs lost in transition
        transition_type: Type of transition (join, split, decay)
        depth_gate_reuse: Reuse bonus when target depth >= threshold (Symmetry Break A)
        same_class_reuse: Reuse bonus when source/target in same class (Symmetry Break B)
        cross_class_reuse: Reuse bonus when source/target in different classes (Symmetry Break B)
        founder_reuse: Reuse bonus for early-visited states (Symmetry Break C)
    """
    reuse_count: int = 0
    depth_change: int = 0
    size_change: int = 0
    motif_gained: set[str] = None
    motif_lost: set[str] = None
    transition_type: str = ""
    depth_gate_reuse: float = 0.0
    same_class_reuse: float = 0.0
    cross_class_reuse: float = 0.0
    founder_reuse: float = 0.0
    
    def __post_init__(self):
        if self.motif_gained is None:
            self.motif_gained = set()
        if self.motif_lost is None:
            self.motif_lost = set()
    
    def to_dict(self) -> dict[str, float]:
        """
        Convert features to flat dict for constraint evaluation.

        Returns:
            Dict of feature_name -> value
        """
        features = {
            'reuse_count': float(self.reuse_count),
            'depth_change': float(self.depth_change),
            'size_change': float(self.size_change),
            'depth_gate_reuse': self.depth_gate_reuse,
            'same_class_reuse': self.same_class_reuse,
            'cross_class_reuse': self.cross_class_reuse,
            'founder_reuse': self.founder_reuse,
        }

        # Add transition type indicator
        if self.transition_type:
            features[f'transition_{self.transition_type}'] = 1.0

        # Add motif indicators
        for motif in self.motif_gained:
            features[f'motif_gained_{motif}'] = 1.0
        for motif in self.motif_lost:
            features[f'motif_lost_{motif}'] = 1.0

        return features


class AssemblyFeatureExtractor:
    """
    Extract features from assembly transitions.
    
    This is Layer 1 (mechanics) - defines WHAT features exist.
    Layer 2 (theory) defines which features matter and their weights.
    
    Features are:
    - Cheap: O(1) or O(parts) computation
    - Local: Only depend on source and target states
    - Compositional: Combine linearly
    - Interpretable: Clear physical meaning
    """
    
    def __init__(
        self,
        depth_gate_threshold: int | None = None,
        primitive_classes: dict[str, str] | None = None,
        founder_rank_threshold: int | None = None,
    ):
        self.depth_gate_threshold = depth_gate_threshold
        self.primitive_classes = primitive_classes or {}
        self.founder_rank_threshold = founder_rank_threshold

    def extract_features(
        self,
        source: AssemblyState,
        target: AssemblyState,
        transition_type: TransitionType,
        *,
        target_depth: int | None = None,
        founder_rank: int | None = None,
    ) -> TransitionFeatures:
        """
        Extract all features from a transition.

        Args:
            source: Source assembly state
            target: Target assembly state
            transition_type: Type of transition
            target_depth: Optional depth for Symmetry Break A (overrides target.assembly_depth)
            founder_rank: Optional rank for Symmetry Break C (1 = founder, higher = derived)

        Returns:
            TransitionFeatures object
        """
        features = TransitionFeatures()

        # Reuse: how many times does source appear in target's history?
        reuse = self._compute_reuse(source, target)
        features.reuse_count = reuse

        # Depth change
        features.depth_change = target.assembly_depth - source.assembly_depth

        # Size change
        features.size_change = target.size - source.size

        # Motif changes
        features.motif_gained = target.motifs - source.motifs
        features.motif_lost = source.motifs - target.motifs

        # Transition type
        features.transition_type = transition_type.value if transition_type else ""

        # Symmetry Break A: Depth-gated reuse
        # Bonus when reuse occurs at depth >= threshold
        effective_depth = target_depth if target_depth is not None else target.assembly_depth
        if (
            reuse > 0
            and self.depth_gate_threshold is not None
            and effective_depth >= self.depth_gate_threshold
        ):
            features.depth_gate_reuse = float(reuse)

        # Symmetry Break B: Context-class reuse
        # Bonus for same-class reuse, penalty for cross-class
        if reuse > 0 and self.primitive_classes:
            source_classes = {self.primitive_classes.get(p, "unknown") for p in source.get_parts_dict().keys()}
            target_classes = {self.primitive_classes.get(p, "unknown") for p in target.get_parts_dict().keys()}
            if source_classes and target_classes:
                if source_classes == target_classes:
                    features.same_class_reuse = float(reuse)
                else:
                    features.cross_class_reuse = float(reuse)

        # Symmetry Break C: Founder bias
        # Bonus for early-visited (founder) states
        if reuse > 0 and founder_rank is not None and self.founder_rank_threshold is not None:
            if founder_rank <= self.founder_rank_threshold:
                features.founder_reuse = float(reuse)

        return features
    
    def _compute_reuse(self, source: AssemblyState, target: AssemblyState) -> int:
        """
        Count reuse: how many subassemblies in target match source?
        
        Simple heuristic: check if source is a subassembly of target.
        Full implementation would track assembly history.
        """
        if self._is_submultiset(source.get_parts_dict(), target.get_parts_dict()):
            return 1
        return 0
    

    @staticmethod
    def _is_submultiset(source_parts: dict[str, int], target_parts: dict[str, int]) -> bool:
        """
        Check whether every part in source_parts appears at least as many times in target_parts.

        This intentionally allows the target to have additional copies of a part so that
        reuse detection considers true subassemblies rather than requiring an exact match.
        """
        for part, count in source_parts.items():
            if target_parts.get(part, 0) < count:
                return False
        return True
    
    def get_feature_names(self) -> list[str]:
        """
        Get list of all possible feature names.

        Returns:
            List of feature names
        """
        return [
            'reuse_count',
            'depth_change',
            'size_change',
            'depth_gate_reuse',
            'same_class_reuse',
            'cross_class_reuse',
            'founder_reuse',
        ]

    def __str__(self) -> str:
        return (
            "AssemblyFeatureExtractor(features: reuse, depth, size, motifs, "
            "depth_gate, context_class, founder_bias)"
        )
