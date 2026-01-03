"""
Assembly feature extractors (Layer 1 - mechanics).

This module defines WHAT features can be computed from assembly transitions.
It does NOT define which features matter or what direction they push.

Features are hypothesis-neutral. Weights are hypotheses.

Key principle: Features are cheap, local, compositional, interpretable.
"""

from typing import Dict, Set, Optional
from dataclasses import dataclass
import numpy as np

from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import TransitionType


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
        symmetry_score: Symmetry of target state (0 = asymmetric, 1 = symmetric)
        diversity_score: Part diversity in target (Shannon entropy)
    """
    reuse_count: int = 0
    depth_change: int = 0
    size_change: int = 0
    motif_gained: Set[str] = None
    motif_lost: Set[str] = None
    symmetry_score: float = 0.0
    diversity_score: float = 0.0
    
    def __post_init__(self):
        if self.motif_gained is None:
            self.motif_gained = set()
        if self.motif_lost is None:
            self.motif_lost = set()
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert features to flat dict for constraint evaluation.
        
        Returns:
            Dict of feature_name -> value
        """
        features = {
            'reuse_count': float(self.reuse_count),
            'depth_change': float(self.depth_change),
            'size_change': float(self.size_change),
            'symmetry_score': self.symmetry_score,
            'diversity_score': self.diversity_score,
        }
        
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
    
    def extract_features(
        self,
        source: AssemblyState,
        target: AssemblyState,
        transition_type: TransitionType,
    ) -> TransitionFeatures:
        """
        Extract all features from a transition.
        
        Args:
            source: Source assembly state
            target: Target assembly state
            transition_type: Type of transition
            
        Returns:
            TransitionFeatures object
        """
        features = TransitionFeatures()
        
        # Reuse: how many times does source appear in target's history?
        features.reuse_count = self._compute_reuse(source, target)
        
        # Depth change
        features.depth_change = target.assembly_depth - source.assembly_depth
        
        # Size change
        features.size_change = target.size - source.size
        
        # Motif changes
        features.motif_gained = target.motifs - source.motifs
        features.motif_lost = source.motifs - target.motifs
        
        # Symmetry (of target)
        features.symmetry_score = self._compute_symmetry(target)
        
        # Diversity (of target)
        features.diversity_score = self._compute_diversity(target)
        
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
    
    def _compute_symmetry(self, state: AssemblyState) -> float:
        """
        Compute symmetry score of state.
        
        Simple heuristic: ratio of most common part to total parts.
        1.0 = all same part (maximally symmetric)
        1/n = all different parts (minimally symmetric)
        """
        if state.size == 0:
            return 0.0
        
        parts_dict = state.get_parts_dict()
        if not parts_dict:
            return 0.0
        
        max_count = max(parts_dict.values())
        return max_count / state.size
    
    def _compute_diversity(self, state: AssemblyState) -> float:
        """
        Compute part diversity (Shannon entropy).
        
        Higher = more diverse parts
        Lower = dominated by few parts
        """
        if state.size == 0:
            return 0.0
        
        parts_dict = state.get_parts_dict()
        if not parts_dict:
            return 0.0
        
        # Shannon entropy: -sum(p_i * log(p_i))
        entropy = 0.0
        for count in parts_dict.values():
            p = count / state.size
            if p > 0:
                entropy -= p * np.log(p)
        
        return entropy

    @staticmethod
    def _is_submultiset(source_parts: Dict[str, int], target_parts: Dict[str, int]) -> bool:
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
            'symmetry_score',
            'diversity_score',
            # Motif features are dynamic (motif_gained_X, motif_lost_X)
        ]
    
    def __str__(self) -> str:
        return "AssemblyFeatureExtractor(features: reuse, depth, size, motifs, symmetry, diversity)"
