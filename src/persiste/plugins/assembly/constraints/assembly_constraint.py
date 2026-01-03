"""
Assembly theory constraint model.

Constraint encodes:
- Motif stability
- Reusability of subassemblies
- Environmental compatibility
- Recursive reuse (core assembly theory idea)
"""

import numpy as np

from persiste.core.constraint_utils import MultiplicativeConstraint
from persiste.plugins.assembly.baselines.assembly_baseline import TransitionType
from persiste.plugins.assembly.features.assembly_features import AssemblyFeatureExtractor
from persiste.plugins.assembly.states.assembly_state import AssemblyState


class AssemblyConstraint(MultiplicativeConstraint):
    """
    Constraint model for assembly theory (Layer 2 - theories).

    Defines constraint contribution C(i → j; θ) that modifies baseline rates:
        λ_eff(i → j) = λ_baseline(i → j) × exp(C(i → j; θ))

    Where C(i → j; θ) = θ · f(i → j)

    θ = feature weights (this is the theory/hypothesis)
    f = feature vector (from Layer 1 feature extractor)

    Crucially: This class does NOT assume which features matter.
    It accepts arbitrary feature weights from the user or inference.

    Attributes:
        feature_weights: Dict of feature_name -> weight (log-scale)
        feature_extractor: Extracts features from transitions
    """

    def __init__(
        self,
        feature_weights: dict[str, float] | None = None,
        feature_extractor: AssemblyFeatureExtractor | None = None,
    ):
        """
        Initialize assembly constraint model.

        Args:
            feature_weights: Dict of feature_name -> weight (default: {} = null model)
            feature_extractor: Feature extractor (default: AssemblyFeatureExtractor())

        Examples:
            # Null model (no constraints)
            constraint = AssemblyConstraint()

            # Reuse-only model
            constraint = AssemblyConstraint({'reuse_count': 1.0})

            # Assembly theory model
            constraint = AssemblyConstraint({
                'reuse_count': 1.0,
                'depth_change': -0.3,
                'motif_gained_helix': 2.0,
            })
        """
        self.feature_weights = feature_weights if feature_weights is not None else {}
        self.feature_extractor = (
            feature_extractor if feature_extractor is not None else AssemblyFeatureExtractor()
        )

    def constraint_contribution(
        self,
        source: AssemblyState,
        target: AssemblyState,
        transition_type: TransitionType,
    ) -> float:
        """
        Compute constraint contribution C(i → j; θ).

        C(i → j; θ) = θ · f(i → j)

        Where:
        - θ = feature weights (the theory/hypothesis)
        - f = feature vector (from feature extractor)

        This is added to log-rate: log(λ_eff) = log(λ_baseline) + C

        Args:
            source: Source assembly state
            target: Target assembly state
            transition_type: Type of transition

        Returns:
            Constraint contribution (log-scale)
        """
        # Extract features (Layer 1 - mechanics)
        features = self.feature_extractor.extract_features(source, target, transition_type)
        feature_dict = features.to_dict()

        contribution = 0.0
        for feature_name, feature_value in feature_dict.items():
            weight = self.feature_weights.get(feature_name, 0.0)
            contribution += weight * feature_value
        return contribution

    def _is_reused(self, source: AssemblyState, target: AssemblyState) -> bool:
        """
        Check if target reuses source as subassembly.

        Core assembly theory idea: recursive reuse is favored.
        """
        return source.is_subassembly_of(target)

    def _env_score(self, state: AssemblyState) -> float:
        """
        Environmental compatibility score.

        Could be based on:
        - Solubility proxies
        - Size compatibility
        - Functional group exposure

        For now, simple placeholder.
        """
        # Favor intermediate sizes (bell curve)
        optimal_size = 5
        size_diff = abs(state.total_parts() - optimal_size)
        return -0.1 * size_diff

    def get_parameters(self) -> dict[str, float]:
        """
        Get constraint parameters θ (feature weights).

        Returns:
            Dict of feature_name -> weight
        """
        return self.feature_weights.copy()

    def set_parameters(self, params: dict[str, float]):
        """
        Set constraint parameters θ (feature weights).

        Args:
            params: Dict of feature_name -> weight
        """
        self.feature_weights = params.copy()

    def __str__(self) -> str:
        if not self.feature_weights:
            return "AssemblyConstraint(null model: θ = {})"

        weights_str = ", ".join(f"{k}={v:.2f}" for k, v in self.feature_weights.items())
        return f"AssemblyConstraint({weights_str})"

    def get_rate_multipliers(
        self,
        theta: float | None = None,
        *,
        family_idx: int | None = None,
        family_id: str | None = None,
        lineage_id: str | None = None,
        context: dict | None = None,
    ) -> dict[tuple[int, int], float]:
        """
        Map constraint contribution onto a 2-state placeholder matrix for shared helpers.
        """
        if context is None:
            raise ValueError("AssemblyConstraint requires context with source/target states.")
        source = context["source"]
        target = context["target"]
        transition_type = context["transition_type"]
        contrib = self.constraint_contribution(source, target, transition_type)
        multiplier = float(np.exp(contrib))
        return {(0, 1): multiplier}

    @classmethod
    def null_model(cls) -> "AssemblyConstraint":
        """Create null constraint model (no constraints)."""
        return cls(feature_weights={})

    @classmethod
    def reuse_only(cls, reuse_weight: float = 1.0) -> "AssemblyConstraint":
        """Create reuse-only constraint model."""
        return cls(feature_weights={"reuse_count": reuse_weight})

    @classmethod
    def assembly_theory(
        cls, reuse: float = 1.0, depth_penalty: float = -0.3
    ) -> "AssemblyConstraint":
        """Create standard assembly theory constraint model."""
        return cls(
            feature_weights={
                "reuse_count": reuse,
                "depth_change": depth_penalty,
            }
        )

    # ========================================================================
    # ConstraintModel Interface (Phase 1.6 - Plumbing)
    # ========================================================================

    def pack(self, parameters: dict | None = None) -> np.ndarray:
        """
        Pack feature weights into flat vector.

        Required for scipy.optimize integration.

        Args:
            parameters: Optional parameters dict (uses self.feature_weights if None)

        Returns:
            Flat numpy array of weights
        """
        params = parameters if parameters is not None else self.feature_weights

        if not params:
            return np.array([])

        # Sort keys for reproducibility
        keys = sorted(params.keys())
        return np.array([params[k] for k in keys])

    def unpack(self, vector: np.ndarray) -> dict:
        """
        Unpack flat vector into feature weights dict.

        Inverse of pack().

        Args:
            vector: Flat numpy array

        Returns:
            Dict of feature_name -> weight
        """
        if len(vector) == 0:
            return {}

        # Use current feature_weights keys for ordering
        keys = sorted(self.feature_weights.keys())

        if len(vector) != len(keys):
            raise ValueError(
                f"Vector length {len(vector)} != expected {len(keys)}. Expected features: {keys}"
            )

        return {k: float(vector[i]) for i, k in enumerate(keys)}

    def num_free_parameters(self, parameters: dict | None = None) -> int:
        """
        Count number of free parameters.

        Required for AIC/BIC computation.

        Args:
            parameters: Optional parameters dict (uses self.feature_weights if None)

        Returns:
            Number of parameters
        """
        params = parameters if parameters is not None else self.feature_weights
        return len(params)

    def initial_parameters(self) -> np.ndarray:
        """
        Get initial parameter vector for optimization.

        Returns neutral starting point (all weights = 0.0).

        Returns:
            Initial parameter vector
        """
        if not self.feature_weights:
            return np.array([])

        # Start from zeros (neutral)
        n_params = len(self.feature_weights)
        return np.zeros(n_params)

    def get_constrained_baseline(self, parameters: dict | None = None):
        """
        Create baseline with constrained rates.

        This is where θ gets applied to baseline rates.
        Returns a modified baseline that can be used for simulation/inference.

        Args:
            parameters: Optional parameters dict (uses self.feature_weights if None)

        Returns:
            Modified baseline (conceptually - actual implementation returns self)

        Note:
            For assembly, the constraint is applied during graph neighbor generation,
            not by modifying the baseline directly. This method exists for interface
            compatibility but doesn't create a new baseline object.
        """
        # For assembly theory, constraints are applied in graph.get_neighbors()
        # via constraint.constraint_contribution()
        #
        # We don't modify the baseline itself - the baseline is physics-agnostic.
        # Instead, we return self so that graph operations can access constraint_contribution()

        if parameters is not None:
            self.feature_weights = parameters.copy()
            # Return self with updated parameters
            # (In a more sophisticated implementation, we'd return a copy)

        return self
