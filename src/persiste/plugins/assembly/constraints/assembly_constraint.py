"""
Assembly theory constraint model.

Constraint encodes:
- Motif stability
- Reusability of subassemblies
- Environmental compatibility
- Recursive reuse (core assembly theory idea)
"""

import numpy as np

from persiste.core.constraints import ConstraintModel, ParameterSpace
from persiste.plugins.assembly.baselines.assembly_baseline import TransitionType
from persiste.plugins.assembly.features.assembly_features import AssemblyFeatureExtractor
from persiste.plugins.assembly.states.assembly_state import AssemblyState


class AssemblyConstraint(ConstraintModel):
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
        depth_gate_threshold: int | None = None,
        primitive_classes: dict[str, str] | None = None,
        founder_rank_threshold: int | None = None,
    ):
        """
        Initialize assembly constraint model.

        Args:
            feature_weights: Dict of feature_name -> weight (default: {} = null model)
            feature_extractor: Feature extractor (default: AssemblyFeatureExtractor())
            depth_gate_threshold: Depth threshold for Symmetry Break A
            primitive_classes: Mapping primitive -> class for Symmetry Break B
            founder_rank_threshold: Rank threshold for Symmetry Break C
        """
        # Call core ConstraintModel __init__ with placeholder values
        # Assembly models don't use the full core state space enumeration yet.
        super().__init__(
            states=None,  # type: ignore
            baseline=None,  # type: ignore
            graph=None,  # type: ignore
            parameters={"theta": feature_weights if feature_weights is not None else {}},
            allow_facilitation=False,  # Assembly theory defaults to suppression only
        )
        self.feature_weights = self.parameters["theta"]
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else AssemblyFeatureExtractor(
                depth_gate_threshold=depth_gate_threshold,
                primitive_classes=primitive_classes,
                founder_rank_threshold=founder_rank_threshold,
            )
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

    def get_parameters(self) -> dict[str, float]:
        """
        Get constraint parameters θ (feature weights).

        Returns:
            Dict of feature_name -> weight
        """
        return self.feature_weights.copy()

    def set_parameters(self, **params):
        """
        Set constraint parameters (feature weights).

        Args:
            **params: Parameters dict, expected to contain 'theta'.
        """
        super().set_parameters(**params)
        if "theta" in params:
            self.feature_weights = self.parameters["theta"]

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
        Pack feature weights into flat vector for optimization.

        Args:
            parameters: Optional parameters dict (uses self.parameters if None).
                Can be nested {"theta": {...}} or a flat feature weight dict.

        Returns:
            Flat numpy array of weights
        """
        params = parameters if parameters is not None else self.parameters
        theta = params.get("theta", params)

        if not theta or not isinstance(theta, dict):
            return np.array([], dtype=float)

        # Sort keys for reproducibility
        keys = sorted(theta.keys())
        return np.array([theta[k] for k in keys], dtype=float)

    def unpack(self, vector: np.ndarray) -> dict:
        """
        Unpack flat vector into feature weights dict.

        Args:
            vector: Flat numpy array

        Returns:
            Dict of {"theta": {feature_name: weight}}
        """
        if len(vector) == 0:
            return {"theta": {}}

        # Use current feature_weights keys for ordering
        keys = sorted(self.feature_weights.keys())

        if len(vector) != len(keys):
            raise ValueError(
                f"Vector length {len(vector)} != expected {len(keys)}. Expected features: {keys}"
            )

        return {"theta": {k: float(vector[i]) for i, k in enumerate(keys)}}

    def get_parameter_space(self, parameters: dict | None = None) -> ParameterSpace:
        """
        Describe the structural parameter layout.

        Args:
            parameters: Optional parameters dict.

        Returns:
            ParameterSpace with feature names as keys and 0.0 as neutral.
        """
        params = parameters if parameters is not None else self.parameters
        theta = params.get("theta", params)
        if isinstance(theta, dict) and theta:
            keys = tuple(sorted(theta.keys()))
        else:
            keys = tuple(sorted(self.feature_weights.keys()))
        return ParameterSpace(keys=keys, neutral_value=0.0)

    def num_free_parameters(self, parameters: dict | None = None) -> int:
        """
        Count number of free parameters.

        Args:
            parameters: Optional parameters dict.

        Returns:
            Number of parameters
        """
        params = parameters if parameters is not None else self.parameters
        theta = params.get("theta", params)
        return len(theta) if isinstance(theta, dict) else 0

    def initial_parameters(self) -> np.ndarray:
        """
        Get initial parameter vector for optimization.

        Returns:
            Initial parameter vector (neutral starting point = 0.0)
        """
        if not self.feature_weights:
            return np.array([], dtype=float)

        return np.zeros(len(self.feature_weights), dtype=float)

    def get_constrained_baseline(self, parameters: dict | None = None):
        """
        Create baseline with constrained rates.

        DEVIATION RATIONALE:
        For assembly theory, constraints are applied dynamically during graph neighbor
        generation (lazy state-space traversal), not by wrapping a pre-enumerated
        rate matrix. We return 'self' so that graph operations and stochastic
        simulators can access constraint_contribution() directly.

        Args:
            parameters: Optional parameters dict (uses self.feature_weights if None)

        Returns:
            This constraint object (AssemblyConstraint) acting as a constrained baseline.
        """
        if parameters is not None:
            # We must be careful not to mutate global state if possible,
            # but currently ConstraintInference expects this object to have the params.
            self.set_parameters(**parameters)

        return self
