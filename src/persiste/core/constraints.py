"""Constraint parameter models."""

from typing import Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

EPSILON = 1e-6

if TYPE_CHECKING:
    from persiste.core.states import StateSpace
    from persiste.core.baseline import Baseline
    from persiste.core.transitions import TransitionGraph


@dataclass(frozen=True)
class ParameterSpace:
    """Structural definition of constraint parameters."""
    
    keys: tuple[Any, ...]
    neutral_value: float = 1.0
    
    def size(self) -> int:
        return len(self.keys)
    
    def neutral_vector(self) -> np.ndarray:
        if not self.keys:
            return np.array([])
        return np.full(len(self.keys), self.neutral_value)


@dataclass
class ConstraintModel:
    """
    Constraint model specification.
    
    Defines how constraint parameters θ modify opportunity (baseline rates).
    Each θ configuration transforms baseline λ_ij → constrained λ*_ij.
    
    GENERATIVE SEMANTICS:
    - ConstraintModel is both descriptive ("these θ explain data") and generative
    - get_constrained_baseline() creates a new Baseline representing the constrained process
    - This enables simulation, likelihood comparison, and hypothesis testing
    
    REGIME DISTINCTION:
    - Pure constraint model (θ ≤ 1): suppression only, no facilitation
      → Appropriate for: early life, assembly theory, pre-Darwinian regimes
    - Selection-enabled model (θ unrestricted): allows facilitation (θ > 1)
      → Appropriate for: phylogenetics (HyPhy), post-Darwinian evolution
    
    Key method: effective_rate(i, j) returns constrained rate λ*_ij = f(λ_ij, θ)
    
    Constraint structures:
    - per_transition: λ*_ij = θ_ij × λ_ij (most general)
    - per_state: λ*_ij = θ_i × λ_ij (canalization)
    - hierarchical: θ_ij ~ LogNormal(μ_group, σ) (shared strength)
    - sparse: most transitions constrained, few allowed (strong suppression)
    
    Sparsity modes (for sparse structure):
    - "soft" (Bayesian shrinkage): "Most transitions are constrained, but uncertainty matters."
      → Prior strongly shrinks toward low values, data can override
      → Best for: metagenomics, early life chemistry, viral quasispecies
      → Constraint precedes observation
    
    - "penalized" (penalized MLE): "Most transitions are unlikely, but I only trust the data."
      → Objective = log-likelihood - penalty (L1/L0-ish)
      → Best for: large datasets, exploratory analysis, CI pipelines
      → Constraint is inferred, not assumed
    
    - "latent" (spike-and-slab): "Some transitions are truly forbidden; others are allowed."
      → Mixture model with discrete classes (z ∈ {0,1})
      → Best for: assembly theory, early life origin, pathway discovery
      → Life is defined by allowed transitions
    
    Attributes:
        states: State space
        baseline: Baseline rate process
        graph: Transition graph (defines allowed transitions)
        constraint_structure: Type of constraint parameterization
        parameters: Constraint parameters (θ_ij, θ_i, etc.)
        allow_facilitation: If False, enforce θ ≤ 1 (pure constraint model)
        sparsity: Sparsity mode ("soft", "penalized", "latent") - only for sparse structure
        strength: Sparsity strength (interpretable scale, default 10.0)
        priors: Prior specifications for Bayesian inference
    """
    
    states: "StateSpace"
    baseline: "Baseline"
    graph: "TransitionGraph"
    constraint_structure: str = "per_transition"
    parameters: Dict[str, Any] = field(default_factory=dict)
    allow_facilitation: bool = True
    sparsity: str = "soft"  # "soft", "penalized", "latent"
    strength: float = 10.0  # Interpretable scale for sparsity
    priors: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate constraint structure."""
        valid_structures = [
            "per_transition",
            "per_state",
            "hierarchical",
            "sparse",
        ]
        if self.constraint_structure not in valid_structures:
            raise ValueError(
                f"constraint_structure must be one of {valid_structures}, "
                f"got {self.constraint_structure}"
            )
    
    def effective_rate(self, i: int, j: int, parameters: Optional[Dict[str, Any]] = None) -> float:
        """
        Return constrained rate λ*_ij = f(λ_ij, θ).
        
        This is the key abstraction: transforms baseline → constrained.
        
        Respects graph topology: returns 0.0 for transitions forbidden by graph.
        Constraint cannot create transitions that don't exist in the graph.
        
        Enforces facilitation policy: if allow_facilitation=False, clips θ to [0, 1].
        
        PURE FUNCTIONAL: parameters argument allows evaluation without mutating state.
        Essential for autodiff, vectorization, and parallel optimization.
        
        Args:
            i: Source state index
            j: Target state index
            parameters: Optional constraint parameters (uses self.parameters if None)
            
        Returns:
            Constrained transition rate λ*_ij
            
        Examples:
            θ_ij < 1 → constraint (suppression)
            θ_ij = 1 → neutral (no constraint)
            θ_ij > 1 → facilitation (positive selection, if allowed)
        """
        # Check graph topology first
        if not self.graph.allows(i, j):
            return 0.0
        
        # Get baseline rate
        λ_ij = self.baseline.get_rate(i, j)
        
        # Use provided parameters or fall back to instance parameters
        params = parameters if parameters is not None else self.parameters
        
        # Get constraint parameter via helper (using params)
        θ_ij = self._get_theta_from_params(i, j, params)
        
        # Enforce facilitation policy
        if not self.allow_facilitation and θ_ij > 1.0:
            θ_ij = 1.0
        
        # Apply constraint: λ*_ij = θ_ij × λ_ij
        return θ_ij * λ_ij
    
    def get_theta(self, i: int, j: int) -> float:
        """
        Get constraint parameter θ for transition i → j.
        
        Uses instance parameters. For pure functional evaluation,
        use _get_theta_from_params() instead.
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            Constraint parameter θ (before facilitation policy enforcement)
        """
        return self._get_theta_from_params(i, j, self.parameters)
    
    def _get_theta_from_params(self, i: int, j: int, params: Dict[str, Any]) -> float:
        """
        Get constraint parameter θ from given parameters dict.
        
        Internal helper for pure functional evaluation.
        
        Args:
            i: Source state index
            j: Target state index
            params: Parameters dict
            
        Returns:
            Constraint parameter θ
        """
        θ = params.get("theta", {})
        
        if self.constraint_structure == "per_transition":
            return θ.get((i, j), 1.0)
        
        elif self.constraint_structure == "per_state":
            return θ.get(i, 1.0)
        
        elif self.constraint_structure == "hierarchical":
            groups = params.get("groups", {})
            group = groups.get((i, j), "default")
            return θ.get((i, j), θ.get(group, 1.0))
        
        elif self.constraint_structure == "sparse":
            # ε ~ 1e-6: small but non-zero
            # Avoids numerical issues, allows gradient flow
            return θ.get((i, j), EPSILON)
        
        else:
            raise ValueError(f"Unknown constraint structure: {self.constraint_structure}")
    
    def get_parameter_space(
        self,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ParameterSpace:
        """
        Describe the structural parameter layout without mutating model state.
        
        Returns:
            ParameterSpace capturing canonical ordering and neutral defaults.
        """
        params = parameters if parameters is not None else self.parameters
        theta = params.get("theta", {})
        
        if self.constraint_structure == "per_transition":
            if theta:
                keys = tuple(sorted(theta.keys()))
            else:
                keys = tuple(sorted(self.graph.iter_edges()))
            return ParameterSpace(keys=keys, neutral_value=1.0)
        
        if self.constraint_structure == "per_state":
            if theta:
                keys = tuple(sorted(theta.keys()))
            else:
                keys = tuple(range(len(self.states)))
            return ParameterSpace(keys=keys, neutral_value=1.0)
        
        if self.constraint_structure in {"hierarchical", "sparse"}:
            if self.constraint_structure == "hierarchical":
                groups = params.get("groups", {})
                group_names = tuple(sorted(set(groups.values())))
                transition_keys = tuple(sorted(k for k in theta.keys() if isinstance(k, tuple)))
                keys = group_names + transition_keys
                return ParameterSpace(keys=keys, neutral_value=1.0)
            
            # sparse
            keys = tuple(sorted(theta.keys()))
            return ParameterSpace(keys=keys, neutral_value=EPSILON)
        
        return ParameterSpace(keys=tuple())
    
    def num_free_parameters(self, parameters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count number of free parameters in constraint model.
        
        Essential for AIC, BIC, and likelihood ratio tests.
        
        Args:
            parameters: Constraint parameters (uses self.parameters if None)
            
        Returns:
            Number of free parameters
        """
        params = parameters if parameters is not None else self.parameters
        space = self.get_parameter_space(params)
        return space.size()
    
    def set_parameters(self, **params) -> None:
        """
        Set constraint parameters.
        
        Args:
            **params: Parameter values (theta, groups, etc.)
        """
        self.parameters.update(params)
    
    def pack(self, parameters: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Pack constraint parameters into flat vector.
        
        Bridge between dict representation and optimization.
        Essential for scipy.optimize, JAX, PyTorch.
        
        Args:
            parameters: Constraint parameters (uses self.parameters if None)
            
        Returns:
            Flat parameter vector
        """
        params = parameters if parameters is not None else self.parameters
        θ = params.get("theta", {})
        
        if self.constraint_structure == "per_transition":
            space = self.get_parameter_space(params)
            return np.array([θ.get(k, space.neutral_value) for k in space.keys])
        
        elif self.constraint_structure == "per_state":
            space = self.get_parameter_space(params)
            return np.array([θ.get(k, space.neutral_value) for k in space.keys])
        
        elif self.constraint_structure == "hierarchical":
            # Pack group-level + transition-specific parameters
            groups = params.get("groups", {})
            group_names = sorted(set(groups.values()))
            transition_keys = sorted([k for k in θ.keys() if isinstance(k, tuple)])
            
            vec = []
            # Group-level parameters first
            for g in group_names:
                vec.append(θ.get(g, 1.0))
            # Transition-specific overrides
            for k in transition_keys:
                vec.append(θ[k])
            
            return np.array(vec)
        
        elif self.constraint_structure == "sparse":
            # Pack binary indicators
            keys = sorted(θ.keys())
            return np.array([θ[k] for k in keys])
        
        else:
            raise ValueError(f"Unknown constraint structure: {self.constraint_structure}")
    
    def unpack(self, vector: np.ndarray) -> Dict[str, Any]:
        """
        Unpack flat vector into constraint parameters dict.
        
        Inverse of pack(). Essential for optimization.
        
        Args:
            vector: Flat parameter vector
            
        Returns:
            Constraint parameters dict
        """
        θ = {}
        
        if self.constraint_structure == "per_transition":
            space = self.get_parameter_space()
            
            if len(vector) != space.size():
                raise ValueError(f"Vector length {len(vector)} != expected {space.size()}")
            
            for i, k in enumerate(space.keys):
                θ[k] = float(vector[i])
            
            return {"theta": θ}
        
        elif self.constraint_structure == "per_state":
            space = self.get_parameter_space()
            
            if len(vector) != space.size():
                raise ValueError(f"Vector length {len(vector)} != expected {space.size()}")
            
            for i, k in enumerate(space.keys):
                θ[k] = float(vector[i])
            
            return {"theta": θ}
        
        elif self.constraint_structure == "hierarchical":
            space = self.get_parameter_space()
            groups = self.parameters.get("groups", {})
            
            if len(vector) != space.size():
                raise ValueError(f"Vector length {len(vector)} != expected {space.size()}")
            
            for i, k in enumerate(space.keys):
                θ[k] = float(vector[i])
            
            return {"theta": θ, "groups": groups}
        
        elif self.constraint_structure == "sparse":
            space = self.get_parameter_space()
            
            if len(vector) != space.size():
                raise ValueError(f"Vector length {len(vector)} != expected {space.size()}")
            
            for i, k in enumerate(space.keys):
                θ[k] = float(vector[i])
            
            return {"theta": θ}
        
        else:
            raise ValueError(f"Unknown constraint structure: {self.constraint_structure}")
    
    def initial_parameters(self) -> np.ndarray:
        """
        Get initial parameter vector for optimization.
        
        Returns neutral starting point (θ = 1 everywhere).
        
        Returns:
            Initial parameter vector
        """
        # Start from current parameters or neutral
        if self.parameters.get("theta"):
            return self.pack()
        
        # Default: neutral (θ = 1.0 for all parameters)
        space = self.get_parameter_space()
        return space.neutral_vector()
    
    def get_constrained_baseline(self, parameters: Optional[Dict[str, Any]] = None) -> "Baseline":
        """
        Create a new Baseline object with constrained rates.
        
        GENERATIVE SEMANTICS:
        This returns a new generative process with rates λ*_ij = f(λ_ij, θ).
        The returned Baseline can be used for:
        - Likelihood computation: logL(data | constrained_baseline)
        - Simulation: generate data from constrained process
        - Hypothesis testing: compare baseline vs constrained
        
        PURE FUNCTIONAL: parameters argument allows evaluation without mutating state.
        
        Useful for likelihood comparisons:
        Δ = logL(data | baseline) - logL(data | constrained_baseline)
        
        Args:
            parameters: Optional constraint parameters (uses self.parameters if None)
        
        Returns:
            Baseline with effective rates λ*_ij
        """
        from persiste.core.baseline import Baseline
        
        return Baseline(rate_fn=lambda i, j: self.effective_rate(i, j, parameters))
    
    def fit(
        self,
        data: Any,
        obs_model: Optional[Any] = None,
        method: str = "MLE",
        **kwargs
    ) -> Any:
        """
        Fit constraint parameters to data.
        
        Thin dispatcher to ConstraintInference engine.
        Keeps user-facing API convenient without contaminating model specification.
        
        Args:
            data: Observed transition data
            obs_model: ObservationModel (required for inference)
            method: Inference method ('MLE', 'MCMC', 'variational')
            **kwargs: Method-specific arguments
            
        Returns:
            ConstraintResult object
            
        Raises:
            ValueError: If obs_model not provided
        """
        if obs_model is None:
            raise ValueError(
                "obs_model (ObservationModel) required for inference. "
                "Example: model.fit(data, obs_model=PoissonObservationModel(graph))"
            )
        
        from persiste.core.inference import ConstraintInference
        
        engine = ConstraintInference(self, obs_model)
        return engine.fit(data, method=method, **kwargs)
    
    def test(
        self,
        data: Any,
        null_result: Any,
        alternative_result: Any,
        obs_model: Optional[Any] = None,
        method: str = "LRT",
        **kwargs
    ) -> Any:
        """
        Test hypothesis about constraints.
        
        Thin dispatcher to ConstraintInference engine.
        
        Args:
            data: Observed transition data
            null_result: Fitted null model (ConstraintResult)
            alternative_result: Fitted alternative model (ConstraintResult)
            obs_model: ObservationModel (required for testing)
            method: Test method ('LRT', 'parametric_bootstrap')
            **kwargs: Method-specific arguments
            
        Returns:
            ConstraintTestResult object
            
        Raises:
            ValueError: If obs_model not provided
        """
        if obs_model is None:
            raise ValueError(
                "obs_model (ObservationModel) required for testing. "
                "Example: model.test(data, null, alt, obs_model=PoissonObservationModel(graph))"
            )
        
        from persiste.core.inference import ConstraintInference
        
        engine = ConstraintInference(self, obs_model)
        return engine.test(data, null_result, alternative_result, method=method, **kwargs)
    
    def __repr__(self) -> str:
        return (
            f"ConstraintModel(states={len(self.states)}, "
            f"structure={self.constraint_structure})"
        )
