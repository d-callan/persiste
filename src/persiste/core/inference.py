"""Inference engine for constraint models.

Separates inference logic from model specification:
- ConstraintModel: defines what θ means
- Baseline: defines what rates mean
- ObservationModel: defines how data arises
- ConstraintInference: defines how θ is inferred
- ConstraintResult: defines what came out
"""

from typing import Any, Optional, Dict, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

try:
    from scipy import optimize
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

if TYPE_CHECKING:
    from persiste.core.constraints import ConstraintModel
    from persiste.core.observation_models import ObservationModel
    from persiste.core.data import ObservedTransitions


@dataclass
class ConstraintResult:
    """
    Result of constraint parameter inference.
    
    Pure output object representing one fitted member of a constraint model family.
    Contains estimated parameters, uncertainty, and fit statistics.
    
    Does NOT contain:
    - Likelihood code
    - Optimization logic
    - Sampling logic
    
    That all lives in ConstraintInference.
    
    Attributes:
        model: ConstraintModel specification used
        parameters: Fitted constraint parameters θ̂
        method: Inference method used ("MLE", "MCMC", "variational", etc.)
        log_likelihood: Log-likelihood at fitted parameters
        aic: Akaike Information Criterion (optional)
        bic: Bayesian Information Criterion (optional)
        uncertainty: Parameter uncertainty (SEs, posterior intervals, etc.)
        metadata: Additional inference metadata
    """
    
    model: "ConstraintModel"
    parameters: Dict[str, Any]
    method: str
    log_likelihood: float
    aic: Optional[float] = None
    bic: Optional[float] = None
    uncertainty: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        n_params = self.model.num_free_parameters(self.parameters)
        return (
            f"ConstraintResult(method={self.method}, "
            f"log_likelihood={self.log_likelihood:.4f}, "
            f"n_params={n_params})"
        )


@dataclass
class ConstraintTestResult:
    """
    Result of hypothesis test comparing constraint models.
    
    Typically used for likelihood ratio tests:
    - Null: simpler model (e.g., no constraint)
    - Alternative: more complex model (e.g., with constraint)
    
    Attributes:
        null: Fitted null model result
        alternative: Fitted alternative model result
        statistic: Test statistic (e.g., 2 × Δ log-likelihood for LRT)
        pvalue: p-value under null distribution
        method: Test method ("LRT", "parametric_bootstrap", etc.)
        metadata: Additional test metadata
    """
    
    null: ConstraintResult
    alternative: ConstraintResult
    statistic: float
    pvalue: float
    method: str = "LRT"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (
            f"ConstraintTestResult(method={self.method}, "
            f"statistic={self.statistic:.4f}, "
            f"pvalue={self.pvalue:.4e})"
        )


class ConstraintInference:
    """
    Inference engine for constraint models.
    
    Separates inference logic from model specification.
    Handles likelihood evaluation, optimization, and sampling.
    
    Likelihood evaluation pipeline:
        θ → ConstraintModel.effective_rate(i, j)
          → Constrained Baseline
          → ObservationModel.log_likelihood(data, baseline*, graph)
    
    Attributes:
        model: ConstraintModel specification
        obs_model: ObservationModel for likelihood computation
    """
    
    def __init__(
        self,
        model: "ConstraintModel",
        obs_model: "ObservationModel",
    ):
        """
        Initialize inference engine.
        
        Args:
            model: ConstraintModel defining constraint structure
            obs_model: ObservationModel defining how data arises
        """
        self.model = model
        self.obs_model = obs_model
    
    def log_likelihood(
        self,
        data: "ObservedTransitions",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Evaluate log-likelihood at given parameters.
        
        Pipeline: θ → constrained baseline → observation model → log-likelihood
        
        PURE FUNCTIONAL: No state mutation. Parameters passed through pipeline.
        Essential for autodiff, vectorization, and parallel optimization.
        
        Args:
            data: Observed transition data
            parameters: Constraint parameters (uses model.parameters if None)
            
        Returns:
            Log-likelihood value
        """
        # Get constrained baseline (pure functional, no state mutation)
        constrained_baseline = self.model.get_constrained_baseline(parameters)
        
        # Evaluate likelihood
        # Graph belongs to observation model, not constraint model
        ll = self.obs_model.log_likelihood(data, constrained_baseline, self.obs_model.graph)
        
        return ll
    
    def fit(
        self,
        data: "ObservedTransitions",
        method: str = "MLE",
        **kwargs
    ) -> ConstraintResult:
        """
        Fit constraint parameters to data.
        
        Minimal viable MLE implementation using scipy.optimize.
        Everything else (priors, sparsity, hierarchies) layers on top.
        
        Args:
            data: Observed transition data
            method: Inference method ("MLE", "MCMC", "variational")
            **kwargs: Method-specific arguments
                - bounds: Parameter bounds (default: [0, inf) or [0, 1] if no facilitation)
                - options: scipy.optimize options
            
        Returns:
            ConstraintResult with fitted parameters
            
        Raises:
            ImportError: If scipy not available
            NotImplementedError: If method != "MLE"
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy required for inference. Install with: pip install scipy"
            )
        
        if method != "MLE":
            raise NotImplementedError(
                f"Inference method '{method}' not yet implemented. "
                "Currently only 'MLE' is supported."
            )
        
        # Negative log-likelihood for minimization
        def nll(theta_vec):
            params = self.model.unpack(theta_vec)
            ll = self.log_likelihood(data, params)
            
            # Add sparsity regularization for sparse constraint structure
            if self.model.constraint_structure == "sparse":
                if self.model.sparsity == "soft":
                    # Bayesian shrinkage: log-prior term
                    # Prior: θ ~ Gamma(α, β) with strong shrinkage
                    # Equivalent to adding -log p(θ) to negative log-likelihood
                    α = 1.0  # Shape parameter
                    β = self.model.strength  # Rate parameter (higher = stronger shrinkage)
                    
                    log_prior = 0.0
                    for θ_val in theta_vec:
                        # Gamma log-prior: (α-1)log(θ) - βθ
                        log_prior += (α - 1) * np.log(θ_val + 1e-10) - β * θ_val
                    
                    # Return negative (log-likelihood + log-prior)
                    return -(ll + log_prior)
                
                elif self.model.sparsity == "penalized":
                    # Penalized MLE: L1 penalty
                    # Objective = log-likelihood - λ||θ||_1
                    penalty = self.model.strength * np.sum(theta_vec)
                    return -(ll - penalty)
                
                elif self.model.sparsity == "latent":
                    # Spike-and-slab mixture model
                    # TODO: Implement EM or MCMC for latent indicators
                    raise NotImplementedError(
                        "Latent spike-and-slab not yet implemented. "
                        "Use 'soft' or 'penalized' for now."
                    )
                
                else:
                    raise ValueError(f"Unknown sparsity mode: {self.model.sparsity}")
            
            return -ll
        
        # Initial parameters (neutral: θ = 1 everywhere)
        theta0 = self.model.initial_parameters()
        
        # Parameter bounds
        if "bounds" in kwargs:
            bounds = kwargs["bounds"]
        else:
            # Default bounds based on facilitation policy
            if self.model.allow_facilitation:
                # Allow θ > 1 (facilitation)
                bounds = [(0.0, None) for _ in theta0]
            else:
                # Pure constraint: θ ∈ [0, 1]
                bounds = [(0.0, 1.0) for _ in theta0]
        
        # Optimize
        result = optimize.minimize(
            nll,
            theta0,
            bounds=bounds,
            options=kwargs.get("options", {})
        )
        
        if not result.success:
            import warnings
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        # Unpack fitted parameters
        fitted_params = self.model.unpack(result.x)
        
        # Compute AIC and BIC
        k = self.model.num_free_parameters(fitted_params)
        n = data.total_transitions()
        ll = -result.fun
        
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll
        
        return ConstraintResult(
            model=self.model,
            parameters=fitted_params,
            method="MLE",
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            metadata={
                "success": result.success,
                "message": result.message,
                "nfev": result.nfev,
            }
        )
    
    def test(
        self,
        data: "ObservedTransitions",
        null_result: ConstraintResult,
        alternative_result: ConstraintResult,
        method: str = "LRT",
        **kwargs
    ) -> ConstraintTestResult:
        """
        Test hypothesis about constraints.
        
        Likelihood ratio test becomes trivial once ConstraintResult exists.
        No model logic involved.
        
        Args:
            data: Observed transition data
            null_result: Fitted null model
            alternative_result: Fitted alternative model
            method: Test method ("LRT", "parametric_bootstrap")
            **kwargs: Method-specific arguments
            
        Returns:
            ConstraintTestResult with test statistics
            
        Raises:
            ImportError: If scipy not available
            NotImplementedError: If method != "LRT"
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy required for hypothesis testing. Install with: pip install scipy"
            )
        
        if method == "LRT":
            # Likelihood ratio test
            # Statistic: 2 × (ℓ_alt - ℓ_null)
            # Distribution: χ² with df = k_alt - k_null
            
            ll_null = null_result.log_likelihood
            ll_alt = alternative_result.log_likelihood
            
            k_null = null_result.model.num_free_parameters(null_result.parameters)
            k_alt = alternative_result.model.num_free_parameters(alternative_result.parameters)
            
            # Test statistic
            statistic = 2 * (ll_alt - ll_null)
            
            # Degrees of freedom
            df = k_alt - k_null
            
            if df <= 0:
                raise ValueError(
                    f"Alternative must have more parameters than null. "
                    f"Got df = {df} (k_alt={k_alt}, k_null={k_null})"
                )
            
            # p-value from χ² distribution
            pvalue = stats.chi2.sf(statistic, df)
            
            return ConstraintTestResult(
                null=null_result,
                alternative=alternative_result,
                statistic=statistic,
                pvalue=pvalue,
                method="LRT",
                metadata={
                    "df": df,
                    "k_null": k_null,
                    "k_alt": k_alt,
                }
            )
        
        else:
            raise NotImplementedError(f"Test method '{method}' not yet implemented")
