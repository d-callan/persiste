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
        
        raise NotImplementedError(
            "ConstraintInference.fit() not yet implemented. "
            "Full MLE inference pipeline will land in a future change."
        )
        
        # The remainder of the MLE path will be implemented once the optimizer
        # contract is finalized. For now the explicit NotImplemented above will
        # halt execution before reaching this block.
    
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
