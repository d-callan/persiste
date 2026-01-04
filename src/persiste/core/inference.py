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

        theta0 = np.array(self.model.initial_parameters(), dtype=float)
        if theta0.ndim == 0:
            theta0 = theta0.reshape(1)
        minimize_kwargs = dict(kwargs)
        user_bounds = minimize_kwargs.pop("bounds", None)
        options = minimize_kwargs.pop("options", None) or {}
        minimize_method = minimize_kwargs.pop("minimize_method", None)

        def _effective_sample_size(data_obj: Any) -> int:
            if hasattr(data_obj, "total_transitions"):
                total = data_obj.total_transitions()
            elif hasattr(data_obj, "__len__"):
                try:
                    total = len(data_obj)  # type: ignore[arg-type]
                except TypeError:
                    total = 0
            else:
                total = 0
            return max(int(total), 1)

        def _vector_to_params(vector: np.ndarray) -> Dict[str, Any]:
            vec = np.array(vector, dtype=float)
            return self.model.unpack(vec)

        if theta0.size == 0:
            fitted_params = _vector_to_params(theta0)
            ll_hat = float(self.log_likelihood(data, fitted_params))
            k = self.model.num_free_parameters(fitted_params)
            n_eff = _effective_sample_size(data)
            aic = 2 * k - 2 * ll_hat
            bic = k * np.log(n_eff) - 2 * ll_hat if n_eff > 0 else None
            metadata = {
                "success": True,
                "message": "No free parameters to optimize",
                "nfev": 0,
                "nit": 0,
            }
            return ConstraintResult(
                model=self.model,
                parameters=fitted_params,
                method="MLE",
                log_likelihood=ll_hat,
                aic=aic,
                bic=bic,
                metadata=metadata,
            )

        if minimize_method is None:
            minimize_method = "L-BFGS-B"

        if user_bounds is not None:
            bounds = user_bounds
        else:
            lower = 0.0
            upper = 1.0 if not self.model.allow_facilitation else None
            bounds = [(lower, upper) for _ in range(theta0.size)]

        if bounds is not None and len(bounds) != theta0.size:
            raise ValueError(
                f"Bounds length {len(bounds)} does not match parameter vector length {theta0.size}"
            )

        eval_cache: Dict[str, Any] = {}

        def objective(vector: np.ndarray) -> float:
            params = _vector_to_params(vector)
            ll = float(self.log_likelihood(data, params))
            eval_cache["last_params"] = params
            eval_cache["last_ll"] = ll
            if not np.isfinite(ll):
                return np.inf
            return -ll

        result = optimize.minimize(
            objective,
            theta0,
            method=minimize_method,
            bounds=bounds,
            options=options,
            **minimize_kwargs,
        )

        best_vector = np.array(result.x, dtype=float)
        fitted_params = _vector_to_params(best_vector)
        ll_hat = float(-result.fun) if result.fun is not None else float(eval_cache.get("last_ll", float("-inf")))
        k = self.model.num_free_parameters(fitted_params)
        n_eff = _effective_sample_size(data)
        aic = 2 * k - 2 * ll_hat
        bic = k * np.log(n_eff) - 2 * ll_hat if n_eff > 0 else None
        metadata = {
            "success": bool(result.success),
            "message": result.message,
            "nfev": getattr(result, "nfev", None),
            "nit": getattr(result, "nit", None),
            "fun": result.fun,
        }

        return ConstraintResult(
            model=self.model,
            parameters=fitted_params,
            method="MLE",
            log_likelihood=ll_hat,
            aic=aic,
            bic=bic,
            metadata=metadata,
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
