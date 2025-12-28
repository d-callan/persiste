"""
Core tree-based inference utilities for PERSISTE.

Provides generic MLE optimization and hypothesis testing for models
that compute likelihoods on phylogenetic trees. Plugins provide:
1. Parameter bounds and initial values
2. Likelihood function (via pruning)
3. Null/alternative model specifications

The core handles:
1. Optimization (scipy.optimize)
2. Uncertainty estimation (Hessian, profile likelihood)
3. Hypothesis testing (LRT, parametric bootstrap)

Key design principles:
1. Separation of concerns - core does optimization, plugins do models
2. Flexible parameterization - plugins define parameter space
3. Robust defaults - sensible optimization settings out of the box
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import warnings

try:
    from scipy import optimize
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class MLEResult:
    """
    Result of maximum likelihood estimation.
    
    Attributes:
        parameters: Dict of fitted parameter values
        log_likelihood: Log-likelihood at MLE
        n_parameters: Number of free parameters
        n_observations: Number of observations (sites/families)
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        standard_errors: Optional standard errors from Hessian
        convergence: Whether optimization converged
        message: Optimization message
        n_iterations: Number of iterations
        n_function_evals: Number of function evaluations
    """
    parameters: Dict[str, float]
    log_likelihood: float
    n_parameters: int
    n_observations: int
    aic: float
    bic: float
    standard_errors: Optional[Dict[str, float]] = None
    convergence: bool = True
    message: str = ""
    n_iterations: int = 0
    n_function_evals: int = 0
    
    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v:.4f}" for k, v in self.parameters.items())
        return (
            f"MLEResult({params_str}, "
            f"LL={self.log_likelihood:.4f}, "
            f"AIC={self.aic:.2f})"
        )


@dataclass
class LRTResult:
    """
    Result of likelihood ratio test.
    
    Attributes:
        null_result: MLE result for null model
        alt_result: MLE result for alternative model
        statistic: LRT statistic (2 × Δ log-likelihood)
        df: Degrees of freedom
        pvalue: p-value from chi-squared distribution
        significant: Whether test is significant at alpha level
        alpha: Significance level used
    """
    null_result: MLEResult
    alt_result: MLEResult
    statistic: float
    df: int
    pvalue: float
    significant: bool
    alpha: float = 0.05
    
    def __repr__(self) -> str:
        sig_str = "***" if self.pvalue < 0.001 else "**" if self.pvalue < 0.01 else "*" if self.pvalue < 0.05 else ""
        return (
            f"LRTResult(statistic={self.statistic:.4f}, "
            f"df={self.df}, "
            f"p={self.pvalue:.4e}{sig_str})"
        )


class TreeLikelihoodModel(ABC):
    """
    Abstract base class for tree-based likelihood models.
    
    Plugins implement this to define their parameter space and
    likelihood computation. The core optimizer uses this interface.
    """
    
    @abstractmethod
    def get_parameter_names(self) -> List[str]:
        """Return list of parameter names."""
        ...
    
    @abstractmethod
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for each parameter as (lower, upper)."""
        ...
    
    @abstractmethod
    def get_initial_parameters(self) -> Dict[str, float]:
        """Return initial parameter values for optimization."""
        ...
    
    @abstractmethod
    def log_likelihood(self, parameters: Dict[str, float]) -> float:
        """
        Compute log-likelihood at given parameters.
        
        Args:
            parameters: Dict mapping parameter names to values
            
        Returns:
            Log-likelihood value
        """
        ...
    
    @property
    @abstractmethod
    def n_observations(self) -> int:
        """Number of observations (sites, families, etc.)."""
        ...


class TreeMLEOptimizer:
    """
    Maximum likelihood optimizer for tree-based models.
    
    Wraps scipy.optimize with sensible defaults for phylogenetic
    likelihood optimization.
    
    Usage:
        model = MyTreeModel(tree, data, ...)
        optimizer = TreeMLEOptimizer(model)
        result = optimizer.fit()
    """
    
    def __init__(
        self,
        model: TreeLikelihoodModel,
        method: str = "L-BFGS-B",
        tol: float = 1e-6,
        maxiter: int = 1000,
    ):
        """
        Initialize optimizer.
        
        Args:
            model: TreeLikelihoodModel to optimize
            method: Optimization method (default: L-BFGS-B for bounded)
            tol: Convergence tolerance
            maxiter: Maximum iterations
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy required for optimization. "
                "Install with: pip install scipy"
            )
        
        self.model = model
        self.method = method
        self.tol = tol
        self.maxiter = maxiter
    
    def fit(
        self,
        initial_params: Optional[Dict[str, float]] = None,
        compute_se: bool = True,
    ) -> MLEResult:
        """
        Fit model by maximum likelihood.
        
        Args:
            initial_params: Optional initial parameters (uses model defaults if None)
            compute_se: Whether to compute standard errors from Hessian
            
        Returns:
            MLEResult with fitted parameters
        """
        param_names = self.model.get_parameter_names()
        bounds_dict = self.model.get_parameter_bounds()
        
        # Initial parameters
        if initial_params is None:
            initial_params = self.model.get_initial_parameters()
        
        # Convert to arrays for scipy
        x0 = np.array([initial_params[name] for name in param_names])
        bounds = [bounds_dict[name] for name in param_names]
        
        # Negative log-likelihood for minimization
        def neg_ll(x: np.ndarray) -> float:
            params = {name: val for name, val in zip(param_names, x)}
            ll = self.model.log_likelihood(params)
            return -ll
        
        # Optimize
        result = optimize.minimize(
            neg_ll,
            x0,
            method=self.method,
            bounds=bounds,
            options={
                'maxiter': self.maxiter,
                'ftol': self.tol,
                'gtol': self.tol,
            }
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        # Extract fitted parameters
        fitted_params = {name: val for name, val in zip(param_names, result.x)}
        log_lik = -result.fun
        
        # Compute standard errors from Hessian
        standard_errors = None
        if compute_se:
            try:
                # Numerical Hessian
                hess = self._compute_hessian(neg_ll, result.x)
                # Standard errors from inverse Hessian
                try:
                    cov = np.linalg.inv(hess)
                    se = np.sqrt(np.diag(cov))
                    standard_errors = {name: val for name, val in zip(param_names, se)}
                except np.linalg.LinAlgError:
                    warnings.warn("Could not compute standard errors (singular Hessian)")
            except Exception as e:
                warnings.warn(f"Could not compute standard errors: {e}")
        
        # Compute AIC and BIC
        k = len(param_names)
        n = self.model.n_observations
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik if n > 0 else float('inf')
        
        return MLEResult(
            parameters=fitted_params,
            log_likelihood=log_lik,
            n_parameters=k,
            n_observations=n,
            aic=aic,
            bic=bic,
            standard_errors=standard_errors,
            convergence=result.success,
            message=result.message,
            n_iterations=result.nit if hasattr(result, 'nit') else 0,
            n_function_evals=result.nfev,
        )
    
    def _compute_hessian(
        self,
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """Compute numerical Hessian using finite differences."""
        n = len(x)
        hess = np.zeros((n, n))
        
        f0 = func(x)
        
        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[i] += eps
                x_pp[j] += eps
                x_pm[i] += eps
                x_pm[j] -= eps
                x_mp[i] -= eps
                x_mp[j] += eps
                x_mm[i] -= eps
                x_mm[j] -= eps
                
                hess[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * eps * eps)
                hess[j, i] = hess[i, j]
        
        return hess


def likelihood_ratio_test(
    null_result: MLEResult,
    alt_result: MLEResult,
    alpha: float = 0.05,
) -> LRTResult:
    """
    Perform likelihood ratio test comparing two nested models.
    
    Args:
        null_result: MLE result for null (simpler) model
        alt_result: MLE result for alternative (more complex) model
        alpha: Significance level
        
    Returns:
        LRTResult with test statistics
        
    Raises:
        ValueError: If alternative doesn't have more parameters
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for LRT")
    
    # Degrees of freedom
    df = alt_result.n_parameters - null_result.n_parameters
    
    if df <= 0:
        raise ValueError(
            f"Alternative model must have more parameters than null. "
            f"Got null={null_result.n_parameters}, alt={alt_result.n_parameters}"
        )
    
    # LRT statistic: 2 × (ℓ_alt - ℓ_null)
    statistic = 2 * (alt_result.log_likelihood - null_result.log_likelihood)
    
    # Handle negative statistic (can happen due to optimization issues)
    if statistic < 0:
        warnings.warn(
            f"Negative LRT statistic ({statistic:.4f}). "
            "This may indicate optimization issues."
        )
        pvalue = 1.0
    else:
        # p-value from chi-squared distribution
        pvalue = stats.chi2.sf(statistic, df)
    
    return LRTResult(
        null_result=null_result,
        alt_result=alt_result,
        statistic=statistic,
        df=df,
        pvalue=pvalue,
        significant=pvalue < alpha,
        alpha=alpha,
    )


def model_selection(
    results: List[MLEResult],
    criterion: str = "AIC",
) -> Tuple[int, MLEResult]:
    """
    Select best model using information criterion.
    
    Args:
        results: List of MLEResult objects to compare
        criterion: "AIC" or "BIC"
        
    Returns:
        Tuple of (best_index, best_result)
    """
    if criterion == "AIC":
        scores = [r.aic for r in results]
    elif criterion == "BIC":
        scores = [r.bic for r in results]
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    best_idx = np.argmin(scores)
    return best_idx, results[best_idx]
