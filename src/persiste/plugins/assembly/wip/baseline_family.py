"""
Baseline Family: Inferable baseline models.

This makes the baseline a first-class inferable object, preventing
constraints from compensating for baseline errors.

Joint inference: θ, φ = argmax P(data | baseline(φ), constraints(θ))
"""

from typing import Dict, Optional, Callable
import numpy as np
from dataclasses import dataclass

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline


@dataclass
class BaselinePrior:
    """Prior distribution for a baseline parameter."""
    mean: float
    std: float
    
    def log_prob(self, value: float) -> float:
        """Gaussian log-probability."""
        return -0.5 * ((value - self.mean) / self.std) ** 2 - np.log(self.std * np.sqrt(2 * np.pi))


class BaselineFamily:
    """
    Parametric family of baseline models.
    
    Allows joint inference of baseline parameters and constraints,
    preventing constraints from compensating for baseline errors.
    
    Example:
        family = BaselineFamily(
            parameters={'kappa': 1.0, 'join_exponent': -0.5},
            priors={'kappa': BaselinePrior(1.0, 0.2)}
        )
        
        # Joint inference
        theta, phi = inference.fit_joint(data, family)
    """
    
    def __init__(
        self,
        parameters: Dict[str, float],
        priors: Optional[Dict[str, BaselinePrior]] = None,
        fixed_parameters: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize baseline family.
        
        Args:
            parameters: Initial values for inferable parameters
            priors: Prior distributions for parameters (default: weak priors)
            fixed_parameters: Parameters held fixed (e.g., split_exponent)
        """
        self.parameters = parameters.copy()
        self.fixed_parameters = fixed_parameters or {}
        
        # Default weak priors if not specified
        self.priors = priors or {}
        for param in parameters:
            if param not in self.priors:
                self.priors[param] = BaselinePrior(mean=parameters[param], std=0.5)
    
    def create_baseline(self, parameters: Optional[Dict[str, float]] = None) -> AssemblyBaseline:
        """
        Create baseline instance with given parameters.
        
        Args:
            parameters: Parameter values (uses self.parameters if None)
        
        Returns:
            AssemblyBaseline instance
        """
        params = parameters or self.parameters
        
        # Merge with fixed parameters
        all_params = {**self.fixed_parameters, **params}
        
        return AssemblyBaseline(
            kappa=all_params.get('kappa', 1.0),
            join_exponent=all_params.get('join_exponent', -0.5),
            split_exponent=all_params.get('split_exponent', 0.3),
        )
    
    def log_prior(self, parameters: Dict[str, float]) -> float:
        """
        Compute log-prior probability for parameters.
        
        Args:
            parameters: Parameter values
        
        Returns:
            Log-prior probability
        """
        log_p = 0.0
        for param, value in parameters.items():
            if param in self.priors:
                log_p += self.priors[param].log_prob(value)
        return log_p
    
    def get_parameter_names(self) -> list:
        """Get list of inferable parameter names."""
        return list(self.parameters.keys())
    
    def get_bounds(self) -> Dict[str, tuple]:
        """
        Get reasonable bounds for parameters.
        
        Returns:
            Dict mapping parameter names to (lower, upper) bounds
        """
        bounds = {
            'kappa': (0.1, 10.0),
            'join_exponent': (-2.0, 0.0),
            'split_exponent': (-1.0, 2.0),
        }
        return {k: bounds[k] for k in self.parameters.keys() if k in bounds}


class FixedBaseline(BaselineFamily):
    """
    Fixed baseline (no inference).
    
    This is the default behavior - baseline is treated as ground truth.
    Use this when you're confident in your baseline model.
    """
    
    def __init__(self, baseline: AssemblyBaseline):
        """
        Initialize with fixed baseline.
        
        Args:
            baseline: Fixed baseline instance
        """
        super().__init__(
            parameters={},
            fixed_parameters={
                'kappa': baseline.kappa,
                'join_exponent': baseline.join_exponent,
                'split_exponent': baseline.split_exponent,
            }
        )
        self._baseline = baseline
    
    def create_baseline(self, parameters: Optional[Dict[str, float]] = None) -> AssemblyBaseline:
        """Return fixed baseline (ignores parameters)."""
        return self._baseline


class SimpleBaselineFamily(BaselineFamily):
    """
    Simple baseline family with one inferable parameter.
    
    This is the recommended starting point - allows baseline to adjust
    without over-parameterization.
    
    Example:
        # Infer only the join exponent
        family = SimpleBaselineFamily(
            parameter='join_exponent',
            initial_value=-0.5,
            prior_std=0.2
        )
    """
    
    def __init__(
        self,
        parameter: str = 'join_exponent',
        initial_value: float = -0.5,
        prior_std: float = 0.2,
        kappa: float = 1.0,
        split_exponent: float = 0.3,
    ):
        """
        Initialize simple baseline family.
        
        Args:
            parameter: Which parameter to infer ('kappa', 'join_exponent', or 'split_exponent')
            initial_value: Initial value for inferable parameter
            prior_std: Prior standard deviation (controls regularization)
            kappa: Fixed kappa (if not the inferable parameter)
            split_exponent: Fixed split_exponent (if not the inferable parameter)
        """
        parameters = {parameter: initial_value}
        priors = {parameter: BaselinePrior(mean=initial_value, std=prior_std)}
        
        fixed = {}
        if parameter != 'kappa':
            fixed['kappa'] = kappa
        if parameter != 'join_exponent':
            fixed['join_exponent'] = -0.5
        if parameter != 'split_exponent':
            fixed['split_exponent'] = split_exponent
        
        super().__init__(
            parameters=parameters,
            priors=priors,
            fixed_parameters=fixed,
        )
