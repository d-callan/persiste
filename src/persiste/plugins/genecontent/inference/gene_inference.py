"""
GeneContent inference using core framework utilities.

Provides MLE inference for gene family gain/loss models using:
1. Core TreeStructure for tree representation
2. Core FelsensteinPruning for likelihood computation
3. Core TreeMLEOptimizer for parameter estimation
4. Core likelihood_ratio_test for hypothesis testing

This demonstrates how plugins leverage core utilities while
providing domain-specific model definitions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from persiste.core.trees import TreeStructure
from persiste.core.pruning import (
    FelsensteinPruning,
    SimpleBinaryTransitionProvider,
    ArrayTipConditionalProvider,
)
from persiste.core.tree_inference import (
    TreeLikelihoodModel,
    TreeMLEOptimizer,
    MLEResult,
    LRTResult,
    likelihood_ratio_test,
)

from ..baselines.gene_baseline import (
    GeneContentBaseline,
    GlobalRates,
    RateParameters,
)
from ..constraints.gene_constraint import (
    GeneContentConstraint,
    NullConstraint,
    ConstraintEffect,
)


@dataclass
class GeneContentData:
    """
    Gene content data for inference.
    
    Attributes:
        tree: TreeStructure with phylogeny
        presence_matrix: (n_taxa, n_families) binary matrix
        taxon_names: List of taxon names (row order)
        family_names: List of family names (column order)
    """
    tree: TreeStructure
    presence_matrix: np.ndarray
    taxon_names: List[str]
    family_names: List[str]
    
    @property
    def n_taxa(self) -> int:
        return len(self.taxon_names)
    
    @property
    def n_families(self) -> int:
        return len(self.family_names)
    
    def __repr__(self) -> str:
        return f"GeneContentData({self.n_taxa} taxa, {self.n_families} families)"


class GeneContentModel(TreeLikelihoodModel):
    """
    Gene content likelihood model for MLE optimization.
    
    Implements TreeLikelihoodModel interface to work with core optimizer.
    
    Parameters:
        - log_gain: Log of gain rate (λ = exp(log_gain))
        - log_loss: Log of loss rate (μ = exp(log_loss))
        - constraint_params: Optional constraint-specific parameters
    
    Usage:
        data = GeneContentData(tree, matrix, taxa, families)
        model = GeneContentModel(data)
        optimizer = TreeMLEOptimizer(model)
        result = optimizer.fit()
    """
    
    def __init__(
        self,
        data: GeneContentData,
        constraint: Optional[GeneContentConstraint] = None,
        use_jax: bool = False,
    ):
        """
        Initialize gene content model.
        
        Args:
            data: GeneContentData with tree and presence matrix
            constraint: Optional constraint model (default: NullConstraint)
            use_jax: Whether to use JAX acceleration
        """
        self.data = data
        self.constraint = constraint or NullConstraint()
        self.use_jax = use_jax
        
        # Setup pruning algorithm
        self._pruning = FelsensteinPruning(
            data.tree,
            n_states=2,
            use_jax=use_jax,
        )
        
        # Parameter names depend on constraint
        self._param_names = ['log_gain', 'log_loss']
        self._param_bounds = {
            'log_gain': (-5.0, 2.0),  # exp(-5) ≈ 0.007, exp(2) ≈ 7.4
            'log_loss': (-5.0, 2.0),
        }
        self._initial_params = {
            'log_gain': -1.0,  # exp(-1) ≈ 0.37
            'log_loss': -1.0,
        }
        
        # Add constraint parameters if needed
        constraint_params = self.constraint.get_parameter_names()
        for name in constraint_params:
            self._param_names.append(name)
            self._param_bounds[name] = self.constraint.get_parameter_bounds(name)
            self._initial_params[name] = self.constraint.get_initial_value(name)
    
    def get_parameter_names(self) -> List[str]:
        return self._param_names
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        return self._param_bounds
    
    def get_initial_parameters(self) -> Dict[str, float]:
        return self._initial_params.copy()
    
    @property
    def n_observations(self) -> int:
        return self.data.n_families
    
    def log_likelihood(self, parameters: Dict[str, float]) -> float:
        """
        Compute log-likelihood at given parameters.
        
        Sums log-likelihoods across all gene families.
        """
        # Extract rate parameters
        gain_rate = np.exp(parameters['log_gain'])
        loss_rate = np.exp(parameters['log_loss'])
        
        # Update constraint parameters if present
        constraint_params = {
            k: v for k, v in parameters.items()
            if k not in ('log_gain', 'log_loss')
        }
        if constraint_params:
            self.constraint.set_parameters(constraint_params)
        
        total_ll = 0.0
        
        for fam_idx, fam_name in enumerate(self.data.family_names):
            # Get constraint effect for this family
            effect = self.constraint.get_effect(fam_name)
            
            # Apply constraint to rates
            effective_gain = gain_rate * effect.gain_multiplier
            effective_loss = loss_rate * effect.loss_multiplier
            
            # Create transition provider
            transition_provider = SimpleBinaryTransitionProvider(
                gain_rate=effective_gain,
                loss_rate=effective_loss,
            )
            
            # Create tip provider for this family
            single_family_data = self.data.presence_matrix[:, fam_idx:fam_idx+1]
            tip_provider = ArrayTipConditionalProvider(
                data=single_family_data,
                taxon_names=self.data.taxon_names,
                n_states=2,
            )
            
            # Compute likelihood
            result = self._pruning.compute_likelihood(
                transition_provider=transition_provider,
                tip_provider=tip_provider,
                n_sites=1,
            )
            
            total_ll += result.log_likelihood
        
        return total_ll


class GeneContentInference:
    """
    High-level inference interface for gene content models.
    
    Provides convenient methods for:
    1. Fitting null and alternative models
    2. Likelihood ratio testing
    3. Model comparison
    
    Usage:
        inference = GeneContentInference(data)
        
        # Fit null model
        null_result = inference.fit_null()
        
        # Fit with retention constraint
        alt_result = inference.fit_with_constraint(
            RetentionBiasConstraint(retained_families={'OG0001'})
        )
        
        # Test significance
        lrt = inference.likelihood_ratio_test(null_result, alt_result)
    """
    
    def __init__(
        self,
        data: GeneContentData,
        use_jax: bool = False,
    ):
        """
        Initialize inference engine.
        
        Args:
            data: GeneContentData with tree and presence matrix
            use_jax: Whether to use JAX acceleration
        """
        self.data = data
        self.use_jax = use_jax
    
    def fit_null(
        self,
        initial_params: Optional[Dict[str, float]] = None,
    ) -> MLEResult:
        """
        Fit null model (no constraints, global rates).
        
        Args:
            initial_params: Optional initial parameters
            
        Returns:
            MLEResult with fitted parameters
        """
        model = GeneContentModel(
            self.data,
            constraint=NullConstraint(),
            use_jax=self.use_jax,
        )
        
        optimizer = TreeMLEOptimizer(model)
        return optimizer.fit(initial_params=initial_params)
    
    def fit_with_constraint(
        self,
        constraint: GeneContentConstraint,
        initial_params: Optional[Dict[str, float]] = None,
    ) -> MLEResult:
        """
        Fit model with specified constraint.
        
        Args:
            constraint: GeneContentConstraint to apply
            initial_params: Optional initial parameters
            
        Returns:
            MLEResult with fitted parameters
        """
        model = GeneContentModel(
            self.data,
            constraint=constraint,
            use_jax=self.use_jax,
        )
        
        optimizer = TreeMLEOptimizer(model)
        return optimizer.fit(initial_params=initial_params)
    
    def likelihood_ratio_test(
        self,
        null_result: MLEResult,
        alt_result: MLEResult,
        alpha: float = 0.05,
    ) -> LRTResult:
        """
        Perform likelihood ratio test.
        
        Args:
            null_result: MLE result for null model
            alt_result: MLE result for alternative model
            alpha: Significance level
            
        Returns:
            LRTResult with test statistics
        """
        return likelihood_ratio_test(null_result, alt_result, alpha=alpha)
    
    def fit_and_test(
        self,
        constraint: GeneContentConstraint,
        alpha: float = 0.05,
    ) -> Tuple[MLEResult, MLEResult, LRTResult]:
        """
        Fit null and alternative models, then test.
        
        Convenience method that does everything in one call.
        
        Args:
            constraint: GeneContentConstraint for alternative model
            alpha: Significance level
            
        Returns:
            Tuple of (null_result, alt_result, lrt_result)
        """
        null_result = self.fit_null()
        alt_result = self.fit_with_constraint(constraint)
        lrt_result = self.likelihood_ratio_test(null_result, alt_result, alpha=alpha)
        
        return null_result, alt_result, lrt_result
