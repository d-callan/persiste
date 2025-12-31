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
import sys

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

# Try to import Rust acceleration
try:
    from persiste.core.pruning_rust import (
        compute_likelihoods_batch,
        check_rust_available,
    )
    RUST_AVAILABLE = check_rust_available()
except ImportError:
    RUST_AVAILABLE = False
    compute_likelihoods_batch = None

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
        use_rust: bool = True,
    ):
        """
        Initialize gene content model.
        
        Args:
            data: GeneContentData with tree and presence matrix
            constraint: Optional constraint model (default: NullConstraint)
            use_jax: Whether to use JAX acceleration (for NumPy backend)
            use_rust: Whether to use Rust parallelization (default: True, auto-fallback to NumPy)
        """
        self.data = data
        self.constraint = constraint or NullConstraint()
        self.use_jax = use_jax
        self.use_rust = use_rust and RUST_AVAILABLE
        
        # Log backend selection
        if use_rust and not RUST_AVAILABLE:
            import warnings
            warnings.warn("Rust backend requested but not available. Falling back to NumPy.")
        
        # Setup pruning algorithm (only needed for NumPy backend)
        self._pruning = FelsensteinPruning(
            data.tree,
            n_states=2,
            use_jax=use_jax,
        ) if not self.use_rust else None
        
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
        Uses Rust parallelization when available for 100-300x speedup.
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
        
        # Check if we can use Rust fast path (no constraints or global constraint)
        if self.use_rust and self.constraint.n_parameters() == 0:
            # Fast path: Rust parallelization with uniform rates
            gain_rates = np.full(self.data.n_families, gain_rate)
            loss_rates = np.full(self.data.n_families, loss_rate)
            
            log_liks = compute_likelihoods_batch(
                self.data.tree,
                self.data.presence_matrix,
                gain_rates,
                loss_rates,
                self.data.taxon_names,
                use_rust=True,
            )
            total_ll = np.sum(log_liks)
        
        elif self.use_rust and isinstance(self.constraint, NullConstraint):
            # Fast path: Rust with null constraint
            gain_rates = np.full(self.data.n_families, gain_rate)
            loss_rates = np.full(self.data.n_families, loss_rate)
            
            log_liks = compute_likelihoods_batch(
                self.data.tree,
                self.data.presence_matrix,
                gain_rates,
                loss_rates,
                self.data.taxon_names,
                use_rust=True,
            )
            total_ll = np.sum(log_liks)
        
        elif self.use_rust:
            # Rust with per-family constraints
            gain_rates = np.zeros(self.data.n_families)
            loss_rates = np.zeros(self.data.n_families)
            
            for fam_idx, fam_name in enumerate(self.data.family_names):
                effect = self.constraint.get_effect(fam_name)
                gain_rates[fam_idx] = gain_rate * effect.gain_multiplier
                loss_rates[fam_idx] = loss_rate * effect.loss_multiplier
            
            log_liks = compute_likelihoods_batch(
                self.data.tree,
                self.data.presence_matrix,
                gain_rates,
                loss_rates,
                self.data.taxon_names,
                use_rust=True,
            )
            total_ll = np.sum(log_liks)
        
        else:
            # Slow path: NumPy sequential computation
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
        
        # Add constraint prior (Fix #3: hierarchical shrinkage)
        # This implements MAP estimation instead of pure MLE
        total_ll += self.constraint.log_prior()
        
        # Add weak prior on baseline rates for regularization
        # This provides mild stability without changing behavior significantly
        # Prior: log_gain ~ N(0, 4), log_loss ~ N(0, 4)
        # This is very weak (σ=2 on log scale) and centered at rate=1
        baseline_prior = 0.0
        baseline_prior += -0.5 * (parameters['log_gain'] ** 2) / 4.0
        baseline_prior += -0.5 * (parameters['log_loss'] ** 2) / 4.0
        total_ll += baseline_prior
        
        return total_ll


@dataclass
class BaselineDiagnostics:
    """
    Diagnostic information about baseline model.
    
    Helps users catch nonsense estimates before proceeding to constraint testing.
    """
    gain_rate: float
    loss_rate: float
    equilibrium_presence: float
    mean_transitions_per_branch: float
    log_likelihood: float
    n_families: int
    n_tips: int
    
    def print_report(self, file=None):
        """Print diagnostic report."""
        if file is None:
            file = sys.stdout
        
        print("\nBaseline diagnostics:", file=file)
        print("  Gain rate: {0:.4f}".format(self.gain_rate), file=file)
        print("  Loss rate: {0:.4f}".format(self.loss_rate), file=file)
        print("  Equilibrium presence: {0:.4f}".format(self.equilibrium_presence), file=file)
        print("  Mean transitions per branch: {0:.2f}".format(self.mean_transitions_per_branch), file=file)
        print("  Log-likelihood: {0:.2f}".format(self.log_likelihood), file=file)
        print("  Data: {0} families, {1} tips".format(self.n_families, self.n_tips), file=file)
        print(file=file)
        
        # Data sufficiency diagnostic
        self._print_sufficiency_warning(file=file)
    
    def _print_sufficiency_warning(self, file=None):
        """Print data sufficiency warning if needed."""
        if file is None:
            file = sys.stdout
        
        # Estimate total transitions across tree
        total_rate = self.gain_rate + self.loss_rate
        estimated_transitions = self.n_families * total_rate * self.mean_transitions_per_branch
        
        print("Data sufficiency check:", file=file)
        print("  Families: {0}".format(self.n_families), file=file)
        print("  Tips: {0}".format(self.n_tips), file=file)
        print("  Estimated transitions: ~{0:.0f}".format(estimated_transitions), file=file)
        
        # Warn if data is insufficient (no hard stop, just honesty)
        if self.n_families < 50 or self.n_tips < 6:
            print("  ⚠ Warning: High variance regime – expect wide confidence intervals", file=file)
            print("  ⚠ Recommended: 100+ families, 8+ tips for robust inference", file=file)
        elif self.n_families < 100 or self.n_tips < 8:
            print("  ⚠ Note: Moderate power – results should be interpreted cautiously", file=file)
            print("  ⚠ Recommended: 100+ families, 8+ tips for robust inference", file=file)
        else:
            print("  ✓ Data size is adequate for reliable inference", file=file)
        
        print(file=file)
        
        # Baseline-sensitivity diagnostic (warning only, no correction)
        self._print_baseline_sensitivity_warning(file=file)
    
    def _print_baseline_sensitivity_warning(self, file=None):
        """Print baseline-sensitivity warning."""
        if file is None:
            file = sys.stdout
        
        print("Baseline-sensitivity check:", file=file)
        
        # Check for extreme rates (potential misspecification)
        if self.gain_rate < 0.01 or self.loss_rate < 0.01:
            print("  ⚠ Warning: Very low baseline rates detected", file=file)
            print("  ⚠ This may indicate model misspecification or insufficient data", file=file)
            print("  ⚠ Constraint tests may be unreliable", file=file)
        elif self.gain_rate > 100 or self.loss_rate > 100:
            print("  ⚠ Warning: Very high baseline rates detected", file=file)
            print("  ⚠ This may indicate model misspecification or data quality issues", file=file)
            print("  ⚠ Constraint tests may be unreliable", file=file)
        else:
            # Check for imbalanced rates
            rate_ratio = max(self.gain_rate, self.loss_rate) / max(min(self.gain_rate, self.loss_rate), 1e-10)
            if rate_ratio > 100:
                print("  ⚠ Note: Highly imbalanced gain/loss rates (ratio: {0:.1f})".format(rate_ratio), file=file)
                print("  ⚠ Results may be sensitive to baseline specification", file=file)
            else:
                print("  ✓ Baseline rates are in reasonable range", file=file)
        
        print(file=file)


@dataclass
class ComparisonResult:
    """
    Result of comparing alternative model to null.
    
    Provides interpretation guidance following HyPhy conventions.
    """
    null_result: 'MLEResult'
    alt_result: 'MLEResult'
    lrt_result: 'LRTResult'
    delta_ll: float
    delta_aic: float
    evidence_strength: str  # 'none', 'weak', 'moderate', 'strong'
    
    def print_report(self, file=None):
        """Print comparison report with interpretation guidance."""
        if file is None:
            file = sys.stdout
        
        print("\nNull vs alternative:", file=file)
        print("  ΔLL = {0:.2f}".format(self.delta_ll), file=file)
        
        # Emphasize ΔLL interpretation (HyPhy-style)
        if self.delta_ll < 2.0:
            print("  → Insufficient evidence", file=file)
        elif self.delta_ll < 5.0:
            print("  → Weak/exploratory evidence", file=file)
        elif self.delta_ll < 10.0:
            print("  → Moderate evidence", file=file)
        else:
            print("  → Strong evidence", file=file)
        
        print(file=file)
        
        # Show constraint parameter but de-emphasize it
        constraint_params = {k: v for k, v in self.alt_result.parameters.items() 
                           if k not in ['log_gain', 'log_loss']}
        if constraint_params:
            print("Constraint parameters (do not interpret alone):", file=file)
            for param_name, param_value in constraint_params.items():
                print("  {0} = {1:.4f}".format(param_name, param_value), file=file)
            print(file=file)
        
        print("Model comparison details:", file=file)
        print("  Null LL:  {0:.2f}".format(self.null_result.log_likelihood), file=file)
        print("  Alt LL:   {0:.2f}".format(self.alt_result.log_likelihood), file=file)
        print("  ΔAIC:     {0:.2f}".format(self.delta_aic), file=file)
        print("  p-value:  {0:.4f}".format(self.lrt_result.pvalue), file=file)
        print(file=file)
        
        print("Interpretation guidance:", file=file)
        print("  Evidence strength: {0}".format(self.evidence_strength.upper()), file=file)
        
        if self.evidence_strength == 'none':
            print("  → No evidence for constraint effect", file=file)
            print("  → Null model preferred", file=file)
        elif self.evidence_strength == 'weak':
            print("  → Weak/exploratory evidence", file=file)
            print("  → Interpret with caution", file=file)
        elif self.evidence_strength == 'moderate':
            print("  → Moderate evidence for constraint", file=file)
            print("  → Consider biological plausibility", file=file)
        else:  # strong
            print("  → Strong evidence for constraint", file=file)
            print("  → Effect likely real", file=file)
        
        print(file=file)
        print("Guidance thresholds (following HyPhy conventions):", file=file)
        print("  ΔLL < 2:    no evidence", file=file)
        print("  ΔLL 2-5:    weak/exploratory", file=file)
        print("  ΔLL 5-10:   moderate", file=file)
        print("  ΔLL > 10:   strong evidence", file=file)
        print(file=file)


class GeneContentInference:
    """
    High-level inference interface for gene content models.
    
    Provides convenient methods for:
    1. Fitting null and alternative models
    2. Likelihood ratio testing
    3. Model comparison with interpretation guidance
    4. Baseline diagnostics
    
    Usage:
        inference = GeneContentInference(data)
        
        # Get baseline diagnostics first
        diagnostics = inference.get_baseline_diagnostics()
        diagnostics.print_report()
        
        # Compare to null (recommended)
        result = inference.compare_to_null(
            RetentionBiasConstraint(retained_families={'OG0001'})
        )
        result.print_report()
    """
    
    def __init__(
        self,
        data: GeneContentData,
        use_jax: bool = False,
        use_rust: bool = True,
    ):
        """
        Initialize inference engine.
        
        Args:
            data: GeneContentData with tree and presence matrix
            use_jax: Whether to use JAX acceleration (for NumPy backend)
            use_rust: Whether to use Rust parallelization (default: True, auto-fallback)
        """
        self.data = data
        self.use_jax = use_jax
        self.use_rust = use_rust
        self._baseline_diagnostics = None
    
    def get_baseline_diagnostics(self, verbose: bool = True) -> BaselineDiagnostics:
        """
        Get baseline model diagnostics.
        
        This should be run BEFORE fitting constraints to catch nonsense estimates.
        
        Args:
            verbose: Whether to print diagnostic report
            
        Returns:
            BaselineDiagnostics with rate estimates and quality metrics
        """
        if self._baseline_diagnostics is None:
            # Fit null model
            null_result = self.fit_null()
            
            # Extract rates
            gain_rate = np.exp(null_result.parameters['log_gain'])
            loss_rate = np.exp(null_result.parameters['log_loss'])
            
            # Compute diagnostics
            total_rate = gain_rate + loss_rate
            equilibrium_presence = gain_rate / total_rate if total_rate > 0 else 0.5
            
            # Mean branch length
            mean_branch_length = self.data.tree.branch_lengths[self.data.tree.branch_lengths > 0].mean()
            mean_transitions = total_rate * mean_branch_length
            
            self._baseline_diagnostics = BaselineDiagnostics(
                gain_rate=gain_rate,
                loss_rate=loss_rate,
                equilibrium_presence=equilibrium_presence,
                mean_transitions_per_branch=mean_transitions,
                log_likelihood=null_result.log_likelihood,
                n_families=self.data.n_families,
                n_tips=len(self.data.taxon_names),
            )
        
        if verbose:
            self._baseline_diagnostics.print_report()
        
        return self._baseline_diagnostics
    
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
            use_rust=self.use_rust,
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
            use_rust=self.use_rust,
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
    
    def compare_to_null(
        self,
        constraint: GeneContentConstraint,
        alpha: float = 0.05,
        verbose: bool = True,
    ) -> ComparisonResult:
        """
        Compare constraint model to null with interpretation guidance.
        
        This is the RECOMMENDED way to test constraints. It:
        1. Fits both null and alternative models
        2. Performs likelihood ratio test
        3. Provides interpretation guidance based on ΔLL
        4. Prevents over-interpretation of θ̂
        
        Args:
            constraint: GeneContentConstraint for alternative model
            alpha: Significance level
            verbose: Whether to print comparison report
            
        Returns:
            ComparisonResult with interpretation guidance
        """
        # Fit both models
        null_result, alt_result, lrt_result = self.fit_and_test(constraint, alpha=alpha)
        
        # Compute deltas
        delta_ll = alt_result.log_likelihood - null_result.log_likelihood
        delta_aic = null_result.aic - alt_result.aic  # Positive favors alternative
        
        # Classify evidence strength (HyPhy-style thresholds)
        if delta_ll < 2.0:
            evidence_strength = 'none'
        elif delta_ll < 5.0:
            evidence_strength = 'weak'
        elif delta_ll < 10.0:
            evidence_strength = 'moderate'
        else:
            evidence_strength = 'strong'
        
        result = ComparisonResult(
            null_result=null_result,
            alt_result=alt_result,
            lrt_result=lrt_result,
            delta_ll=delta_ll,
            delta_aic=delta_aic,
            evidence_strength=evidence_strength,
        )
        
        if verbose:
            result.print_report()
        
        return result
