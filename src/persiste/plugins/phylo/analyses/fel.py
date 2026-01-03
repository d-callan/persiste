"""
Fixed Effects Likelihood (FEL) analysis.

FEL tests for site-specific selection by fitting different ω (dN/dS) values
at each codon site independently.

Reference:
    Kosakovsky Pond & Frost (2005) "Not So Different After All: 
    A Comparison of Methods for Detecting Amino Acid Sites Under Selection"
    Molecular Biology and Evolution 22(5):1208-1222
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from dataclasses import dataclass

from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
from persiste.plugins.phylo.data.site_patterns import SitePatterns


@dataclass
class FELSiteResult:
    """
    Results for a single site from FEL analysis.
    
    Attributes:
        site: Site index (0-based)
        alpha: Synonymous substitution rate (dS)
        beta: Nonsynonymous substitution rate (dN)
        omega: dN/dS ratio (β/α)
        log_likelihood: Log-likelihood at MLE
        lrt_statistic: Likelihood ratio test statistic (2Δℓ)
        p_value: P-value from chi-squared test
        significant: Whether site shows significant selection (p ≤ 0.1)
    """
    site: int
    alpha: float  # dS
    beta: float   # dN
    omega: float  # dN/dS
    log_likelihood: float
    lrt_statistic: float
    p_value: float
    significant: bool
    
    def __repr__(self) -> str:
        sig = "***" if self.significant else ""
        return (
            f"Site {self.site}: ω={self.omega:.4f} "
            f"(α={self.alpha:.4f}, β={self.beta:.4f}), "
            f"p={self.p_value:.4f} {sig}"
        )


class FELAnalysis:
    """
    Fixed Effects Likelihood (FEL) analysis for site-specific selection.
    
    FEL fits separate α (synonymous rate, dS) and β (nonsynonymous rate, dN)
    parameters at each codon site independently, matching HyPhy's approach.
    
    Analysis at each site:
    1. Fit MLE for α and β separately (alternative model)
    2. Fit MLE with α = β constraint (null model, neutral evolution)
    3. Test H₀: α = β (neutral) vs H₁: α ≠ β (selection) via LRT
    4. Report sites with significant evidence of selection
    
    Key features:
    - Site-independent: each site gets its own α and β
    - 2-parameter optimization at each site (matches HyPhy FEL)
    - Uses full phylogenetic likelihood (Felsenstein pruning)
    - LRT with 1 degree of freedom (α = β vs α ≠ β)
    - Standard threshold: p ≤ 0.1 (following HyPhy convention)
    
    Selection interpretation:
    - β < α (ω < 1): Purifying (negative) selection
    - β > α (ω > 1): Positive (diversifying) selection
    - β ≈ α (ω ≈ 1): Neutral evolution
    
    Attributes:
        obs_model: PhyloCTMCObservationModel with tree and alignment
        baseline: MG94Baseline for codon model
        p_threshold: P-value threshold for significance (default: 0.1)
    """
    
    def __init__(
        self,
        obs_model: PhyloCTMCObservationModel,
        baseline: MG94Baseline,
        p_threshold: float = 0.1,
        use_site_patterns: bool = True,
    ):
        """
        Initialize FEL analysis.
        
        Args:
            obs_model: PhyloCTMCObservationModel with tree and alignment (uses JAX)
            baseline: MG94Baseline for codon substitution model
            p_threshold: P-value threshold for significance (default: 0.1)
            use_site_patterns: Whether to use site patterns compression (default: True)
        """
        self.obs_model = obs_model
        self.baseline = baseline
        self.p_threshold = p_threshold
        self.use_site_patterns = use_site_patterns
        
        if self.use_site_patterns:
            self.site_patterns = SitePatterns(obs_model.alignment)
        else:
            self.site_patterns = None
        
        # Results storage
        self.site_results: List[FELSiteResult] = []
    
    def fit_site_alpha_beta(
        self,
        site_idx: int,
        rate_bounds: Tuple[float, float] = (0.0, 100.0),
    ) -> Tuple[float, float, float]:
        """
        Fit MLE for α (dS) and β (dN) at a single site.
        
        This matches HyPhy's FEL parameterization where both synonymous
        and nonsynonymous rates are free parameters.
        
        Args:
            site_idx: Site index (0-based)
            rate_bounds: (min, max) bounds for rate optimization
            
        Returns:
            (alpha_mle, beta_mle, log_likelihood) tuple
        """
        # Extract single-site alignment
        site_alignment = self.obs_model.alignment[:, site_idx:site_idx+1]
        
        # Create single-site observation model
        site_obs_model = PhyloCTMCObservationModel(
            graph=self.obs_model.graph,
            tree=self.obs_model.tree,
            alignment=site_alignment,
        )
        
        # Objective: negative log-likelihood as function of (α, β)
        def nll(params):
            alpha, beta = params
            # Ensure positive rates
            if alpha <= 0 or beta <= 0:
                return 1e10
            return -site_obs_model.log_likelihood_with_alpha_beta(alpha, beta, self.baseline)
        
        # Initial guess: α = β = 1 (neutral)
        x0 = [1.0, 1.0]
        
        # Bounds: both rates in [min, max]
        bounds = [rate_bounds, rate_bounds]
        
        # Optimize using L-BFGS-B (supports bounds)
        result = minimize(
            nll,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
        )
        
        alpha_mle, beta_mle = result.x
        log_lik = -result.fun
        
        return alpha_mle, beta_mle, log_lik
    
    def fit_site_constrained(
        self,
        site_idx: int,
        rate_bounds: Tuple[float, float] = (0.0, 100.0),
    ) -> Tuple[float, float]:
        """
        Fit MLE under null hypothesis: α = β (neutral evolution).
        
        Args:
            site_idx: Site index (0-based)
            rate_bounds: (min, max) bounds for rate optimization
            
        Returns:
            (rate_mle, log_likelihood) tuple where α = β = rate_mle
        """
        # Objective: negative log-likelihood with α = β
        # Use site-indexed method to avoid creating new observation models
        def nll(rate):
            if rate <= 0:
                return 1e10
            return -self.obs_model.site_log_likelihood_with_alpha_beta(
                site_idx, rate, rate, self.baseline
            )
        
        # Optimize single rate parameter
        result = minimize_scalar(
            nll,
            bounds=rate_bounds,
            method='bounded',
        )
        
        rate_mle = result.x
        log_lik = -result.fun
        
        return rate_mle, log_lik
    
    def likelihood_ratio_test(
        self,
        site_idx: int,
        log_lik_alt: float,
    ) -> Tuple[float, float, float]:
        """
        Perform likelihood ratio test for H₀: α=β vs H₁: α≠β.
        
        This matches HyPhy's FEL test:
        - Null: neutral evolution (synonymous rate = nonsynonymous rate)
        - Alternative: selection (rates differ)
        
        Args:
            site_idx: Site index
            log_lik_alt: Log-likelihood under alternative (α ≠ β)
            
        Returns:
            (rate_null, lrt_statistic, p_value) tuple
        """
        # Null model: α = β (neutral)
        rate_null, log_lik_null = self.fit_site_constrained(site_idx)
        
        # LRT statistic: 2 * (ℓ_alt - ℓ_null)
        lrt_stat = 2 * (log_lik_alt - log_lik_null)
        
        # Ensure non-negative (numerical issues can cause small negatives)
        lrt_stat = max(0.0, lrt_stat)
        
        # P-value from chi-squared distribution (1 df)
        p_value = stats.chi2.sf(lrt_stat, df=1)
        
        return rate_null, lrt_stat, p_value
    
    def analyze_site(self, site_idx: int) -> FELSiteResult:
        """
        Perform FEL analysis for a single site.
        
        Fits separate α (dS) and β (dN) rates at the site,
        then tests H₀: α = β (neutral) vs H₁: α ≠ β (selection).
        
        Args:
            site_idx: Site index (0-based)
            
        Returns:
            FELSiteResult with α, β estimates and test statistics
        """
        # Fit α and β separately (alternative model)
        alpha_mle, beta_mle, log_lik = self.fit_site_alpha_beta(site_idx)
        
        # LRT for selection (α ≠ β)
        _, lrt_stat, p_value = self.likelihood_ratio_test(site_idx, log_lik)
        
        # Compute ω = β/α
        if alpha_mle > 0:
            omega = beta_mle / alpha_mle
        else:
            omega = float('inf') if beta_mle > 0 else 1.0
        
        # Determine significance
        significant = p_value <= self.p_threshold
        
        return FELSiteResult(
            site=site_idx,
            alpha=alpha_mle,
            beta=beta_mle,
            omega=omega,
            log_likelihood=log_lik,
            lrt_statistic=lrt_stat,
            p_value=p_value,
            significant=significant,
        )
    
    def run(self) -> List[FELSiteResult]:
        """
        Run FEL analysis on all sites using JAX-accelerated likelihood computation.
        
        Returns:
            List of FELSiteResult objects, one per site
        """
        n_sites = self.obs_model.n_sites
        
        self.site_results = []
        for site_idx in range(n_sites):
            result = self.analyze_site(site_idx)
            self.site_results.append(result)
        
        return self.site_results
    
    def get_significant_sites(self) -> List[FELSiteResult]:
        """
        Get sites with significant evidence of selection.
        
        Returns:
            List of FELSiteResult for significant sites
        """
        return [r for r in self.site_results if r.significant]
    
    def get_positively_selected_sites(self) -> List[FELSiteResult]:
        """
        Get sites under positive selection (ω > 1 and significant).
        
        Returns:
            List of FELSiteResult for positively selected sites
        """
        return [r for r in self.site_results if r.significant and r.omega > 1.0]
    
    def get_negatively_selected_sites(self) -> List[FELSiteResult]:
        """
        Get sites under purifying selection (ω < 1 and significant).
        
        Returns:
            List of FELSiteResult for purifying selection sites
        """
        return [r for r in self.site_results if r.significant and r.omega < 1.0]
    
    def summary(self) -> Dict:
        """
        Generate summary statistics for FEL analysis.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.site_results:
            return {"error": "No results available. Run analysis first."}
        
        n_sites = len(self.site_results)
        n_significant = len(self.get_significant_sites())
        n_positive = len(self.get_positively_selected_sites())
        n_negative = len(self.get_negatively_selected_sites())
        
        omega_values = [r.omega for r in self.site_results]
        
        return {
            "n_sites": n_sites,
            "n_significant": n_significant,
            "n_positive_selection": n_positive,
            "n_purifying_selection": n_negative,
            "mean_omega": np.mean(omega_values),
            "median_omega": np.median(omega_values),
            "min_omega": np.min(omega_values),
            "max_omega": np.max(omega_values),
            "p_threshold": self.p_threshold,
        }
    
    def to_hyphy_json(self) -> Dict:
        """
        Export results in HyPhy-compatible JSON format.
        
        Returns:
            Dictionary matching HyPhy FEL output structure
        """
        return {
            "analysis": "FEL (Fixed Effects Likelihood)",
            "input": {
                "n_sequences": self.obs_model.n_taxa,
                "n_sites": self.obs_model.n_sites,
            },
            "test_results": {
                "p_value_threshold": self.p_threshold,
            },
            "MLE": {
                "content": {
                    str(r.site): {
                        "alpha": r.alpha,
                        "beta": r.beta,
                        "omega": r.omega,
                        "LRT": r.lrt_statistic,
                        "p": r.p_value,
                        "significant": r.significant,
                    }
                    for r in self.site_results
                }
            },
            "summary": self.summary(),
        }
    
    def __repr__(self) -> str:
        if not self.site_results:
            return "FELAnalysis(not run)"
        
        summary = self.summary()
        return (
            f"FELAnalysis("
            f"sites={summary['n_sites']}, "
            f"significant={summary['n_significant']}, "
            f"positive={summary['n_positive_selection']}, "
            f"purifying={summary['n_purifying_selection']})"
        )
