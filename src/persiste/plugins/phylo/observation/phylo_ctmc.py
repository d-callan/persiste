"""Phylogenetic CTMC observation model for PERSISTE with JAX acceleration."""

from typing import Optional, Dict, Any
import numpy as np

from persiste.core.observation_models import ObservationModel
from persiste.core.data import ObservedTransitions
from persiste.core.trees import TreeStructure
from persiste.core.pruning import FelsensteinPruning, ArrayTipConditionalProvider


class PhyloCTMCObservationModel(ObservationModel):
    """
    Phylogenetic continuous-time Markov chain observation model with JAX acceleration.
    
    Uses JAX JIT-compiled Felsenstein pruning for fast likelihood computation.
    Integrates with PERSISTE's ObservationModel interface.
    
    Key features:
    - JAX-accelerated pruning algorithm
    - Eigendecomposition caching for fast matrix exponential
    - Site-indexed likelihood computation (avoids per-site object creation)
    
    Attributes:
        tree: TreeStructure with branch lengths
        alignment: (n_taxa, n_sites) array of state indices
        site_weights: Optional weights for site patterns
    """
    
    def __init__(
        self,
        graph: Any,
        tree: TreeStructure,
        alignment: np.ndarray,
        site_weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize phylogenetic observation model with JAX-accelerated pruning.
        
        Args:
            graph: TransitionGraph (required by ObservationModel interface)
            tree: TreeStructure with branch lengths (from core.trees)
            alignment: (n_taxa, n_sites) array of state indices
            site_weights: Optional (n_sites,) array of weights for site patterns
        """
        self.graph = graph
        self.tree = tree
        self.alignment = alignment
        self.site_weights = site_weights if site_weights is not None else np.ones(alignment.shape[1])
        
        # Validate alignment
        if alignment.shape[0] != tree.n_tips:
            raise ValueError(
                f"Alignment has {alignment.shape[0]} sequences, "
                f"but tree has {tree.n_tips} taxa"
            )
        
        # Core pruning will be initialized on first likelihood call
        self._pruning = None
        self.n_states = None
    
    def rate(self, i: int, j: int) -> float:
        """
        Get baseline rate for transition i -> j.
        
        For phylogenetic models, rates are computed from the full rate matrix
        via the Baseline object. This method is required by ObservationModel
        but not directly used in phylogenetic likelihood computation.
        
        Args:
            i: Source state index
            j: Target state index
            
        Returns:
            0.0 (phylogenetic rates computed via rate matrix, not per-transition)
        """
        return 0.0
    
    def site_log_likelihood_with_alpha_beta(
        self,
        site_idx: int,
        alpha: float,
        beta: float,
        baseline: Any,
    ) -> float:
        """
        Compute log-likelihood for a single site with α (dS) and β (dN).
        
        This is the main method used by FEL analysis. Uses JAX-accelerated pruning
        with eigendecomposition caching for fast matrix exponential.
        
        Args:
            site_idx: Site index (0-based)
            alpha: Synonymous rate multiplier (dS)
            beta: Nonsynonymous rate multiplier (dN)
            baseline: Baseline object (e.g., MG94Baseline)
            
        Returns:
            Log-likelihood for the specified site
        """
        # Build rate matrix
        if hasattr(baseline, 'build_rate_matrix_alpha_beta'):
            Q = baseline.build_rate_matrix_alpha_beta(alpha=alpha, beta=beta)
        else:
            raise ValueError("Baseline must have build_rate_matrix_alpha_beta method")
        
        # Get frequencies
        if hasattr(baseline, 'codon_space'):
            freqs = baseline.codon_space.frequencies
        else:
            freqs = np.ones(self.n_states) / self.n_states
        
        # Initialize core pruning if needed
        if self._pruning is None:
            if self.n_states is None:
                self.n_states = Q.shape[0]
            # Use core FelsensteinPruning with JAX acceleration
            self._pruning = FelsensteinPruning(
                self.tree, 
                self.n_states, 
                use_jax=True,
                cache_transitions=True,
            )
        
        # Create tip provider for this site
        site_data = self.alignment[:, site_idx:site_idx+1]
        tip_provider = ArrayTipConditionalProvider(
            data=site_data,
            taxon_names=self.tree.tip_names,
            n_states=self.n_states,
        )
        
        # Precompute transition matrices using fast eigendecomposition method
        if hasattr(baseline, 'matrix_exponential_fast'):
            # Use cached eigendecomposition for fast P(t) computation
            precomputed_matrices = {}
            for node_idx in range(self.tree.n_nodes):
                t = self.tree.branch_lengths[node_idx]
                if t > 0:
                    P = baseline.matrix_exponential_fast(alpha, beta, t)
                    precomputed_matrices[node_idx] = P
                else:
                    precomputed_matrices[node_idx] = np.eye(self.n_states)
            
            # Compute likelihood with precomputed matrices (optimization)
            result = self._pruning.compute_likelihood_with_precomputed(
                precomputed_matrices=precomputed_matrices,
                tip_provider=tip_provider,
                equilibrium_freqs=freqs,
                n_sites=1,
            )
            log_lik = result.log_likelihood
        else:
            # Fall back to standard pruning (computes expm(Q*t) internally)
            from persiste.core.pruning import SimpleBinaryTransitionProvider
            
            # Create simple transition provider from Q
            # Note: This is less efficient than eigendecomposition caching
            class PhyloTransitionProvider:
                def __init__(self, Q_matrix, freqs):
                    self.Q = Q_matrix
                    self.equilibrium_frequencies = freqs
                
                def get_transition_matrix(self, t):
                    from scipy.linalg import expm
                    return expm(self.Q * t)
            
            transition_provider = PhyloTransitionProvider(Q, freqs)
            result = self._pruning.compute_likelihood(
                transition_provider=transition_provider,
                tip_provider=tip_provider,
                n_sites=1,
            )
            log_lik = result.log_likelihood
        
        return log_lik
    
    def log_likelihood_with_alpha_beta(
        self,
        alpha: float,
        beta: float,
        baseline: Any,
    ) -> float:
        """
        Compute total log-likelihood across all sites for fixed α and β.
        
        This preserves the legacy PhyloCTMCObservationModel API that FEL relied on.
        """
        total = 0.0
        for site_idx in range(self.n_sites):
            site_ll = self.site_log_likelihood_with_alpha_beta(
                site_idx,
                alpha,
                beta,
                baseline,
            )
            total += site_ll * self.site_weights[site_idx]
        return float(total)
    
    @property
    def n_sites(self) -> int:
        """Number of sites in alignment."""
        return self.alignment.shape[1]
    
    @property
    def n_taxa(self) -> int:
        """Number of taxa in tree."""
        return self.alignment.shape[0]
    
    def get_site_weights(self) -> np.ndarray:
        """Get site pattern weights."""
        return self.site_weights
    
    def log_likelihood_with_omega(
        self,
        omega: float,
        baseline: Any,
    ) -> float:
        """
        Compute total log-likelihood with specified ω (dN/dS).
        
        Args:
            omega: dN/dS ratio
            baseline: Baseline object (e.g., MG94Baseline)
            
        Returns:
            Total log-likelihood across all sites
        """
        site_lls = self.site_log_likelihoods_with_omega(omega, baseline)
        return float(np.sum(site_lls * self.site_weights))
    
    def site_log_likelihoods_with_omega(
        self,
        omega: float,
        baseline: Any,
    ) -> np.ndarray:
        """
        Compute per-site log-likelihoods with specified ω (dN/dS).
        
        Args:
            omega: dN/dS ratio
            baseline: Baseline object (e.g., MG94Baseline)
            
        Returns:
            Array of per-site log-likelihoods
        """
        site_lls = np.zeros(self.n_sites)
        for site_idx in range(self.n_sites):
            site_lls[site_idx] = self.site_log_likelihood_with_alpha_beta(
                site_idx, alpha=1.0, beta=omega, baseline=baseline
            )
        return site_lls
    
    def log_likelihood(
        self,
        data: ObservedTransitions,
        baseline: Any,
        graph: Any,
    ) -> float:
        """
        Compute log-likelihood via phylogenetic pruning.
        
        For phylogenetic models, the data parameter is unused since
        the alignment is stored in the observation model itself.
        
        Args:
            data: ObservedTransitions (unused for phylo models)
            baseline: Baseline object (e.g., MG94Baseline)
            graph: TransitionGraph (unused for phylo models)
            
        Returns:
            Total log-likelihood across all sites
        """
        # For phylo models, use specialized method with ω=1.0 as default
        # This provides compatibility with the ObservationModel interface
        # while maintaining phylo-specific functionality
        if hasattr(self, 'site_log_likelihoods_with_omega'):
            site_lls = self.site_log_likelihoods_with_omega(1.0, baseline)
            return float(np.sum(site_lls * self.site_weights))
        else:
            # Fallback: compute site-by-site
            total_ll = 0.0
            for site_idx in range(self.n_sites):
                site_ll = self.site_log_likelihood_with_alpha_beta(
                    site_idx, alpha=1.0, beta=1.0, baseline=baseline
                )
                total_ll += site_ll * self.site_weights[site_idx]
            return total_ll
