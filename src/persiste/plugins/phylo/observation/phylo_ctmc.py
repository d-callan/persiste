"""Phylogenetic CTMC observation model for PERSISTE with JAX acceleration."""

from typing import Optional, Dict, Any
import numpy as np

from persiste.core.observation_models import ObservationModel
from persiste.core.data import ObservedTransitions
from persiste.plugins.phylo.data.tree import PhylogeneticTree

try:
    from persiste.plugins.phylo.observation.pruning_jax import JAXFelsensteinPruning, JAX_AVAILABLE
except ImportError:
    JAX_AVAILABLE = False
    JAXFelsensteinPruning = None


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
        tree: PhylogeneticTree with branch lengths
        alignment: (n_taxa, n_sites) array of state indices
        site_weights: Optional weights for site patterns
    """
    
    def __init__(
        self,
        graph: Any,
        tree: PhylogeneticTree,
        alignment: np.ndarray,
        site_weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize phylogenetic observation model with JAX-accelerated pruning.
        
        Args:
            graph: TransitionGraph (required by ObservationModel interface)
            tree: PhylogeneticTree with branch lengths
            alignment: (n_taxa, n_sites) array of state indices
            site_weights: Optional (n_sites,) array of weights for site patterns
            
        Raises:
            ImportError: If JAX is not installed
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is required for PhyloCTMCObservationModel. "
                "Install with: conda install -c conda-forge jax jaxlib"
            )
        
        self.graph = graph
        self.tree = tree
        self.alignment = alignment
        self.site_weights = site_weights if site_weights is not None else np.ones(alignment.shape[1])
        
        # Validate alignment
        if alignment.shape[0] != tree.n_taxa:
            raise ValueError(
                f"Alignment has {alignment.shape[0]} sequences, "
                f"but tree has {tree.n_taxa} taxa"
            )
        
        # JAX pruning will be initialized on first likelihood call
        self._pruning_jax = None
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
        
        # Initialize JAX pruning if needed
        if self._pruning_jax is None:
            if self.n_states is None:
                self.n_states = Q.shape[0]
            self._pruning_jax = JAXFelsensteinPruning(self.tree.tree, self.n_states)
        
        # Precompute transition matrices using fast eigendecomposition method
        if hasattr(baseline, 'matrix_exponential_fast'):
            # Use cached eigendecomposition for fast P(t) computation
            transition_matrices = {}
            for i, node in enumerate(self._pruning_jax.nodes):
                t = self._pruning_jax.branch_lengths[i]
                if t > 0:
                    P = baseline.matrix_exponential_fast(alpha, beta, t)
                    transition_matrices[i] = P
                else:
                    transition_matrices[i] = np.eye(self.n_states)
            
            # Compute likelihood with precomputed matrices
            site_data = self.alignment[:, site_idx:site_idx+1]
            log_lik = self._pruning_jax.compute_likelihood_with_transitions(
                site_data, freqs, transition_matrices
            )
        else:
            # Fall back to computing expm(Q*t) in JAX
            site_data = self.alignment[:, site_idx:site_idx+1]
            log_lik = self._pruning_jax.compute_likelihood(site_data, Q, freqs)
        
        return log_lik
    
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
