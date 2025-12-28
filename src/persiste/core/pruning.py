"""
Core Felsenstein pruning algorithm for PERSISTE.

Provides a generic, backend-agnostic implementation of the pruning algorithm
for computing likelihoods on trees. Plugins provide:
1. Rate matrix Q (or transition probabilities P(t))
2. Tip conditionals (observation likelihoods)
3. Root prior (equilibrium frequencies)

The core handles:
1. Tree traversal
2. Conditional probability propagation
3. JAX acceleration (optional)

Key design principles:
1. Separation of concerns - core does traversal, plugins do rates
2. Backend flexibility - works with numpy, optionally accelerated with JAX
3. Efficient batching - process multiple sites/families simultaneously
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union
import numpy as np

from .trees import TreeStructure


# Check for JAX availability
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    from jax.scipy.linalg import expm as jax_expm
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None


class TransitionMatrixProvider(Protocol):
    """
    Protocol for objects that provide transition probability matrices.
    
    Plugins implement this to provide P(t) = exp(Qt) for their rate models.
    """
    
    def get_transition_matrix(self, branch_length: float) -> np.ndarray:
        """
        Compute transition probability matrix for given branch length.
        
        Args:
            branch_length: Time/distance along branch
            
        Returns:
            (n_states, n_states) transition probability matrix P(t)
        """
        ...
    
    @property
    def n_states(self) -> int:
        """Number of states in the model."""
        ...
    
    @property
    def equilibrium_frequencies(self) -> np.ndarray:
        """Equilibrium state frequencies (root prior)."""
        ...


class TipConditionalProvider(Protocol):
    """
    Protocol for objects that provide tip conditional likelihoods.
    
    Plugins implement this to map observed data to state likelihoods.
    """
    
    def get_tip_conditional(
        self,
        tip_name: str,
        site_idx: int,
    ) -> np.ndarray:
        """
        Get conditional likelihood vector at a tip.
        
        Args:
            tip_name: Name of the tip taxon
            site_idx: Site/family index
            
        Returns:
            (n_states,) array of P(observation | state)
        """
        ...


@dataclass
class PruningResult:
    """
    Result of pruning algorithm.
    
    Attributes:
        log_likelihood: Total log-likelihood
        site_log_likelihoods: Per-site log-likelihoods (if computed)
        root_conditionals: Conditional likelihoods at root (optional)
    """
    log_likelihood: float
    site_log_likelihoods: Optional[np.ndarray] = None
    root_conditionals: Optional[np.ndarray] = None


class FelsensteinPruning:
    """
    Generic Felsenstein pruning algorithm.
    
    This is the core likelihood computation engine for tree-based models.
    Works with any state space and rate model that implements the protocols.
    
    Optimization flags (WIP - exploring different strategies):
    - use_jax: JAX JIT compilation for speed
    - cache_transitions: Precompute transition matrices (useful when same Q for all sites)
    - use_eigen_cache: Use eigendecomposition caching for fast matrix exponentials
    
    Usage:
        # Setup
        tree = TreeStructure.from_newick(newick_str)
        pruning = FelsensteinPruning(tree, n_states=2, use_jax=True)
        
        # Compute likelihood
        result = pruning.compute_likelihood(
            transition_provider=my_rate_model,
            tip_provider=my_observation_model,
        )
    """
    
    def __init__(
        self,
        tree: TreeStructure,
        n_states: int,
        use_jax: bool = True,
        cache_transitions: bool = False,
    ):
        """
        Initialize pruning algorithm.
        
        Args:
            tree: TreeStructure with topology and branch lengths
            n_states: Number of states in the model
            use_jax: Whether to use JAX acceleration (if available)
            cache_transitions: Whether to cache precomputed transition matrices
                             (useful for phylo models with same Q across sites)
        """
        self.tree = tree
        self.n_states = n_states
        self.use_jax = use_jax and JAX_AVAILABLE
        self.cache_transitions = cache_transitions
        
        # Cache for precomputed transition matrices
        self._transition_cache = {} if cache_transitions else None
        
        # Precompute tree structure for efficient traversal
        self._setup_traversal()
        
        if self.use_jax:
            self._setup_jax()
    
    def _setup_traversal(self):
        """Precompute traversal order and structure."""
        # Map tip names to indices
        self.tip_name_to_idx = self.tree.get_tip_index_map()
        
        # Postorder for bottom-up traversal
        self.postorder = self.tree.postorder
        
        # Children array for internal nodes
        self.children_array = self.tree.children_array
    
    def _setup_jax(self):
        """Setup JAX-accelerated functions."""
        
        @jit
        def combine_children(
            child1_cond: jnp.ndarray,
            child2_cond: jnp.ndarray,
            P1: jnp.ndarray,
            P2: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            Combine conditionals from two children.
            
            L_parent[site, state] = 
                (Σ_j P1[state,j] * L_child1[site,j]) *
                (Σ_k P2[state,k] * L_child2[site,k])
            """
            # child_cond: (n_sites, n_states)
            # P: (n_states, n_states)
            term1 = jnp.dot(child1_cond, P1.T)  # (n_sites, n_states)
            term2 = jnp.dot(child2_cond, P2.T)
            return term1 * term2
        
        @jit
        def root_likelihood(
            root_cond: jnp.ndarray,
            freqs: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            Compute site likelihoods at root.
            
            L_site = Σ_states π_state * L_root[site, state]
            """
            return jnp.sum(root_cond * freqs[None, :], axis=1)
        
        self._jax_combine = combine_children
        self._jax_root_lik = root_likelihood
    
    def compute_likelihood(
        self,
        transition_provider: TransitionMatrixProvider,
        tip_provider: TipConditionalProvider,
        n_sites: int = 1,
        site_weights: Optional[np.ndarray] = None,
        return_per_site: bool = False,
    ) -> PruningResult:
        """
        Compute log-likelihood using Felsenstein pruning.
        
        Args:
            transition_provider: Provides P(t) matrices
            tip_provider: Provides tip conditional likelihoods
            n_sites: Number of sites/families to process
            site_weights: Optional weights for each site
            return_per_site: Whether to return per-site likelihoods
            
        Returns:
            PruningResult with log-likelihood
        """
        if self.use_jax:
            return self._compute_likelihood_jax(
                transition_provider, tip_provider, n_sites, 
                site_weights, return_per_site
            )
        else:
            return self._compute_likelihood_numpy(
                transition_provider, tip_provider, n_sites,
                site_weights, return_per_site
            )
    
    def compute_likelihood_with_precomputed(
        self,
        precomputed_matrices: Dict[int, np.ndarray],
        tip_provider: TipConditionalProvider,
        equilibrium_freqs: np.ndarray,
        n_sites: int = 1,
        site_weights: Optional[np.ndarray] = None,
        return_per_site: bool = False,
    ) -> PruningResult:
        """
        Compute likelihood with precomputed transition matrices.
        
        Optimization for phylo models where the same Q is used across sites.
        Precompute P(t) = exp(Qt) for each branch once, then reuse.
        
        Args:
            precomputed_matrices: Dict mapping node_idx -> P(t) matrix
            tip_provider: Provides tip conditional likelihoods
            equilibrium_freqs: Equilibrium state frequencies
            n_sites: Number of sites to process
            site_weights: Optional weights for each site
            return_per_site: Whether to return per-site likelihoods
            
        Returns:
            PruningResult with log-likelihood
        """
        n_nodes = self.tree.n_nodes
        
        # Initialize conditionals: (n_nodes, n_sites, n_states)
        if self.use_jax:
            conditionals = jnp.zeros((n_nodes, n_sites, self.n_states))
        else:
            conditionals = np.zeros((n_nodes, n_sites, self.n_states))
        
        # Initialize tip conditionals
        for tip_idx, tip_name in zip(self.tree.tip_indices, self.tree.tip_names):
            for site in range(n_sites):
                conditionals = conditionals.at[tip_idx, site, :].set(
                    tip_provider.get_tip_conditional(tip_name, site)
                ) if self.use_jax else conditionals
                if not self.use_jax:
                    conditionals[tip_idx, site, :] = tip_provider.get_tip_conditional(tip_name, site)
        
        # Postorder traversal with precomputed matrices
        for parent_idx, child1_idx, child2_idx in self.children_array:
            P1 = precomputed_matrices[child1_idx]
            P2 = precomputed_matrices[child2_idx]
            
            child1_cond = conditionals[child1_idx]
            child2_cond = conditionals[child2_idx]
            
            if self.use_jax:
                parent_cond = self._jax_combine(
                    jnp.array(child1_cond), jnp.array(child2_cond),
                    jnp.array(P1), jnp.array(P2)
                )
                conditionals = conditionals.at[parent_idx].set(parent_cond)
            else:
                term1 = child1_cond @ P1.T
                term2 = child2_cond @ P2.T
                conditionals[parent_idx] = term1 * term2
        
        # Compute likelihood at root
        root_cond = conditionals[self.tree.root_index]
        
        if self.use_jax:
            site_likelihoods = self._jax_root_lik(
                jnp.array(root_cond), jnp.array(equilibrium_freqs)
            )
            site_likelihoods = np.array(site_likelihoods)
        else:
            site_likelihoods = np.sum(root_cond * equilibrium_freqs[None, :], axis=1)
        
        # Apply weights and compute total log-likelihood
        site_log_liks = np.log(site_likelihoods + 1e-300)
        
        if site_weights is None:
            site_weights = np.ones(n_sites)
        
        total_log_lik = np.sum(site_log_liks * site_weights)
        
        return PruningResult(
            log_likelihood=float(total_log_lik),
            site_log_likelihoods=site_log_liks if return_per_site else None,
            root_conditionals=root_cond if return_per_site else None,
        )
    
    def _compute_likelihood_numpy(
        self,
        transition_provider: TransitionMatrixProvider,
        tip_provider: TipConditionalProvider,
        n_sites: int,
        site_weights: Optional[np.ndarray],
        return_per_site: bool,
    ) -> PruningResult:
        """NumPy implementation of pruning."""
        n_nodes = self.tree.n_nodes
        
        # Initialize conditionals: (n_nodes, n_sites, n_states)
        conditionals = np.zeros((n_nodes, n_sites, self.n_states))
        
        # Initialize tip conditionals
        for tip_idx, tip_name in zip(self.tree.tip_indices, self.tree.tip_names):
            for site in range(n_sites):
                conditionals[tip_idx, site, :] = tip_provider.get_tip_conditional(
                    tip_name, site
                )
        
        # Precompute transition matrices
        transition_matrices = {}
        for node_idx in range(n_nodes):
            t = self.tree.branch_lengths[node_idx]
            if t > 0:
                transition_matrices[node_idx] = transition_provider.get_transition_matrix(t)
            else:
                transition_matrices[node_idx] = np.eye(self.n_states)
        
        # Postorder traversal: combine children
        for parent_idx, child1_idx, child2_idx in self.children_array:
            P1 = transition_matrices[child1_idx]
            P2 = transition_matrices[child2_idx]
            
            child1_cond = conditionals[child1_idx]  # (n_sites, n_states)
            child2_cond = conditionals[child2_idx]
            
            # L_parent = (P1 @ child1) * (P2 @ child2)
            term1 = child1_cond @ P1.T  # (n_sites, n_states)
            term2 = child2_cond @ P2.T
            conditionals[parent_idx] = term1 * term2
        
        # Compute likelihood at root
        freqs = transition_provider.equilibrium_frequencies
        root_cond = conditionals[self.tree.root_index]  # (n_sites, n_states)
        
        site_likelihoods = np.sum(root_cond * freqs[None, :], axis=1)  # (n_sites,)
        
        # Apply weights and compute total log-likelihood
        site_log_liks = np.log(site_likelihoods + 1e-300)
        
        if site_weights is not None:
            total_log_lik = np.sum(site_log_liks * site_weights)
        else:
            total_log_lik = np.sum(site_log_liks)
        
        return PruningResult(
            log_likelihood=float(total_log_lik),
            site_log_likelihoods=site_log_liks if return_per_site else None,
            root_conditionals=root_cond if return_per_site else None,
        )
    
    def _compute_likelihood_jax(
        self,
        transition_provider: TransitionMatrixProvider,
        tip_provider: TipConditionalProvider,
        n_sites: int,
        site_weights: Optional[np.ndarray],
        return_per_site: bool,
    ) -> PruningResult:
        """JAX-accelerated implementation of pruning."""
        n_nodes = self.tree.n_nodes
        
        # Initialize conditionals: (n_nodes, n_sites, n_states)
        conditionals = np.zeros((n_nodes, n_sites, self.n_states))
        
        # Initialize tip conditionals
        for tip_idx, tip_name in zip(self.tree.tip_indices, self.tree.tip_names):
            for site in range(n_sites):
                conditionals[tip_idx, site, :] = tip_provider.get_tip_conditional(
                    tip_name, site
                )
        
        # Precompute transition matrices
        transition_matrices = {}
        for node_idx in range(n_nodes):
            t = self.tree.branch_lengths[node_idx]
            if t > 0:
                transition_matrices[node_idx] = transition_provider.get_transition_matrix(t)
            else:
                transition_matrices[node_idx] = np.eye(self.n_states)
        
        # Convert to JAX arrays
        conditionals_jax = jnp.array(conditionals)
        
        # Postorder traversal with JAX
        for parent_idx, child1_idx, child2_idx in self.children_array:
            P1 = jnp.array(transition_matrices[child1_idx])
            P2 = jnp.array(transition_matrices[child2_idx])
            
            child1_cond = conditionals_jax[child1_idx]
            child2_cond = conditionals_jax[child2_idx]
            
            parent_cond = self._jax_combine(child1_cond, child2_cond, P1, P2)
            conditionals_jax = conditionals_jax.at[parent_idx].set(parent_cond)
        
        # Compute likelihood at root
        freqs = jnp.array(transition_provider.equilibrium_frequencies)
        root_cond = conditionals_jax[self.tree.root_index]
        
        site_likelihoods = self._jax_root_lik(root_cond, freqs)
        site_log_liks = jnp.log(site_likelihoods + 1e-300)
        
        # Apply weights
        if site_weights is not None:
            weights_jax = jnp.array(site_weights)
            total_log_lik = jnp.sum(site_log_liks * weights_jax)
        else:
            total_log_lik = jnp.sum(site_log_liks)
        
        return PruningResult(
            log_likelihood=float(total_log_lik),
            site_log_likelihoods=np.array(site_log_liks) if return_per_site else None,
            root_conditionals=np.array(root_cond) if return_per_site else None,
        )


class SimpleBinaryTransitionProvider:
    """
    Simple transition provider for binary (0/1) state models.
    
    This is useful for gene content (presence/absence) models.
    
    Attributes:
        gain_rate: Rate of 0 → 1 transition
        loss_rate: Rate of 1 → 0 transition
    """
    
    def __init__(self, gain_rate: float, loss_rate: float):
        """
        Initialize binary transition provider.
        
        Args:
            gain_rate: Rate of gaining (0 → 1)
            loss_rate: Rate of losing (1 → 0)
        """
        self.gain_rate = gain_rate
        self.loss_rate = loss_rate
        self._n_states = 2
    
    @property
    def n_states(self) -> int:
        return self._n_states
    
    @property
    def equilibrium_frequencies(self) -> np.ndarray:
        """Equilibrium frequencies from detailed balance."""
        total = self.gain_rate + self.loss_rate
        if total < 1e-10:
            return np.array([0.5, 0.5])
        
        # π_0 * λ_gain = π_1 * λ_loss (detailed balance)
        # π_0 + π_1 = 1
        pi_1 = self.gain_rate / total
        pi_0 = self.loss_rate / total
        return np.array([pi_0, pi_1])
    
    def get_transition_matrix(self, branch_length: float) -> np.ndarray:
        """
        Compute P(t) for binary model.
        
        Closed-form solution for 2-state CTMC.
        """
        t = branch_length
        λ = self.gain_rate
        μ = self.loss_rate
        total = λ + μ
        
        if total < 1e-10:
            return np.eye(2)
        
        exp_term = np.exp(-total * t)
        
        p00 = (μ + λ * exp_term) / total
        p01 = (λ - λ * exp_term) / total
        p10 = (μ - μ * exp_term) / total
        p11 = (λ + μ * exp_term) / total
        
        return np.array([[p00, p01], [p10, p11]])


class ArrayTipConditionalProvider:
    """
    Tip conditional provider backed by a numpy array.
    
    For gene content: array[taxon_idx, family_idx] = 0 or 1
    """
    
    def __init__(
        self,
        data: np.ndarray,
        taxon_names: List[str],
        n_states: int = 2,
    ):
        """
        Initialize from data array.
        
        Args:
            data: (n_taxa, n_sites) array of observed states
            taxon_names: List of taxon names matching row order
            n_states: Number of states
        """
        self.data = data
        self.taxon_names = taxon_names
        self.taxon_to_idx = {name: i for i, name in enumerate(taxon_names)}
        self.n_states = n_states
    
    def get_tip_conditional(self, tip_name: str, site_idx: int) -> np.ndarray:
        """
        Get conditional likelihood at tip.
        
        For observed state s: P(obs | state) = 1 if state == s, else 0
        """
        taxon_idx = self.taxon_to_idx.get(tip_name)
        
        if taxon_idx is None:
            # Unknown taxon - all states equally likely
            return np.ones(self.n_states)
        
        observed_state = int(self.data[taxon_idx, site_idx])
        
        if observed_state < 0 or observed_state >= self.n_states:
            # Missing data - all states equally likely
            return np.ones(self.n_states)
        
        # Deterministic observation
        cond = np.zeros(self.n_states)
        cond[observed_state] = 1.0
        return cond
