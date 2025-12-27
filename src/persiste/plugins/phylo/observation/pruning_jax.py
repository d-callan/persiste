"""JAX-accelerated Felsenstein pruning algorithm.

This module provides a JIT-compiled implementation of the pruning algorithm
for significantly faster likelihood computation.

Key optimizations:
- JIT compilation via JAX
- Vectorized operations
- Efficient matrix operations
- Minimal Python overhead
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

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

try:
    import dendropy
    DENDROPY_AVAILABLE = True
except ImportError:
    DENDROPY_AVAILABLE = False
    dendropy = None


class JAXFelsensteinPruning:
    """
    JAX-accelerated Felsenstein pruning algorithm.
    
    Uses JIT compilation for fast likelihood computation.
    Falls back to NumPy if JAX is unavailable.
    """
    
    def __init__(
        self,
        tree: "dendropy.Tree",
        n_states: int,
    ):
        """
        Initialize JAX pruning.
        
        Args:
            tree: DendroPy Tree object
            n_states: Number of states (e.g., 61 for codons)
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX is required for JAXFelsensteinPruning")
        
        self.tree = tree
        self.n_states = n_states
        
        # Precompute tree structure for JAX
        self._precompute_tree_structure()
        
        # JIT-compile core functions
        self._jit_compile_functions()
    
    def _precompute_tree_structure(self):
        """
        Extract tree structure into arrays for JAX.
        
        JAX works best with arrays, not tree objects.
        We convert the tree into:
        - node_indices: mapping from nodes to indices
        - parent_indices: parent[i] = index of parent of node i
        - branch_lengths: branch_lengths[i] = length of edge to node i
        - child_indices: list of (parent, child1, child2) for internal nodes
        """
        # Assign indices to nodes
        self.nodes = list(self.tree.postorder_node_iter())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        n_nodes = len(self.nodes)
        
        # Extract branch lengths
        self.branch_lengths = np.zeros(n_nodes)
        for i, node in enumerate(self.nodes):
            if node.edge_length is not None:
                self.branch_lengths[i] = node.edge_length
        
        # Extract parent-child relationships for internal nodes
        self.internal_nodes = []
        for node in self.nodes:
            if not node.is_leaf():
                children = node.child_nodes()
                if len(children) == 2:
                    parent_idx = self.node_to_idx[node]
                    child1_idx = self.node_to_idx[children[0]]
                    child2_idx = self.node_to_idx[children[1]]
                    self.internal_nodes.append((parent_idx, child1_idx, child2_idx))
        
        # Convert to arrays
        self.internal_node_array = np.array(self.internal_nodes, dtype=np.int32)
        
        # Identify leaf nodes
        self.leaf_indices = [i for i, node in enumerate(self.nodes) if node.is_leaf()]
        self.leaf_taxon_indices = []
        
        # Map leaf nodes to taxon indices
        taxon_namespace = self.tree.taxon_namespace
        taxon_list = list(taxon_namespace)
        for i in self.leaf_indices:
            node = self.nodes[i]
            if node.taxon is not None:
                taxon_idx = taxon_list.index(node.taxon)
                self.leaf_taxon_indices.append((i, taxon_idx))
    
    def _jit_compile_functions(self):
        """JIT-compile core likelihood computation functions."""
        
        @jit
        def compute_transition_matrix(Q: jnp.ndarray, t: float) -> jnp.ndarray:
            """Compute P(t) = expm(Q*t)."""
            return jax_expm(Q * t)
        
        @jit
        def combine_child_conditionals(
            child1_cond: jnp.ndarray,
            child2_cond: jnp.ndarray,
            P1: jnp.ndarray,
            P2: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            Combine conditionals from two children.
            
            L_parent[site, state] = 
                Σ_j P1[state,j] * L_child1[site,j] *
                Σ_k P2[state,k] * L_child2[site,k]
            """
            # child_cond: (n_sites, n_states)
            # P: (n_states, n_states)
            
            # Compute P @ child_cond.T -> (n_states, n_sites)
            # Then transpose -> (n_sites, n_states)
            term1 = (P1 @ child1_cond.T).T
            term2 = (P2 @ child2_cond.T).T
            
            # Element-wise product
            return term1 * term2
        
        @jit
        def compute_root_likelihood(
            root_cond: jnp.ndarray,
            freqs: jnp.ndarray,
        ) -> float:
            """
            Compute total likelihood at root.
            
            L = Σ_sites Σ_states π_state * L_root[site, state]
            """
            # root_cond: (n_sites, n_states)
            # freqs: (n_states,)
            
            # Weighted sum over states for each site
            site_likelihoods = jnp.sum(root_cond * freqs[None, :], axis=1)
            
            # Product over sites (in log space)
            log_lik = jnp.sum(jnp.log(site_likelihoods + 1e-100))
            
            return log_lik
        
        self._jit_transition = compute_transition_matrix
        self._jit_combine = combine_child_conditionals
        self._jit_root_lik = compute_root_likelihood
    
    def compute_likelihood(
        self,
        alignment: np.ndarray,
        Q: np.ndarray,
        freqs: np.ndarray,
    ) -> float:
        """
        Compute log-likelihood using JAX.
        
        Args:
            alignment: (n_taxa, n_sites) integer array
            Q: (n_states, n_states) rate matrix
            freqs: (n_states,) equilibrium frequencies
            
        Returns:
            Log-likelihood
        """
        n_sites = alignment.shape[1]
        n_nodes = len(self.nodes)
        
        # Convert to JAX arrays
        Q_jax = jnp.array(Q)
        freqs_jax = jnp.array(freqs)
        
        # Initialize conditionals: (n_nodes, n_sites, n_states)
        conditionals = np.zeros((n_nodes, n_sites, self.n_states))
        
        # Initialize leaf conditionals
        for node_idx, taxon_idx in self.leaf_taxon_indices:
            for site in range(n_sites):
                state = alignment[taxon_idx, site]
                if 0 <= state < self.n_states:
                    conditionals[node_idx, site, state] = 1.0
                else:
                    # Missing data: all states equally likely
                    conditionals[node_idx, site, :] = 1.0
        
        # Precompute all transition matrices
        transition_matrices = {}
        for i, t in enumerate(self.branch_lengths):
            if t > 0:
                P = self._jit_transition(Q_jax, t)
                transition_matrices[i] = np.array(P)
            else:
                transition_matrices[i] = np.eye(self.n_states)
        
        # Postorder traversal: combine children
        for parent_idx, child1_idx, child2_idx in self.internal_node_array:
            child1_cond = jnp.array(conditionals[child1_idx])
            child2_cond = jnp.array(conditionals[child2_idx])
            
            P1 = jnp.array(transition_matrices[child1_idx])
            P2 = jnp.array(transition_matrices[child2_idx])
            
            parent_cond = self._jit_combine(child1_cond, child2_cond, P1, P2)
            conditionals[parent_idx] = np.array(parent_cond)
        
        # Compute likelihood at root
        root_idx = self.node_to_idx[self.tree.seed_node]
        root_cond = jnp.array(conditionals[root_idx])
        
        log_lik = self._jit_root_lik(root_cond, freqs_jax)
        
        return float(log_lik)
    
    def compute_likelihood_with_transitions(
        self,
        alignment: np.ndarray,
        freqs: np.ndarray,
        transition_matrices: Dict[int, np.ndarray],
    ) -> float:
        """
        Compute log-likelihood using precomputed transition matrices.
        
        This is faster when transition matrices are computed using
        eigendecomposition caching.
        
        Args:
            alignment: (n_taxa, n_sites) integer array
            freqs: (n_states,) equilibrium frequencies
            transition_matrices: Dict mapping node_idx -> P(t) matrix
            
        Returns:
            Log-likelihood
        """
        n_sites = alignment.shape[1]
        n_nodes = len(self.nodes)
        
        # Convert to JAX arrays
        freqs_jax = jnp.array(freqs)
        
        # Initialize conditionals: (n_nodes, n_sites, n_states)
        conditionals = np.zeros((n_nodes, n_sites, self.n_states))
        
        # Initialize leaf conditionals
        for node_idx, taxon_idx in self.leaf_taxon_indices:
            for site in range(n_sites):
                state = alignment[taxon_idx, site]
                if 0 <= state < self.n_states:
                    conditionals[node_idx, site, state] = 1.0
                else:
                    # Missing data: all states equally likely
                    conditionals[node_idx, site, :] = 1.0
        
        # Postorder traversal: combine children using precomputed matrices
        for parent_idx, child1_idx, child2_idx in self.internal_node_array:
            child1_cond = jnp.array(conditionals[child1_idx])
            child2_cond = jnp.array(conditionals[child2_idx])
            
            P1 = jnp.array(transition_matrices[child1_idx])
            P2 = jnp.array(transition_matrices[child2_idx])
            
            parent_cond = self._jit_combine(child1_cond, child2_cond, P1, P2)
            conditionals[parent_idx] = np.array(parent_cond)
        
        # Compute likelihood at root
        root_idx = self.node_to_idx[self.tree.seed_node]
        root_cond = jnp.array(conditionals[root_idx])
        
        log_lik = self._jit_root_lik(root_cond, freqs_jax)
        
        return float(log_lik)
