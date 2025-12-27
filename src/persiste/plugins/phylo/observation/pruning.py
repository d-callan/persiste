"""Felsenstein pruning algorithm for phylogenetic likelihood."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import expm

try:
    import dendropy
    DENDROPY_AVAILABLE = True
except ImportError:
    DENDROPY_AVAILABLE = False
    dendropy = None


class FelsensteinPruning:
    """
    Felsenstein's pruning algorithm for computing phylogenetic likelihood.
    
    The pruning algorithm computes the likelihood of observing data at the tips
    of a phylogenetic tree given a continuous-time Markov chain (CTMC) model.
    
    Key concepts:
    - Conditional likelihoods: P(data below node | state at node)
    - Postorder traversal: compute from leaves to root
    - Transition probabilities: P(t) = exp(Qt) via matrix exponential
    
    This is the core computational engine for all phylogenetic selection analyses.
    
    References:
    - Felsenstein (1981) "Evolutionary trees from DNA sequences"
    - Yang (2006) "Computational Molecular Evolution" Chapter 4
    """
    
    def __init__(
        self,
        tree: "dendropy.Tree",
        n_states: int,
        rate_matrix_fn: callable,
    ):
        """
        Initialize pruning algorithm.
        
        Args:
            tree: DendroPy Tree object
            n_states: Number of states (e.g., 61 for codons)
            rate_matrix_fn: Function that returns rate matrix Q given parameters
        """
        self.tree = tree
        self.n_states = n_states
        self.rate_matrix_fn = rate_matrix_fn
        
        # Cache for conditional likelihoods
        # conditionals[node] = (n_sites, n_states) array
        self.conditionals: Dict = {}
        
        # Cache for transition probability matrices
        # P[node] = exp(Q * t) where t is branch length
        self.transition_probs: Dict = {}
        
        # Cache for transition matrices keyed by (Q_hash, branch_length)
        # This avoids recomputing expm(Q*t) for the same Q across sites
        self._transition_cache: Dict[Tuple[int, float], np.ndarray] = {}
        self._last_Q_hash: Optional[int] = None
    
    def compute_transition_probabilities(self, Q: np.ndarray):
        """
        Compute transition probability matrices for all branches.
        
        P(t) = exp(Q * t) for each branch with length t.
        
        Args:
            Q: Rate matrix (n_states × n_states)
        """
        self.transition_probs.clear()
        
        for node in self.tree.postorder_node_iter():
            if node.edge_length is not None and node.edge_length > 0:
                t = node.edge_length
                # P(t) = exp(Q * t)
                self.transition_probs[node] = expm(Q * t)
            else:
                # No branch or zero length: identity matrix
                self.transition_probs[node] = np.eye(self.n_states)
    
    def initialize_leaf_conditionals(self, site_data: np.ndarray):
        """
        Initialize conditional likelihoods at leaves.
        
        For observed data:
        - L[node, site, state] = 1 if state matches observation, 0 otherwise
        
        For ambiguous/missing data:
        - L[node, site, state] = 1 for all states (integrate over uncertainty)
        
        Args:
            site_data: (n_taxa, n_sites) array of state indices
                       -1 indicates missing/ambiguous data
        """
        self.conditionals.clear()
        
        leaf_idx = 0
        for node in self.tree.leaf_node_iter():
            if leaf_idx >= site_data.shape[0]:
                raise ValueError(f"More leaves in tree than taxa in data")
            
            n_sites = site_data.shape[1]
            conditionals = np.zeros((n_sites, self.n_states))
            
            for site in range(n_sites):
                state = site_data[leaf_idx, site]
                if state == -1:
                    # Missing data: all states equally likely
                    conditionals[site, :] = 1.0
                else:
                    # Observed state
                    conditionals[site, state] = 1.0
            
            self.conditionals[node] = conditionals
            leaf_idx += 1
    
    def compute_internal_conditionals(self):
        """
        Compute conditional likelihoods at internal nodes via postorder traversal.
        
        For internal node i with children j, k:
        L[i, site, state_i] = 
            (Σ_j P(state_i → state_j) * L[j, site, state_j]) *
            (Σ_k P(state_i → state_k) * L[k, site, state_k])
        
        This is the core of the pruning algorithm.
        """
        for node in self.tree.postorder_node_iter():
            if node.is_leaf():
                continue  # Already initialized
            
            children = list(node.child_node_iter())
            if len(children) == 0:
                continue  # Root with no children (shouldn't happen)
            
            # Get first child's conditionals to determine n_sites
            first_child = children[0]
            n_sites = self.conditionals[first_child].shape[0]
            
            # Initialize conditionals for this node
            conditionals = np.ones((n_sites, self.n_states))
            
            # Multiply contributions from all children
            for child in children:
                child_conditionals = self.conditionals[child]
                P = self.transition_probs[child]  # (n_states, n_states)
                
                # For each site and parent state, sum over child states
                # result[site, parent_state] = Σ_child_state P[parent, child] * L[child, site, child_state]
                # This is a matrix multiplication: L @ P.T
                contribution = child_conditionals @ P.T  # (n_sites, n_states)
                
                conditionals *= contribution
            
            self.conditionals[node] = conditionals
    
    def compute_log_likelihood(self, root_frequencies: np.ndarray) -> float:
        """
        Compute log-likelihood at root.
        
        L = Σ_site log(Σ_state π[state] * L[root, site, state])
        
        where π is the equilibrium frequency distribution.
        
        Args:
            root_frequencies: (n_states,) array of equilibrium frequencies
            
        Returns:
            Log-likelihood
        """
        root = self.tree.seed_node
        root_conditionals = self.conditionals[root]  # (n_sites, n_states)
        
        # Site likelihoods: π[state] * L[root, site, state], sum over states
        site_likelihoods = root_conditionals @ root_frequencies  # (n_sites,)
        
        # Log-likelihood: sum of log site likelihoods
        log_likelihood = np.sum(np.log(site_likelihoods))
        
        return log_likelihood
    
    def compute_likelihood(
        self,
        site_data: np.ndarray,
        Q: np.ndarray,
        root_frequencies: np.ndarray,
    ) -> float:
        """
        Compute full phylogenetic likelihood.
        
        This is the main entry point that orchestrates the pruning algorithm:
        1. Compute transition probabilities P(t) = exp(Qt) for all branches
        2. Initialize leaf conditionals from observed data
        3. Compute internal conditionals via postorder traversal
        4. Compute likelihood at root
        
        Args:
            site_data: (n_taxa, n_sites) array of state indices
            Q: Rate matrix (n_states, n_states)
            root_frequencies: (n_states,) equilibrium frequencies
            
        Returns:
            Log-likelihood
        """
        # Step 1: Compute P(t) = exp(Qt) for all branches
        self.compute_transition_probabilities(Q)
        
        # Step 2: Initialize leaves
        self.initialize_leaf_conditionals(site_data)
        
        # Step 3: Compute internal nodes (postorder)
        self.compute_internal_conditionals()
        
        # Step 4: Compute likelihood at root
        log_likelihood = self.compute_log_likelihood(root_frequencies)
        
        return log_likelihood
    
    def compute_site_likelihoods(
        self,
        site_data: np.ndarray,
        Q: np.ndarray,
        root_frequencies: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-site log-likelihoods.
        
        Useful for site-level analyses like FEL.
        
        Args:
            site_data: (n_taxa, n_sites) array of state indices
            Q: Rate matrix (n_states, n_states)
            root_frequencies: (n_states,) equilibrium frequencies
            
        Returns:
            (n_sites,) array of log-likelihoods
        """
        # Run pruning algorithm
        self.compute_transition_probabilities(Q)
        self.initialize_leaf_conditionals(site_data)
        self.compute_internal_conditionals()
        
        # Get root conditionals
        root = self.tree.seed_node
        root_conditionals = self.conditionals[root]  # (n_sites, n_states)
        
        # Site likelihoods
        site_likelihoods = root_conditionals @ root_frequencies  # (n_sites,)
        
        return np.log(site_likelihoods)
    
    def clear_cache(self):
        """Clear cached conditionals and transition probabilities."""
        self.conditionals.clear()
        self.transition_probs.clear()


def compute_phylogenetic_likelihood(
    tree: "dendropy.Tree",
    alignment: np.ndarray,
    rate_matrix: np.ndarray,
    frequencies: np.ndarray,
) -> float:
    """
    Convenience function for computing phylogenetic likelihood.
    
    Args:
        tree: DendroPy Tree object
        alignment: (n_taxa, n_sites) array of state indices
        rate_matrix: (n_states, n_states) rate matrix Q
        frequencies: (n_states,) equilibrium frequencies
        
    Returns:
        Log-likelihood
    """
    n_states = rate_matrix.shape[0]
    
    pruning = FelsensteinPruning(
        tree=tree,
        n_states=n_states,
        rate_matrix_fn=lambda: rate_matrix,
    )
    
    return pruning.compute_likelihood(alignment, rate_matrix, frequencies)
