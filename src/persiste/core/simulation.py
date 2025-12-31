"""
Core simulation utilities for phylogenetic models.

Provides efficient simulation of discrete trait evolution on trees,
used for validation, testing, and power analysis.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.linalg import expm

from .trees import TreeStructure


def simulate_binary_evolution(
    tree: TreeStructure,
    gain_rate: float,
    loss_rate: float,
    n_sites: int,
    rng: Optional[np.random.Generator] = None,
    site_specific_rates: Optional[Dict[int, Tuple[float, float]]] = None,
    root_frequencies: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Simulate binary trait evolution (0/1) on a phylogenetic tree.
    
    Uses continuous-time Markov chain with gain/loss rates.
    Simulates each site independently along the tree.
    
    Args:
        tree: TreeStructure with branch lengths
        gain_rate: Rate of 0→1 transitions (global default)
        loss_rate: Rate of 1→0 transitions (global default)
        n_sites: Number of independent sites to simulate
        rng: Random number generator (default: create new one)
        site_specific_rates: Optional dict mapping site_idx → (gain, loss)
                           for heterogeneous rates across sites
        root_frequencies: Optional (π₀, π₁) root state frequencies
                         (default: equilibrium frequencies)
    
    Returns:
        presence_matrix: (n_tips, n_sites) binary array
        
    Example:
        >>> tree = TreeStructure.from_newick("((A:1,B:1):1,(C:1,D:1):1);")
        >>> rng = np.random.default_rng(42)
        >>> matrix = simulate_binary_evolution(tree, 1.5, 2.0, 100, rng)
        >>> matrix.shape
        (4, 100)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Compute equilibrium frequencies if not provided
    if root_frequencies is None:
        pi_0 = loss_rate / (gain_rate + loss_rate)
        pi_1 = gain_rate / (gain_rate + loss_rate)
        root_frequencies = (pi_0, pi_1)
    
    # Initialize output matrix
    n_tips = tree.n_tips
    presence_matrix = np.zeros((n_tips, n_sites), dtype=np.int8)
    
    # Simulate each site
    for site_idx in range(n_sites):
        # Get rates for this site
        if site_specific_rates is not None and site_idx in site_specific_rates:
            site_gain, site_loss = site_specific_rates[site_idx]
        else:
            site_gain, site_loss = gain_rate, loss_rate
        
        # Build rate matrix Q
        Q = np.array([
            [-site_gain, site_gain],
            [site_loss, -site_loss]
        ])
        
        # Initialize root state
        pi_0, pi_1 = compute_equilibrium_frequencies(gain_rate, loss_rate)
        root_state = rng.choice([0, 1], p=[pi_0, pi_1])
        node_states = {int(tree.root_index): root_state}
        
        # Build children dictionary for proper traversal
        children = {i: [] for i in range(tree.n_nodes)}
        for child_idx in range(tree.n_nodes):
            parent_idx = tree.parent_indices[child_idx]
            if parent_idx >= 0:
                children[int(parent_idx)].append(child_idx)
        
        # Traverse tree from root to tips (pre-order)
        def simulate_subtree(node_idx):
            for child_idx in children[node_idx]:
                parent_state = node_states[int(node_idx)]
                branch_length = tree.branch_lengths[child_idx]
                
                # Compute transition probability matrix
                P = expm(Q * branch_length)
                
                # Sample child state
                child_state = rng.choice([0, 1], p=P[parent_state, :])
                node_states[int(child_idx)] = child_state
                
                # Recurse to children
                simulate_subtree(child_idx)
        
        simulate_subtree(tree.root_index)
        
        # Extract tip states
        for tip_idx_pos, tip_idx in enumerate(tree.tip_indices):
            presence_matrix[tip_idx_pos, site_idx] = node_states[int(tip_idx)]
    
    return presence_matrix


def simulate_binary_evolution_vectorized(
    tree: TreeStructure,
    gain_rate: float,
    loss_rate: float,
    n_sites: int,
    rng: Optional[np.random.Generator] = None,
    site_specific_rates: Optional[Dict[int, Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Vectorized version of binary evolution simulation.
    
    More efficient for large numbers of sites by batching operations.
    Currently just calls the standard version - JAX implementation
    would provide true vectorization.
    
    Args:
        Same as simulate_binary_evolution
        
    Returns:
        presence_matrix: (n_tips, n_sites) binary array
        
    Note:
        This is a placeholder for future JAX implementation.
        For now, it's equivalent to simulate_binary_evolution.
    """
    # TODO: Implement JAX-based vectorization
    # For now, use standard implementation
    return simulate_binary_evolution(
        tree=tree,
        gain_rate=gain_rate,
        loss_rate=loss_rate,
        n_sites=n_sites,
        rng=rng,
        site_specific_rates=site_specific_rates,
    )


def compute_equilibrium_frequencies(
    gain_rate: float,
    loss_rate: float
) -> Tuple[float, float]:
    """
    Compute equilibrium frequencies for binary trait evolution.
    
    Args:
        gain_rate: Rate of 0→1 transitions
        loss_rate: Rate of 1→0 transitions
        
    Returns:
        (pi_0, pi_1): Equilibrium frequencies for states 0 and 1
        
    Example:
        >>> pi_0, pi_1 = compute_equilibrium_frequencies(1.5, 2.0)
        >>> pi_0
        0.5714285714285714
        >>> pi_1
        0.42857142857142855
        >>> pi_0 + pi_1
        1.0
    """
    total = gain_rate + loss_rate
    pi_0 = loss_rate / total
    pi_1 = gain_rate / total
    return pi_0, pi_1


def compute_stationary_frequency(
    gain_rate: float,
    loss_rate: float
) -> float:
    """
    Compute stationary frequency π₁ (presence frequency).
    
    This is the equilibrium probability of state 1 (present).
    More identifiable than individual rates.
    
    Args:
        gain_rate: Rate of 0→1 transitions
        loss_rate: Rate of 1→0 transitions
        
    Returns:
        pi_1: Stationary frequency of state 1
        
    Example:
        >>> compute_stationary_frequency(1.5, 2.0)
        0.42857142857142855
    """
    return gain_rate / (gain_rate + loss_rate)


def compute_mean_transitions(
    tree: TreeStructure,
    gain_rate: float,
    loss_rate: float
) -> float:
    """
    Compute expected number of transitions per branch.
    
    Args:
        tree: TreeStructure with branch lengths
        gain_rate: Rate of 0→1 transitions
        loss_rate: Rate of 1→0 transitions
        
    Returns:
        mean_transitions: Expected transitions per branch
        
    Example:
        >>> tree = TreeStructure.from_newick("((A:1,B:1):1,(C:1,D:1):1);")
        >>> compute_mean_transitions(tree, 1.5, 2.0)
        3.5
    """
    total_rate = gain_rate + loss_rate
    mean_branch_length = np.mean(tree.branch_lengths[tree.branch_lengths > 0])
    return total_rate * mean_branch_length
