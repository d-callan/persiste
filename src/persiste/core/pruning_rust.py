"""
Rust-accelerated pruning with automatic fallback to NumPy.

This module provides a unified interface that uses Rust parallelization
when available, falling back to NumPy for compatibility.
"""

import numpy as np
from typing import Optional
import warnings

# Try to import Rust extension
try:
    from persiste_rust import compute_likelihoods_parallel as _rust_compute_likelihoods
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    _rust_compute_likelihoods = None

from .trees import TreeStructure
from .pruning import FelsensteinPruning, SimpleBinaryTransitionProvider, ArrayTipConditionalProvider


def compute_likelihoods_batch(
    tree: TreeStructure,
    presence_matrix: np.ndarray,
    gain_rates: np.ndarray,
    loss_rates: np.ndarray,
    taxon_names: list,
    use_rust: bool = True,
) -> np.ndarray:
    """
    Compute log-likelihoods for all families.
    
    Automatically uses Rust parallelization if available, otherwise falls back
    to NumPy sequential computation.
    
    Args:
        tree: TreeStructure object
        presence_matrix: (n_tips, n_families) binary presence/absence data
        gain_rates: (n_families,) gain rates
        loss_rates: (n_families,) loss rates
        taxon_names: List of taxon names matching tree tips
        use_rust: Whether to use Rust if available (default: True)
        
    Returns:
        (n_families,) array of log-likelihoods
    """
    if use_rust and RUST_AVAILABLE:
        return _compute_likelihoods_rust(
            tree, presence_matrix, gain_rates, loss_rates, taxon_names
        )
    else:
        return _compute_likelihoods_numpy(
            tree, presence_matrix, gain_rates, loss_rates, taxon_names
        )


def _compute_likelihoods_rust(
    tree: TreeStructure,
    presence_matrix: np.ndarray,
    gain_rates: np.ndarray,
    loss_rates: np.ndarray,
    taxon_names: list,
) -> np.ndarray:
    """Compute likelihoods using Rust parallelization."""
    # Pass tree structure directly to Rust
    log_likelihoods = _rust_compute_likelihoods(
        tree.parent_indices.astype(np.int32),
        tree.branch_lengths.astype(np.float64),
        presence_matrix.astype(np.int8),
        gain_rates.astype(np.float64),
        loss_rates.astype(np.float64),
        int(tree.n_tips),
    )
    
    return np.array(log_likelihoods)


def _compute_likelihoods_numpy(
    tree: TreeStructure,
    presence_matrix: np.ndarray,
    gain_rates: np.ndarray,
    loss_rates: np.ndarray,
    taxon_names: list,
) -> np.ndarray:
    """Compute likelihoods using NumPy (sequential fallback)."""
    n_families = presence_matrix.shape[1]
    log_likelihoods = np.zeros(n_families)
    
    pruning = FelsensteinPruning(tree, n_states=2, use_jax=False)
    
    # Use tree's own tip names for consistency
    tree_tip_names = tree.tip_names
    
    for fam_idx in range(n_families):
        # Create transition provider
        transition_provider = SimpleBinaryTransitionProvider(
            gain_rates[fam_idx],
            loss_rates[fam_idx],
            use_cache=True
        )
        
        # Create tip provider
        tip_data = presence_matrix[:, fam_idx:fam_idx+1]
        tip_provider = ArrayTipConditionalProvider(tip_data, tree_tip_names, n_states=2)
        
        # Compute likelihood
        result = pruning.compute_likelihood(
            transition_provider=transition_provider,
            tip_provider=tip_provider,
            n_sites=1
        )
        log_likelihoods[fam_idx] = result.log_likelihood
    
    return log_likelihoods


def _tree_to_newick(tree: TreeStructure) -> str:
    """
    Convert TreeStructure to Newick format.
    
    This is a helper function for Rust integration.
    TODO: Move this to TreeStructure.to_newick() method.
    """
    def build_newick(node_idx: int) -> str:
        node = tree.nodes[node_idx]
        
        if node.is_tip:
            name = node.name or f"tip{node_idx}"
            branch_len = tree.branch_lengths[node_idx]
            return f"{name}:{branch_len:.6f}"
        else:
            # Internal node: recursively build children
            children_newick = []
            for child_idx in range(tree.n_nodes):
                if tree.parent_indices[child_idx] == node_idx:
                    children_newick.append(build_newick(child_idx))
            
            children_str = ",".join(children_newick)
            branch_len = tree.branch_lengths[node_idx]
            
            if node_idx == tree.root_index:
                return f"({children_str});"
            else:
                return f"({children_str}):{branch_len:.6f}"
    
    return build_newick(tree.root_index)


def check_rust_available() -> bool:
    """Check if Rust acceleration is available."""
    return RUST_AVAILABLE


def get_backend_info() -> dict:
    """Get information about available backends."""
    return {
        'rust_available': RUST_AVAILABLE,
        'default_backend': 'rust' if RUST_AVAILABLE else 'numpy',
    }


def benchmark_backends(
    tree: TreeStructure,
    presence_matrix: np.ndarray,
    gain_rates: np.ndarray,
    loss_rates: np.ndarray,
    taxon_names: list,
) -> dict:
    """
    Benchmark Rust vs NumPy backends.
    
    Returns:
        Dictionary with timing and speedup information
    """
    import time
    
    # Benchmark NumPy
    start = time.time()
    ll_numpy = _compute_likelihoods_numpy(
        tree, presence_matrix, gain_rates, loss_rates, taxon_names
    )
    time_numpy = time.time() - start
    
    results = {
        'numpy_time': time_numpy,
        'numpy_throughput': len(gain_rates) / time_numpy,
    }
    
    # Benchmark Rust if available
    if RUST_AVAILABLE:
        start = time.time()
        ll_rust = _compute_likelihoods_rust(
            tree, presence_matrix, gain_rates, loss_rates, taxon_names
        )
        time_rust = time.time() - start
        
        results['rust_time'] = time_rust
        results['rust_throughput'] = len(gain_rates) / time_rust
        results['speedup'] = time_numpy / time_rust
        
        # Check correctness
        max_diff = np.max(np.abs(ll_numpy - ll_rust))
        results['max_difference'] = max_diff
        results['correct'] = max_diff < 1e-6
    else:
        results['rust_time'] = None
        results['speedup'] = None
        results['max_difference'] = None
        results['correct'] = None
    
    return results
