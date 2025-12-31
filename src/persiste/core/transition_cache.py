"""
Global transition matrix cache for gene content inference.

Provides LRU caching of transition matrices across all families,
giving 2-3x speedup for global rate models where all families
share the same gain/loss rates.
"""

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=10000)
def compute_binary_transition_matrix_cached(
    gain_rate: float,
    loss_rate: float,
    branch_length: float
) -> tuple:
    """
    Compute transition matrix for binary evolution with LRU caching.
    
    This function is cached globally, so all families with the same
    rates benefit from shared computation.
    
    Args:
        gain_rate: Rate of 0→1 transitions
        loss_rate: Rate of 1→0 transitions
        branch_length: Branch length (time)
        
    Returns:
        Tuple of matrix elements (for hashability)
    """
    total = gain_rate + loss_rate
    
    if total < 1e-10:
        return (1.0, 0.0, 0.0, 1.0)  # Identity matrix
    
    exp_term = np.exp(-total * branch_length)
    
    p00 = (loss_rate + gain_rate * exp_term) / total
    p01 = (gain_rate - gain_rate * exp_term) / total
    p10 = (loss_rate - loss_rate * exp_term) / total
    p11 = (gain_rate + loss_rate * exp_term) / total
    
    return (p00, p01, p10, p11)


def get_binary_transition_matrix(
    gain_rate: float,
    loss_rate: float,
    branch_length: float,
    use_cache: bool = True
) -> np.ndarray:
    """
    Get transition matrix for binary evolution.
    
    Args:
        gain_rate: Rate of 0→1 transitions
        loss_rate: Rate of 1→0 transitions
        branch_length: Branch length (time)
        use_cache: Whether to use global LRU cache
        
    Returns:
        2×2 transition probability matrix
    """
    if use_cache:
        p00, p01, p10, p11 = compute_binary_transition_matrix_cached(
            gain_rate, loss_rate, branch_length
        )
        return np.array([[p00, p01], [p10, p11]])
    else:
        # Direct computation without caching
        total = gain_rate + loss_rate
        
        if total < 1e-10:
            return np.eye(2)
        
        exp_term = np.exp(-total * branch_length)
        
        p00 = (loss_rate + gain_rate * exp_term) / total
        p01 = (gain_rate - gain_rate * exp_term) / total
        p10 = (loss_rate - loss_rate * exp_term) / total
        p11 = (gain_rate + loss_rate * exp_term) / total
        
        return np.array([[p00, p01], [p10, p11]])


def clear_transition_cache():
    """Clear the global transition matrix cache."""
    compute_binary_transition_matrix_cached.cache_clear()


def get_cache_info():
    """Get cache statistics."""
    return compute_binary_transition_matrix_cached.cache_info()
