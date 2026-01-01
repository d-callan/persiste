"""
Tree inference from presence/absence matrices (PAM).

Provides fast, transparent tree inference when phylogeny is not available.
All inference is explicit and metadata is preserved.

This module is now a thin wrapper around persiste.core.tree_building.
"""

from typing import Literal, Tuple
import numpy as np

from persiste.core.trees import TreeStructure
from persiste.core.tree_building import (
    TreeInferenceMetadata,
    jaccard_distance as _jaccard_distance,
    hamming_distance as _hamming_distance,
    upgma_tree as _upgma_tree,
    neighbor_joining_tree as _neighbor_joining_tree,
    infer_tree_from_binary_matrix,
)


def jaccard_distance(pam: np.ndarray, use_rust: bool = True) -> np.ndarray:
    """Wrapper for core jaccard_distance - kept for backward compatibility."""
    return _jaccard_distance(pam, use_rust)


def hamming_distance(pam: np.ndarray, use_rust: bool = True) -> np.ndarray:
    """Wrapper for core hamming_distance - kept for backward compatibility."""
    return _hamming_distance(pam, use_rust)


def upgma_tree(distance_matrix: np.ndarray, taxon_names: list[str]) -> TreeStructure:
    """Wrapper for core upgma_tree - kept for backward compatibility."""
    return _upgma_tree(distance_matrix, taxon_names)


def neighbor_joining_tree(distance_matrix: np.ndarray, taxon_names: list[str]) -> TreeStructure:
    """Wrapper for core neighbor_joining_tree - kept for backward compatibility."""
    return _neighbor_joining_tree(distance_matrix, taxon_names)


def infer_tree_from_pam(
    pam: np.ndarray,
    taxon_names: list[str],
    method: Literal["jaccard_upgma", "hamming_upgma", "jaccard_nj"] = "jaccard_upgma"
) -> Tuple[TreeStructure, TreeInferenceMetadata]:
    """
    Infer phylogenetic tree from presence/absence matrix.
    
    Wrapper around core infer_tree_from_binary_matrix for backward compatibility.
    
    Args:
        pam: Binary presence/absence matrix (n_taxa × n_genes)
        taxon_names: List of taxon names (length n_taxa)
        method: Tree inference method
            
    Returns:
        tree: Inferred TreeStructure
        metadata: TreeInferenceMetadata with method details
    """
    tree, metadata = infer_tree_from_binary_matrix(pam, taxon_names, method)
    # Update metadata to use n_genes instead of n_features for PAM context
    metadata.n_genes = metadata.n_features
    return tree, metadata


def validate_inferred_tree(tree: TreeStructure, pam: np.ndarray) -> dict:
    """
    Validate inferred tree for mechanical correctness.
    
    Checks:
    - Tree is valid (all nodes reachable from root)
    - Branch lengths are positive and finite
    - Number of tips matches PAM rows
    
    Args:
        tree: Inferred TreeStructure
        pam: Original presence/absence matrix
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check tree structure
    if tree.n_tips != pam.shape[0]:
        results["valid"] = False
        results["errors"].append(f"Tree has {tree.n_tips} tips but PAM has {pam.shape[0]} rows")
    
    # Check branch lengths
    if not np.all(np.isfinite(tree.branch_lengths)):
        results["valid"] = False
        results["errors"].append("Tree has non-finite branch lengths")
    
    if np.any(tree.branch_lengths < 0):
        results["valid"] = False
        results["errors"].append("Tree has negative branch lengths")
    
    # Check for very short branches (potential numerical issues)
    positive_branches = tree.branch_lengths[tree.branch_lengths > 0]
    if len(positive_branches) > 0:
        min_branch = np.min(positive_branches)
        if min_branch < 1e-10:
            results["warnings"].append(f"Very short branch lengths detected (min: {min_branch:.2e})")
    
    # Check for very long branches (potential outliers)
    max_branch = np.max(tree.branch_lengths)
    if max_branch > 10:
        results["warnings"].append(f"Very long branch detected (max: {max_branch:.2f})")
    
    return results
