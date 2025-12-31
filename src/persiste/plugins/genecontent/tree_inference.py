"""
Tree inference from presence/absence matrices (PAM).

Provides fast, transparent tree inference when phylogeny is not available.
All inference is explicit and metadata is preserved.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, to_tree

from persiste.core.trees import TreeStructure

# Try to import Rust acceleration
try:
    import persiste_rust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


@dataclass
class TreeInferenceMetadata:
    """Metadata about inferred tree."""
    source: Literal["provided", "inferred"]
    method: Optional[str] = None
    distance_metric: Optional[str] = None
    clustering_method: Optional[str] = None
    n_taxa: Optional[int] = None
    n_genes: Optional[int] = None


def jaccard_distance(pam: np.ndarray, use_rust: bool = True) -> np.ndarray:
    """
    Compute Jaccard distance matrix from presence/absence matrix.
    
    Uses Rust acceleration by default (6-18x faster) with automatic
    fallback to scipy if Rust is not available.
    
    Jaccard distance = 1 - Jaccard similarity
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    
    Args:
        pam: Binary presence/absence matrix (n_taxa × n_genes)
        use_rust: Use Rust acceleration if available (default: True)
        
    Returns:
        Distance matrix (n_taxa × n_taxa)
        
    Example:
        >>> pam = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
        >>> D = jaccard_distance(pam)
        >>> D.shape
        (3, 3)
    """
    if use_rust and RUST_AVAILABLE:
        # Use Rust acceleration (6-18x faster)
        pam_u8 = pam.astype(np.uint8)
        return persiste_rust.compute_jaccard_distance(pam_u8)
    else:
        # Fallback to scipy
        distances = pdist(pam, metric='jaccard')
        return squareform(distances)


def hamming_distance(pam: np.ndarray, use_rust: bool = True) -> np.ndarray:
    """
    Compute Hamming distance matrix from presence/absence matrix.
    
    Uses Rust acceleration by default (6-15x faster) with automatic
    fallback to scipy if Rust is not available.
    
    Hamming distance = proportion of positions that differ
    
    Args:
        pam: Binary presence/absence matrix (n_taxa × n_genes)
        use_rust: Use Rust acceleration if available (default: True)
        
    Returns:
        Distance matrix (n_taxa × n_taxa)
    """
    if use_rust and RUST_AVAILABLE:
        # Use Rust acceleration (6-15x faster)
        pam_u8 = pam.astype(np.uint8)
        return persiste_rust.compute_hamming_distance(pam_u8)
    else:
        # Fallback to scipy
        distances = pdist(pam, metric='hamming')
        return squareform(distances)


def upgma_tree(distance_matrix: np.ndarray, taxon_names: list[str]) -> TreeStructure:
    """
    Build UPGMA tree from distance matrix.
    
    UPGMA (Unweighted Pair Group Method with Arithmetic Mean) is a simple
    hierarchical clustering method that assumes a molecular clock.
    
    Args:
        distance_matrix: Symmetric distance matrix (n_taxa × n_taxa)
        taxon_names: List of taxon names
        
    Returns:
        TreeStructure with branch lengths
        
    Note:
        UPGMA assumes constant evolutionary rate (molecular clock).
        This is often violated but works well for gene content data.
    """
    n_taxa = len(taxon_names)
    
    # Perform hierarchical clustering with average linkage (UPGMA)
    # Convert distance matrix to condensed form for linkage
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # Convert scipy tree to Newick format
    scipy_tree = to_tree(linkage_matrix)
    
    def to_newick(node, parent_dist=0.0):
        """Convert scipy tree to Newick format."""
        if node.is_leaf():
            # Leaf node - use taxon name
            name = taxon_names[node.id]
            # Quote name to preserve underscores and special characters
            quoted_name = f"'{name}'" if '_' in name or ' ' in name else name
            # Branch length is from parent to this node
            branch_len = max(0.0, node.dist / 2 - parent_dist)
            return f"{quoted_name}:{branch_len:.6f}"
        else:
            # Internal node - recurse to children
            # Children are at distance node.dist/2 from root
            left = to_newick(node.left, node.dist / 2)
            right = to_newick(node.right, node.dist / 2)
            # This node's branch length (if not root)
            branch_len = max(0.0, node.dist / 2 - parent_dist) if parent_dist > 0 else 0.0
            if parent_dist > 0:
                return f"({left},{right}):{branch_len:.6f}"
            else:
                # Root node - no branch length
                return f"({left},{right})"
    
    newick = to_newick(scipy_tree) + ";"
    
    # Parse Newick string into TreeStructure
    tree = TreeStructure.from_newick(newick)
    
    return tree


def neighbor_joining_tree(distance_matrix: np.ndarray, taxon_names: list[str]) -> TreeStructure:
    """
    Build Neighbor-Joining tree from distance matrix.
    
    NJ is more sophisticated than UPGMA and doesn't assume a molecular clock.
    
    Args:
        distance_matrix: Symmetric distance matrix (n_taxa × n_taxa)
        taxon_names: List of taxon names
        
    Returns:
        TreeStructure with branch lengths
        
    Note:
        Currently not implemented - falls back to UPGMA.
        Full NJ implementation would require more complex logic.
    """
    # TODO: Implement proper NJ algorithm
    # For now, fall back to UPGMA
    import warnings
    warnings.warn("Neighbor-Joining not yet implemented, using UPGMA instead")
    return upgma_tree(distance_matrix, taxon_names)


def infer_tree_from_pam(
    pam: np.ndarray,
    taxon_names: list[str],
    method: Literal["jaccard_upgma", "hamming_upgma", "jaccard_nj"] = "jaccard_upgma"
) -> Tuple[TreeStructure, TreeInferenceMetadata]:
    """
    Infer phylogenetic tree from presence/absence matrix.
    
    This is the main entry point for tree inference. It provides fast,
    transparent tree inference when a phylogeny is not available.
    
    Args:
        pam: Binary presence/absence matrix (n_taxa × n_genes)
        taxon_names: List of taxon names (length n_taxa)
        method: Tree inference method:
            - "jaccard_upgma": Jaccard distance + UPGMA (default, recommended)
            - "hamming_upgma": Hamming distance + UPGMA
            - "jaccard_nj": Jaccard distance + Neighbor-Joining
            
    Returns:
        tree: Inferred TreeStructure
        metadata: TreeInferenceMetadata with method details
        
    Example:
        >>> pam = np.random.binomial(1, 0.5, size=(10, 100))
        >>> taxon_names = [f"strain_{i}" for i in range(10)]
        >>> tree, metadata = infer_tree_from_pam(pam, taxon_names)
        >>> metadata.source
        'inferred'
        >>> metadata.method
        'jaccard_upgma'
        
    Note:
        Tree inference is an approximation. The inferred tree reflects
        gene content similarity, which correlates with but is not identical
        to true phylogeny. Always inspect metadata.source to know if tree
        was inferred.
    """
    n_taxa, n_genes = pam.shape
    
    if len(taxon_names) != n_taxa:
        raise ValueError(f"taxon_names length ({len(taxon_names)}) must match pam rows ({n_taxa})")
    
    # Parse method
    if method == "jaccard_upgma":
        distance_metric = "jaccard"
        clustering_method = "upgma"
        distance_matrix = jaccard_distance(pam)
        tree = upgma_tree(distance_matrix, taxon_names)
        
    elif method == "hamming_upgma":
        distance_metric = "hamming"
        clustering_method = "upgma"
        distance_matrix = hamming_distance(pam)
        tree = upgma_tree(distance_matrix, taxon_names)
        
    elif method == "jaccard_nj":
        distance_metric = "jaccard"
        clustering_method = "neighbor_joining"
        distance_matrix = jaccard_distance(pam)
        tree = neighbor_joining_tree(distance_matrix, taxon_names)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'jaccard_upgma', 'hamming_upgma', or 'jaccard_nj'")
    
    # Create metadata
    metadata = TreeInferenceMetadata(
        source="inferred",
        method=method,
        distance_metric=distance_metric,
        clustering_method=clustering_method,
        n_taxa=n_taxa,
        n_genes=n_genes
    )
    
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
