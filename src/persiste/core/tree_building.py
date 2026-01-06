"""
Distance-based tree building utilities for PERSISTE.

Provides fast, transparent tree inference from distance matrices.
Used by plugins when phylogeny is not available.

Methods:
- UPGMA (Unweighted Pair Group Method with Arithmetic Mean)
- Neighbor-Joining (TODO)

Distance metrics:
- Jaccard distance (for presence/absence data)
- Hamming distance (for binary data)
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
    n_features: Optional[int] = None  # genes, sites, etc.


def jaccard_distance(binary_matrix: np.ndarray, use_rust: bool = True) -> np.ndarray:
    """
    Compute Jaccard distance matrix from binary matrix.
    
    Uses Rust acceleration by default (6-18x faster) with automatic
    fallback to scipy if Rust is not available.
    
    Jaccard distance = 1 - Jaccard similarity
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    
    Args:
        binary_matrix: Binary matrix (n_taxa × n_features)
        use_rust: Use Rust acceleration if available (default: True)
        
    Returns:
        Distance matrix (n_taxa × n_taxa)
        
    Example:
        >>> matrix = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
        >>> D = jaccard_distance(matrix)
        >>> D.shape
        (3, 3)
    """
    if use_rust and RUST_AVAILABLE:
        # Use Rust acceleration (6-18x faster)
        matrix_u8 = binary_matrix.astype(np.uint8)
        return persiste_rust.compute_jaccard_distance(matrix_u8)
    else:
        # Fallback to scipy
        distances = pdist(binary_matrix, metric='jaccard')
        return squareform(distances)


def hamming_distance(binary_matrix: np.ndarray, use_rust: bool = True) -> np.ndarray:
    """
    Compute Hamming distance matrix from binary matrix.
    
    Uses Rust acceleration by default (6-15x faster) with automatic
    fallback to scipy if Rust is not available.
    
    Hamming distance = proportion of positions that differ
    
    Args:
        binary_matrix: Binary matrix (n_taxa × n_features)
        use_rust: Use Rust acceleration if available (default: True)
        
    Returns:
        Distance matrix (n_taxa × n_taxa)
    """
    if use_rust and RUST_AVAILABLE:
        # Use Rust acceleration (6-15x faster)
        matrix_u8 = binary_matrix.astype(np.uint8)
        return persiste_rust.compute_hamming_distance(matrix_u8)
    else:
        # Fallback to scipy
        distances = pdist(binary_matrix, metric='hamming')
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
    
    def to_newick(node, parent_height: float | None = None):
        """Convert scipy tree to Newick format.

        SciPy's ClusterNode.dist is the linkage distance at which the cluster
        was formed. For UPGMA (average linkage), the ultrametric height is
        dist/2, and the branch length from a child to its parent is
        parent_height - child_height.
        """

        height = float(node.dist) / 2.0
        branch_len = None if parent_height is None else max(0.0, parent_height - height)

        if node.is_leaf():
            name = taxon_names[node.id]
            if branch_len is None:
                return f"{name}"
            return f"{name}:{branch_len:.6f}"

        left = to_newick(node.left, height)
        right = to_newick(node.right, height)
        inner = f"({left},{right})"
        if branch_len is None:
            return inner
        return f"{inner}:{branch_len:.6f}"
    
    newick = to_newick(scipy_tree) + ";"
    
    # Parse Newick string into TreeStructure
    tree = TreeStructure.from_newick(newick, backend="simple")

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


def infer_tree_from_binary_matrix(
    binary_matrix: np.ndarray,
    taxon_names: list[str],
    method: Literal["jaccard_upgma", "hamming_upgma", "jaccard_nj"] = "jaccard_upgma",
    *,
    verbose: bool = False,
) -> Tuple[TreeStructure, TreeInferenceMetadata]:
    """
    Infer phylogenetic tree from binary matrix (PAM, binned CN, etc.).
    
    This is the main entry point for tree inference. It provides fast,
    transparent tree inference when a phylogeny is not available.
    
    Args:
        binary_matrix: Binary matrix (n_taxa × n_features)
            For PAM: (n_taxa × n_genes)
            For CN: (n_taxa × n_families) after binarization
        taxon_names: List of taxon names (length n_taxa)
        method: Tree inference method:
            - "jaccard_upgma": Jaccard distance + UPGMA (default, recommended)
            - "hamming_upgma": Hamming distance + UPGMA
            - "jaccard_nj": Jaccard distance + Neighbor-Joining
            
    Returns:
        tree: Inferred TreeStructure
        metadata: TreeInferenceMetadata with method details
        
    Example:
        >>> matrix = np.random.binomial(1, 0.5, size=(10, 100))
        >>> taxon_names = [f"strain_{i}" for i in range(10)]
        >>> tree, metadata = infer_tree_from_binary_matrix(matrix, taxon_names)
        >>> metadata.source
        'inferred'
        >>> metadata.method
        'jaccard_upgma'
        
    Note:
        Tree inference is an approximation. The inferred tree reflects
        similarity in the binary matrix, which correlates with but is not
        identical to true phylogeny. Always inspect metadata.source to know
        if tree was inferred.
    """

    def summarize_distance_matrix(distance_matrix: np.ndarray) -> None:
        # Report off-diagonal summary (diagonal is always 0)
        if distance_matrix.size == 0:
            print("  Distance matrix: empty")
            return

        triu = np.triu_indices(distance_matrix.shape[0], k=1)
        vals = distance_matrix[triu]
        if vals.size == 0:
            print("  Distance matrix: no off-diagonal entries")
            return

        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            print("  Distance matrix: all off-diagonal values non-finite")
            return

        zero_frac = float(np.mean(finite == 0.0))
        print(
            "  Distance matrix off-diagonal: "
            f"min={float(np.min(finite)):.6g}, "
            f"median={float(np.median(finite)):.6g}, "
            f"max={float(np.max(finite)):.6g}, "
            f"zeros={zero_frac:.1%}"
        )

    def summarize_branch_lengths(tree: TreeStructure) -> None:
        bl = tree.branch_lengths
        if bl.size == 0:
            print("  Branch lengths: empty")
            return

        finite = bl[np.isfinite(bl)]
        if finite.size == 0:
            print("  Branch lengths: all non-finite")
            return

        zero_frac = float(np.mean(finite == 0.0))
        positive = finite[finite > 0]
        min_pos = float(np.min(positive)) if positive.size else float("nan")
        print(
            "  Branch lengths: "
            f"min_pos={min_pos:.6g}, "
            f"median={float(np.median(finite)):.6g}, "
            f"max={float(np.max(finite)):.6g}, "
            f"zeros={zero_frac:.1%}, "
            f"sum={float(np.sum(finite)):.6g}"
        )
    n_taxa, n_features = binary_matrix.shape
    
    if len(taxon_names) != n_taxa:
        raise ValueError(f"taxon_names length ({len(taxon_names)}) must match matrix rows ({n_taxa})")
    
    # Parse method
    if method == "jaccard_upgma":
        distance_metric = "jaccard"
        clustering_method = "upgma"
        distance_matrix = jaccard_distance(binary_matrix)
        if verbose:
            summarize_distance_matrix(distance_matrix)
        tree = upgma_tree(distance_matrix, taxon_names)
        
    elif method == "hamming_upgma":
        distance_metric = "hamming"
        clustering_method = "upgma"
        distance_matrix = hamming_distance(binary_matrix)
        if verbose:
            summarize_distance_matrix(distance_matrix)
        tree = upgma_tree(distance_matrix, taxon_names)
        
    elif method == "jaccard_nj":
        distance_metric = "jaccard"
        clustering_method = "neighbor_joining"
        distance_matrix = jaccard_distance(binary_matrix)
        if verbose:
            summarize_distance_matrix(distance_matrix)
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
        n_features=n_features
    )

    if verbose:
        summarize_branch_lengths(tree)
    
    return tree, metadata
