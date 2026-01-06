from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np

from persiste.core.tree_building import (
    TreeInferenceMetadata,
    infer_tree_from_binary_matrix,
)
from persiste.core.trees import TreeStructure, load_tree


def validate_tree_with_binary_matrix(
    tree: TreeStructure,
    binary_matrix: np.ndarray,
) -> dict:
    """Basic validation to ensure a tree matches a binary matrix."""
    results = {"valid": True, "errors": [], "warnings": []}

    n_taxa = binary_matrix.shape[0]
    if tree.n_tips != n_taxa:
        results["valid"] = False
        results["errors"].append(
            f"Tree has {tree.n_tips} tips but matrix has {n_taxa} taxa"
        )

    branch_lengths = tree.branch_lengths
    if not np.all(np.isfinite(branch_lengths)):
        results["valid"] = False
        results["errors"].append("Tree has non-finite branch lengths")

    if np.any(branch_lengths < 0):
        results["valid"] = False
        results["errors"].append("Tree has negative branch lengths")

    positive_branches = branch_lengths[branch_lengths > 0]
    if positive_branches.size:
        min_branch = float(np.min(positive_branches))
        if min_branch < 1e-10:
            results["warnings"].append(
                f"Very short branch lengths detected (min: {min_branch:.2e})"
            )

    if branch_lengths.size:
        max_branch = float(np.max(branch_lengths))
        if max_branch > 10:
            results["warnings"].append(
                f"Very long branch detected (max: {max_branch:.2f})"
            )

    return results


def prepare_tree_from_binary_matrix(
    binary_matrix: np.ndarray,
    taxon_names: list[str],
    tree: Union[None, str, Path, TreeStructure] = None,
    tree_method: str = "jaccard_upgma",
    *,
    verbose: bool = False,
    validator: Optional[Callable[[TreeStructure, np.ndarray], dict]] = validate_tree_with_binary_matrix,
) -> Tuple[TreeStructure, TreeInferenceMetadata]:
    """
    Shared helper for loading or inferring trees from binary matrices.
    """
    tree_obj: TreeStructure
    metadata: TreeInferenceMetadata

    if tree is None:
        if verbose:
            print(f"\nInferring tree from binary matrix using {tree_method}...")
        tree_obj, metadata = infer_tree_from_binary_matrix(
            binary_matrix=binary_matrix,
            taxon_names=taxon_names,
            method=tree_method,
            verbose=verbose,
        )
    elif isinstance(tree, (str, Path)):
        if verbose:
            print(f"\nLoading tree from: {tree}")
        tree_obj = load_tree(str(tree))
        metadata = TreeInferenceMetadata(source="provided")
    elif isinstance(tree, TreeStructure):
        tree_obj = tree
        metadata = TreeInferenceMetadata(source="provided")
    else:
        raise TypeError(
            "tree must be None, a path to Newick file, or TreeStructure instance"
        )

    if validator is not None:
        validation = validator(tree_obj, binary_matrix)
        if not validation["valid"]:
            raise ValueError(
                f"Tree validation failed: {', '.join(validation['errors'])}"
            )
        if verbose and validation["warnings"]:
            print("  Warnings:")
            for warning in validation["warnings"]:
                print(f"    ⚠ {warning}")

    if verbose:
        print(f"  Tree: {tree_obj.n_tips} tips, {tree_obj.n_nodes} total nodes")

    return tree_obj, metadata
