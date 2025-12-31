#!/usr/bin/env python
"""Test tree structure to understand parent_indices."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from persiste.core.trees import TreeStructure

# Create simple tree
newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
tree = TreeStructure.from_newick(newick, backend="simple")

print("Tree structure analysis:")
print("=" * 80)
print(f"n_nodes: {tree.n_nodes}")
print(f"n_tips: {tree.n_tips}")
print(f"root_index: {tree.root_index}")
print()

print("Node details:")
print("-" * 80)
for i, node in enumerate(tree.nodes):
    print(f"Node {i}: name={node.name}, parent_id={node.parent_id}, "
          f"children={node.children_ids}, branch_len={node.branch_length:.2f}, "
          f"is_tip={node.is_tip}")
print()

print("Arrays:")
print("-" * 80)
print(f"tip_indices: {tree.tip_indices}")
print(f"internal_indices: {tree.internal_indices}")
print(f"postorder: {tree.postorder}")
print(f"parent_indices: {tree.parent_indices}")
print(f"branch_lengths: {tree.branch_lengths}")
print()

print("Children array:")
print("-" * 80)
print(tree.children_array)
print()

print("Edge list (parent -> child):")
print("-" * 80)
for child_idx in range(tree.n_nodes):
    parent_idx = tree.parent_indices[child_idx]
    if parent_idx >= 0:
        branch_len = tree.branch_lengths[child_idx]
        child_name = tree.nodes[child_idx].name or f"internal_{child_idx}"
        parent_name = tree.nodes[parent_idx].name or f"internal_{parent_idx}"
        print(f"  {parent_name} ({parent_idx}) -> {child_name} ({child_idx}), len={branch_len:.2f}")
