"""Debug Rust vs NumPy differences."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from persiste.core.trees import TreeStructure
from persiste.core.pruning import FelsensteinPruning, SimpleBinaryTransitionProvider, ArrayTipConditionalProvider
from persiste_rust import compute_likelihoods_parallel

# Simple test case
newick = "(A:1.0,B:1.0);"
tree = TreeStructure.from_newick(newick, backend="simple")

print("Tree structure:")
print(f"  n_nodes: {tree.n_nodes}")
print(f"  n_tips: {tree.n_tips}")
print(f"  parent_indices: {tree.parent_indices}")
print(f"  branch_lengths: {tree.branch_lengths}")
print(f"  root_index: {tree.root_index}")

# Simple data: both tips present
presence_matrix = np.array([[1], [1]], dtype=np.int8)
gain_rate = 1.5
loss_rate = 2.0

print("\nData:")
print(f"  presence_matrix:\n{presence_matrix}")
print(f"  gain_rate: {gain_rate}")
print(f"  loss_rate: {loss_rate}")

# NumPy computation
print("\n" + "="*50)
print("NumPy computation:")
pruning = FelsensteinPruning(tree, n_states=2, use_jax=False)
transition_provider = SimpleBinaryTransitionProvider(gain_rate, loss_rate, use_cache=False)
taxon_names = ["A", "B"]
tip_provider = ArrayTipConditionalProvider(presence_matrix, taxon_names, n_states=2)

result = pruning.compute_likelihood(transition_provider, tip_provider, n_sites=1)
print(f"  log_likelihood: {result.log_likelihood}")

# Rust computation
print("\n" + "="*50)
print("Rust computation:")
ll_rust = compute_likelihoods_parallel(
    tree.parent_indices.astype(np.int32),
    tree.branch_lengths.astype(np.float64),
    presence_matrix,
    np.array([gain_rate], dtype=np.float64),
    np.array([loss_rate], dtype=np.float64),
    tree.n_tips,
)
print(f"  log_likelihood: {ll_rust[0]}")

print("\n" + "="*50)
print(f"Difference: {abs(result.log_likelihood - ll_rust[0]):.6e}")
