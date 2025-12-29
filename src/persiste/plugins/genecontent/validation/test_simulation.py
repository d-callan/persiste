#!/usr/bin/env python
"""Test that simulation produces expected statistics."""

import sys
import numpy as np
from pathlib import Path
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from persiste.core.trees import TreeStructure

# Create simple tree
newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
tree = TreeStructure.from_newick(newick, backend="simple")

# Simulation parameters
gain_rate = 2.0
loss_rate = 3.0
n_replicates = 10000

# Equilibrium frequencies
pi_0 = loss_rate / (gain_rate + loss_rate)
pi_1 = gain_rate / (gain_rate + loss_rate)

print("Simulation test:")
print("=" * 80)
print(f"Gain rate: {gain_rate}")
print(f"Loss rate: {loss_rate}")
print(f"Equilibrium P(present): {pi_1:.4f}")
print(f"Replicates: {n_replicates}")
print()

# Build rate matrix
Q = np.array([
    [-gain_rate, gain_rate],
    [loss_rate, -loss_rate]
])

# Simulate many replicates
rng = np.random.default_rng(42)
tip_states = np.zeros((n_replicates, tree.n_tips), dtype=int)

for rep in range(n_replicates):
    # Sample root state
    root_state = rng.choice([0, 1], p=[pi_0, pi_1])
    
    # Simulate down tree
    node_states = {tree.root_index: root_state}
    
    for child_idx in range(tree.n_nodes):
        parent_idx = tree.parent_indices[child_idx]
        if parent_idx >= 0:  # Not root
            parent_state = node_states[parent_idx]
            t = tree.branch_lengths[child_idx]
            P = expm(Q * t)
            child_state = rng.choice([0, 1], p=P[parent_state, :])
            node_states[child_idx] = child_state
    
    # Extract tip states
    for tip_idx_pos, tip_idx in enumerate(tree.tip_indices):
        tip_states[rep, tip_idx_pos] = node_states[tip_idx]

# Analyze statistics
print("Simulated statistics:")
print("-" * 80)
print(f"Mean presence per tip: {tip_states.mean():.4f}")
print(f"Expected (equilibrium): {pi_1:.4f}")
print()

# Check pairwise correlations (should be positive due to shared ancestry)
for i in range(tree.n_tips):
    for j in range(i+1, tree.n_tips):
        corr = np.corrcoef(tip_states[:, i], tip_states[:, j])[0, 1]
        tip_i_name = tree.nodes[tree.tip_indices[i]].name
        tip_j_name = tree.nodes[tree.tip_indices[j]].name
        print(f"Correlation {tip_i_name}-{tip_j_name}: {corr:.4f}")

print()
print("If mean presence ≈ equilibrium and correlations > 0, simulation is working!")
