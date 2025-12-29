#!/usr/bin/env python
"""Debug why rate estimation is off by ~10x."""

import sys
import numpy as np
from pathlib import Path
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from persiste.core.trees import TreeStructure
from persiste.core.pruning import FelsensteinPruning, SimpleBinaryTransitionProvider, ArrayTipConditionalProvider
from persiste.plugins.genecontent.inference.gene_inference import GeneContentData, GeneContentModel
from persiste.plugins.genecontent.constraints.gene_constraint import NullConstraint

# Create tree
newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
tree = TreeStructure.from_newick(newick, backend="simple")
taxon_names = ["A", "B", "C", "D"]

# True rates
true_gain = 2.0
true_loss = 3.0

print("Rate Estimation Debug")
print("=" * 80)
print(f"True gain rate: {true_gain}")
print(f"True loss rate: {true_loss}")
print()

# Simulate ONE family with known rates
rng = np.random.default_rng(42)

pi_0 = true_loss / (true_gain + true_loss)
pi_1 = true_gain / (true_gain + true_loss)

Q = np.array([
    [-true_gain, true_gain],
    [true_loss, -true_loss]
])

# Simulate
root_state = rng.choice([0, 1], p=[pi_0, pi_1])
node_states = {tree.root_index: root_state}

for child_idx in range(tree.n_nodes):
    parent_idx = tree.parent_indices[child_idx]
    if parent_idx >= 0:
        parent_state = node_states[parent_idx]
        t = tree.branch_lengths[child_idx]
        P = expm(Q * t)
        child_state = rng.choice([0, 1], p=P[parent_state, :])
        node_states[child_idx] = child_state

# Extract tip states
presence_data = np.array([[node_states[tip_idx]] for tip_idx in tree.tip_indices], dtype=int)

print("Simulated data (single family):")
print("-" * 80)
for i, tip_idx in enumerate(tree.tip_indices):
    tip_name = tree.nodes[tip_idx].name
    state = presence_data[i, 0]
    print(f"  {tip_name}: {state}")
print()

# Now compute likelihood at TRUE parameters
print("Test 1: Likelihood at TRUE parameters")
print("-" * 80)

pruning = FelsensteinPruning(tree, n_states=2, use_jax=False)
transition_provider = SimpleBinaryTransitionProvider(
    gain_rate=true_gain,
    loss_rate=true_loss,
)
tip_provider = ArrayTipConditionalProvider(
    data=presence_data,
    taxon_names=taxon_names,
    n_states=2,
)

result = pruning.compute_likelihood(
    transition_provider=transition_provider,
    tip_provider=tip_provider,
    n_sites=1,
)

print(f"Log-likelihood at true rates: {result.log_likelihood:.6f}")
print()

# Test 2: Likelihood at SCALED parameters
print("Test 2: Likelihood at different rate scales")
print("-" * 80)
print("Scale | Gain  | Loss  | Log-likelihood")
print("-" * 50)

for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
    transition_provider = SimpleBinaryTransitionProvider(
        gain_rate=true_gain * scale,
        loss_rate=true_loss * scale,
    )
    
    result = pruning.compute_likelihood(
        transition_provider=transition_provider,
        tip_provider=tip_provider,
        n_sites=1,
    )
    
    print(f"{scale:>5.1f} | {true_gain*scale:>5.2f} | {true_loss*scale:>5.2f} | {result.log_likelihood:>14.6f}")

print()

# Test 3: Use the GeneContentModel to fit
print("Test 3: Fit using GeneContentModel")
print("-" * 80)

gene_data = GeneContentData(
    tree=tree,
    presence_matrix=presence_data,
    taxon_names=taxon_names,
    family_names=["fam1"],
)

model = GeneContentModel(data=gene_data, constraint=NullConstraint())

# Try different initial parameters
print("Initial params | Estimated gain | Estimated loss | Log-likelihood")
print("-" * 70)

from persiste.core.tree_inference import TreeMLEOptimizer

for init_log_gain in [-2, -1, 0, np.log(true_gain)]:
    for init_log_loss in [-2, -1, 0, np.log(true_loss)]:
        initial_params = {
            'log_gain': init_log_gain,
            'log_loss': init_log_loss,
        }
        
        optimizer = TreeMLEOptimizer(model)
        result = optimizer.fit(initial_params=initial_params)
        
        est_gain = np.exp(result.parameters['log_gain'])
        est_loss = np.exp(result.parameters['log_loss'])
        
        print(f"({init_log_gain:>5.2f}, {init_log_loss:>5.2f}) | "
              f"{est_gain:>14.4f} | {est_loss:>14.4f} | {result.log_likelihood:>14.6f}")

print()
print("=" * 80)
print("If all optimizations converge to similar rates, the model is working.")
print("If rates are consistently ~10x too low, there's a systematic issue.")
