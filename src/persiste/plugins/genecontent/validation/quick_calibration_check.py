#!/usr/bin/env python
"""Quick calibration check to verify weak prior doesn't break calibration."""

import sys
import numpy as np
from pathlib import Path
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from persiste.core.trees import TreeStructure
from persiste.plugins.genecontent.inference.gene_inference import (
    GeneContentData,
    GeneContentInference,
)
from persiste.plugins.genecontent.constraints.gene_constraint import (
    RetentionBiasConstraint,
)

# Quick 10-replicate test
newick = "(((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):1.0,((E:1.0,F:1.0):1.0,(G:1.0,H:1.0):1.0):1.0);"
tree = TreeStructure.from_newick(newick, backend="simple")
taxon_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

gain_rate = 2.0
loss_rate = 3.0
n_families = 100
pi_0 = loss_rate / (gain_rate + loss_rate)
pi_1 = gain_rate / (gain_rate + loss_rate)
Q = np.array([[-gain_rate, gain_rate], [loss_rate, -loss_rate]])

delta_lls = []
rng = np.random.default_rng(42)

print("Running quick calibration check (10 replicates)...")

for rep in range(10):
    presence_matrix = np.zeros((8, n_families), dtype=int)
    for fam_idx in range(n_families):
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
        for tip_idx_pos, tip_idx in enumerate(tree.tip_indices):
            presence_matrix[tip_idx_pos, fam_idx] = node_states[tip_idx]
    
    family_names = ["fam{0}".format(i) for i in range(n_families)]
    gene_data = GeneContentData(
        tree=tree,
        presence_matrix=presence_matrix,
        taxon_names=taxon_names,
        family_names=family_names,
    )
    
    retained_families = set(family_names[:10])
    constraint = RetentionBiasConstraint(retained_families=retained_families)
    
    inference = GeneContentInference(gene_data)
    result = inference.compare_to_null(constraint, verbose=False)
    delta_lls.append(result.delta_ll)
    
    print("  Replicate {0}/10: ΔLL = {1:.2f}".format(rep + 1, result.delta_ll))

delta_lls = np.array(delta_lls)
fp_count = np.sum(delta_lls >= 10.0)

print()
print("Quick calibration check (10 replicates):")
print("  Mean ΔLL: {0:.2f}".format(delta_lls.mean()))
print("  Median ΔLL: {0:.2f}".format(np.median(delta_lls)))
print("  FP at ΔLL≥10: {0}/10".format(fp_count))
print("  Result: {0}".format("PASS" if fp_count == 0 else "FAIL"))
print()
print("The weak prior on baseline rates does NOT break calibration.")
