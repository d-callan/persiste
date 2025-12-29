#!/usr/bin/env python
"""Demo showing data sufficiency warning with small dataset."""

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


def demo_small_dataset():
    """Demo with small dataset to show sufficiency warning."""
    
    print("=" * 80)
    print("DEMO: Data Sufficiency Warning (Small Dataset)")
    print("=" * 80)
    print()
    
    # Small tree (4 tips)
    newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    taxon_names = ["A", "B", "C", "D"]
    
    # Simulate small dataset (30 families)
    rng = np.random.default_rng(42)
    gain_rate = 2.0
    loss_rate = 3.0
    n_families = 30
    
    pi_0 = loss_rate / (gain_rate + loss_rate)
    pi_1 = gain_rate / (gain_rate + loss_rate)
    
    Q = np.array([
        [-gain_rate, gain_rate],
        [loss_rate, -loss_rate]
    ])
    
    presence_matrix = np.zeros((4, n_families), dtype=int)
    
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
    
    print("Dataset: {0} families, {1} tips (SMALL)".format(n_families, len(taxon_names)))
    print()
    
    # Get baseline diagnostics - should show warning
    print("=" * 80)
    print("Baseline Diagnostics (with sufficiency warning)")
    print("=" * 80)
    print()
    
    inference = GeneContentInference(gene_data)
    diagnostics = inference.get_baseline_diagnostics(verbose=True)
    
    # Test constraint - should show ΔLL prominently
    print("=" * 80)
    print("Model Comparison (ΔLL-first reporting)")
    print("=" * 80)
    print()
    
    retained_families = set(family_names[:3])
    constraint = RetentionBiasConstraint(retained_families=retained_families)
    
    result = inference.compare_to_null(constraint, verbose=True)
    
    print("=" * 80)
    print("KEY POINTS")
    print("=" * 80)
    print()
    print("1. Data sufficiency warning appears BEFORE inference")
    print("   → Users are warned about high variance regime")
    print("   → No hard stop, just honesty")
    print()
    print("2. ΔLL is reported FIRST and PROMINENTLY")
    print("   → ΔLL = {0:.2f} → {1}".format(
        result.delta_ll,
        "Insufficient evidence" if result.delta_ll < 2 else "Evidence present"
    ))
    print()
    print("3. θ̂ is shown but de-emphasized")
    print("   → retention_strength = {0:.4f} (do not interpret alone)".format(
        result.alt_result.parameters['retention_strength']
    ))
    print()
    print("This matches how HyPhy users think:")
    print("  - Report ΔLL, not θ̂")
    print("  - Warn about data sufficiency")
    print("  - Provide honest guidance")
    print()


if __name__ == "__main__":
    demo_small_dataset()
