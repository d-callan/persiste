#!/usr/bin/env python
"""
Demo of the new user-friendly API.

Shows how the new features make correct use easy and incorrect use hard.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from persiste.core.trees import TreeStructure
from persiste.plugins.genecontent.inference.gene_inference import (
    GeneContentData,
    GeneContentInference,
)
from persiste.plugins.genecontent.constraints.gene_constraint import (
    RetentionBiasConstraint,
)


def demo_new_api():
    """Demonstrate the new user-friendly API."""
    
    print("=" * 80)
    print("DEMO: New User-Friendly API")
    print("=" * 80)
    print()
    
    # Load some example data (using simulated data for demo)
    from scipy.linalg import expm
    
    newick = "(((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):1.0,((E:1.0,F:1.0):1.0,(G:1.0,H:1.0):1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    taxon_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
    
    # Simulate data
    rng = np.random.default_rng(42)
    gain_rate = 2.0
    loss_rate = 3.0
    n_families = 100
    
    pi_0 = loss_rate / (gain_rate + loss_rate)
    pi_1 = gain_rate / (gain_rate + loss_rate)
    
    Q = np.array([
        [-gain_rate, gain_rate],
        [loss_rate, -loss_rate]
    ])
    
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
    
    print("Loaded data: {0} families, {1} tips".format(n_families, len(taxon_names)))
    print()
    
    # Step 1: Get baseline diagnostics (RECOMMENDED FIRST STEP)
    print("=" * 80)
    print("STEP 1: Baseline Diagnostics")
    print("=" * 80)
    print()
    print("Always check baseline model first to catch nonsense estimates.")
    print()
    
    inference = GeneContentInference(gene_data)
    diagnostics = inference.get_baseline_diagnostics(verbose=True)
    
    # Step 2: Compare to null (RECOMMENDED WAY TO TEST CONSTRAINTS)
    print("=" * 80)
    print("STEP 2: Test Constraint (compare_to_null)")
    print("=" * 80)
    print()
    print("This is the RECOMMENDED way to test constraints.")
    print("It automatically:")
    print("  - Fits both null and alternative models")
    print("  - Performs likelihood ratio test")
    print("  - Provides interpretation guidance")
    print("  - Prevents over-interpretation of θ̂")
    print()
    
    # Test retention bias on first 10 families
    retained_families = set(family_names[:10])
    constraint = RetentionBiasConstraint(retained_families=retained_families)
    
    result = inference.compare_to_null(constraint, verbose=True)
    
    # Step 3: Interpret results
    print("=" * 80)
    print("STEP 3: Interpretation")
    print("=" * 80)
    print()
    
    if result.evidence_strength == 'none':
        print("✓ No evidence for retention bias")
        print("  → Null model is sufficient")
        print("  → Do not report constraint effect")
    elif result.evidence_strength == 'weak':
        print("⚠ Weak evidence for retention bias")
        print("  → Interpret with extreme caution")
        print("  → Consider as exploratory only")
    elif result.evidence_strength == 'moderate':
        print("⚠ Moderate evidence for retention bias")
        print("  → Check biological plausibility")
        print("  → Consider additional validation")
    else:  # strong
        print("✓ Strong evidence for retention bias")
        print("  → Effect is likely real")
        print("  → Safe to report with confidence")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("The new API makes it easy to:")
    print("  1. Check baseline model quality (get_baseline_diagnostics)")
    print("  2. Test constraints properly (compare_to_null)")
    print("  3. Interpret results correctly (automatic guidance)")
    print()
    print("This prevents common mistakes:")
    print("  ✗ Reporting θ̂ without comparing to null")
    print("  ✗ Over-interpreting weak evidence")
    print("  ✗ Missing nonsense baseline estimates")
    print()
    print("The model is production-ready with these safeguards in place.")
    print()


if __name__ == "__main__":
    demo_new_api()
