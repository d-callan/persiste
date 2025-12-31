#!/usr/bin/env python
"""
Demo of standard gene content analyses.

Shows the opinionated, easy-to-use API for common biological questions.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from persiste.core.trees import TreeStructure
from persiste.plugins.genecontent.inference.gene_inference import GeneContentData
from persiste.plugins.genecontent.analyses import GeneContentAnalysis


def demo_standard_analyses():
    """Demonstrate all standard analyses."""
    
    print("=" * 80)
    print("DEMO: Standard Gene Content Analyses")
    print("=" * 80)
    print()
    print("These are opinionated, easy-to-use analysis recipes.")
    print("Each answers a clear biological question with automatic:")
    print("  - Null comparison")
    print("  - Interpretation guidance")
    print("  - Diagnostic checks")
    print()
    
    # Create example data
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
    
    # Initialize analysis engine
    analysis = GeneContentAnalysis(gene_data)
    
    # =========================================================================
    # Analysis 1: Global Gain/Loss Estimation
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 1: Global Gain/Loss Estimation")
    print("=" * 80)
    print()
    print("This is the default analysis - just estimates baseline rates.")
    print("Useful even without any constraints.")
    print()
    
    result1 = analysis.global_rates(verbose=True)
    
    # =========================================================================
    # Analysis 2: Targeted Retention Test (RELAX-style)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Targeted Retention Test")
    print("=" * 80)
    print()
    print("This is the core use case - test specific gene families.")
    print("Like RELAX, but for gene content.")
    print()
    
    # Test first 10 families
    test_families = family_names[:10]
    result2 = analysis.retention_test(families=test_families, verbose=True)
    
    # =========================================================================
    # Analysis 3: Branch-Set Gain/Loss Shift
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Branch-Set Gain/Loss Shift")
    print("=" * 80)
    print()
    print("Test whether gene dynamics differ on specific lineages.")
    print()
    
    # Test first clade (A, B, C, D)
    result3 = analysis.branch_shift(foreground_taxa=['A', 'B', 'C', 'D'], verbose=True)
    
    # =========================================================================
    # Analysis 4: Host/Metadata Association Test
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Host/Metadata Association Test")
    print("=" * 80)
    print()
    print("Test whether gene retention is associated with a trait.")
    print()
    
    # Create example trait (first 4 taxa = host1, last 4 = host2)
    trait_values = {
        'A': 0, 'B': 0, 'C': 0, 'D': 0,
        'E': 1, 'F': 1, 'G': 1, 'H': 1,
    }
    
    result4 = analysis.association_test(
        trait_name='host_type',
        trait_values=trait_values,
        verbose=True
    )
    
    # =========================================================================
    # Analysis 5: Exploratory Family Screening
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 5: Exploratory Family Screening")
    print("=" * 80)
    print()
    print("CAUTION: This is exploratory, not confirmatory.")
    print("Use for hypothesis generation only.")
    print()
    print("NOTE: Skipping this in demo (would take ~10 minutes for 100 families)")
    print()
    
    # This would screen all families
    # result5 = analysis.exploratory_screening(verbose=True, top_n=20)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Standard analyses provide:")
    print("  ✓ Clear biological questions")
    print("  ✓ Opinionated defaults (no flag hell)")
    print("  ✓ Automatic null comparison")
    print("  ✓ Interpretable results")
    print("  ✓ Built-in diagnostics")
    print()
    print("Usage is simple:")
    print()
    print("  analysis = GeneContentAnalysis(data)")
    print("  result = analysis.retention_test(families=['OG0001', 'OG0002'])")
    print()
    print("This is the RELAX/BUSTED/aBSREL approach - named analyses, not flags.")
    print()


if __name__ == "__main__":
    demo_standard_analyses()
