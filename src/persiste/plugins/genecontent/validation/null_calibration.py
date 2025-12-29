#!/usr/bin/env python
"""
Replicate-based null calibration test.

Simulates data under θ=0, fits model, measures false positive rate.
This is the final validation test before freezing v1.
"""

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


def run_null_calibration(
    n_replicates: int = 100,
    n_families: int = 100,
    n_tips: int = 8,
    delta_ll_threshold: float = 10.0,
    seed: int = 42,
):
    """
    Run null calibration test.
    
    Simulates data under θ=0, fits model, measures false positive rate.
    
    Args:
        n_replicates: Number of simulation replicates
        n_families: Number of gene families per replicate
        n_tips: Number of tips in tree
        delta_ll_threshold: ΔLL threshold for calling "significant"
        seed: Random seed
        
    Returns:
        Dict with calibration results
    """
    print("=" * 80)
    print("NULL CALIBRATION TEST")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  Replicates: {0}".format(n_replicates))
    print("  Families per replicate: {0}".format(n_families))
    print("  Tips: {0}".format(n_tips))
    print("  ΔLL threshold: {0}".format(delta_ll_threshold))
    print()
    
    # Create tree (8 tips)
    newick = "(((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):1.0,((E:1.0,F:1.0):1.0,(G:1.0,H:1.0):1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    taxon_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
    
    # Simulation parameters (reasonable rates)
    gain_rate = 2.0
    loss_rate = 3.0
    
    # Equilibrium frequencies
    pi_0 = loss_rate / (gain_rate + loss_rate)
    pi_1 = gain_rate / (gain_rate + loss_rate)
    
    # Build rate matrix
    Q = np.array([
        [-gain_rate, gain_rate],
        [loss_rate, -loss_rate]
    ])
    
    # Storage for results
    delta_lls = []
    p_values = []
    false_positives = 0
    
    rng = np.random.default_rng(seed)
    
    print("Running {0} replicates...".format(n_replicates))
    print()
    
    for rep in range(n_replicates):
        if (rep + 1) % 10 == 0:
            print("  Replicate {0}/{1}...".format(rep + 1, n_replicates))
        
        # Simulate data with NO retention bias (θ = 0)
        presence_matrix = np.zeros((n_tips, n_families), dtype=int)
        
        for fam_idx in range(n_families):
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
                presence_matrix[tip_idx_pos, fam_idx] = node_states[tip_idx]
        
        family_names = ["fam{0}".format(i) for i in range(n_families)]
        
        # Create data
        gene_data = GeneContentData(
            tree=tree,
            presence_matrix=presence_matrix,
            taxon_names=taxon_names,
            family_names=family_names,
        )
        
        # Test for retention bias on random subset (10% of families)
        n_retained = max(1, n_families // 10)
        retained_families = set(family_names[:n_retained])
        constraint = RetentionBiasConstraint(retained_families=retained_families)
        
        # Fit and compare
        inference = GeneContentInference(gene_data)
        result = inference.compare_to_null(constraint, verbose=False)
        
        delta_lls.append(result.delta_ll)
        p_values.append(result.lrt_result.pvalue)
        
        if result.delta_ll >= delta_ll_threshold:
            false_positives += 1
    
    # Compute statistics
    delta_lls = np.array(delta_lls)
    p_values = np.array(p_values)
    
    false_positive_rate = false_positives / n_replicates
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    print("ΔLL distribution:")
    print("  Mean:   {0:.2f}".format(delta_lls.mean()))
    print("  Median: {0:.2f}".format(np.median(delta_lls)))
    print("  Std:    {0:.2f}".format(delta_lls.std()))
    print("  Min:    {0:.2f}".format(delta_lls.min()))
    print("  Max:    {0:.2f}".format(delta_lls.max()))
    print()
    
    print("False positive rate (ΔLL >= {0}):".format(delta_ll_threshold))
    print("  FP count: {0}/{1}".format(false_positives, n_replicates))
    print("  FP rate:  {0:.1%}".format(false_positive_rate))
    print()
    
    # Check calibration
    target_fp_rate = 0.10  # 10% is acceptable
    
    if false_positive_rate <= target_fp_rate:
        print("✓ CALIBRATION PASSED")
        print("  False positive rate ({0:.1%}) is within acceptable range (<= {1:.1%})".format(
            false_positive_rate, target_fp_rate
        ))
        print("  The model is well-calibrated for ΔLL threshold of {0}".format(delta_ll_threshold))
    else:
        print("✗ CALIBRATION FAILED")
        print("  False positive rate ({0:.1%}) exceeds acceptable range ({1:.1%})".format(
            false_positive_rate, target_fp_rate
        ))
        print("  Consider:")
        print("    - Increasing ΔLL threshold")
        print("    - Adding stronger priors")
        print("    - Investigating systematic bias")
    
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    
    print("This test simulates data under the null hypothesis (θ = 0) and")
    print("measures how often the model incorrectly infers a constraint effect.")
    print()
    print("A well-calibrated model should have:")
    print("  - FP rate < 5-10% at ΔLL threshold of 10")
    print("  - ΔLL distribution centered near 0")
    print("  - No systematic bias toward positive or negative ΔLL")
    print()
    
    if false_positive_rate <= 0.05:
        print("Your model is EXCELLENT - ready for production use.")
    elif false_positive_rate <= 0.10:
        print("Your model is GOOD - acceptable for production use with caution.")
    else:
        print("Your model needs IMPROVEMENT - use with extreme caution.")
    
    print()
    
    return {
        'n_replicates': n_replicates,
        'delta_lls': delta_lls,
        'p_values': p_values,
        'false_positive_rate': false_positive_rate,
        'false_positives': false_positives,
        'threshold': delta_ll_threshold,
        'passed': false_positive_rate <= target_fp_rate,
    }


if __name__ == "__main__":
    # Run calibration test
    results = run_null_calibration(
        n_replicates=100,
        n_families=100,
        n_tips=8,
        delta_ll_threshold=10.0,
        seed=42,
    )
    
    # Save results
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    np.savez(
        output_dir / "null_calibration_results.npz",
        delta_lls=results['delta_lls'],
        p_values=results['p_values'],
        false_positive_rate=results['false_positive_rate'],
        false_positives=results['false_positives'],
        n_replicates=results['n_replicates'],
        threshold=results['threshold'],
        passed=results['passed'],
    )
    
    print("Results saved to: {0}".format(output_dir / "null_calibration_results.npz"))
