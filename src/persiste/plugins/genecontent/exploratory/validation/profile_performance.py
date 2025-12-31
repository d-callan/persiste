#!/usr/bin/env python
"""
Profile genecontent performance to identify bottlenecks.
"""

import cProfile
import pstats
import io
import sys
import numpy as np
from pathlib import Path
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from persiste.core.trees import TreeStructure
from persiste.plugins.genecontent.inference.gene_inference import GeneContentData, GeneContentInference
from persiste.plugins.genecontent.constraints.gene_constraint import RetentionBiasConstraint


def create_test_data(n_tips=50, n_families=500):
    """Create test data for profiling."""
    # Create balanced tree
    newick_parts = []
    for i in range(n_tips):
        newick_parts.append(f"tip{i}:1.0")
    
    # Build balanced binary tree
    while len(newick_parts) > 1:
        new_parts = []
        for i in range(0, len(newick_parts), 2):
            if i + 1 < len(newick_parts):
                new_parts.append(f"({newick_parts[i]},{newick_parts[i+1]}):1.0")
            else:
                new_parts.append(newick_parts[i])
        newick_parts = new_parts
    
    newick = newick_parts[0] + ";"
    tree = TreeStructure.from_newick(newick, backend="simple")
    
    # Simulate data
    rng = np.random.default_rng(42)
    gain_rate = 1.5
    loss_rate = 2.0
    
    pi_0 = loss_rate / (gain_rate + loss_rate)
    pi_1 = gain_rate / (gain_rate + loss_rate)
    
    Q = np.array([
        [-gain_rate, gain_rate],
        [loss_rate, -loss_rate]
    ])
    
    presence_matrix = np.zeros((n_tips, n_families), dtype=np.int8)
    
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
    
    taxon_names = [f"tip{i}" for i in range(n_tips)]
    family_names = [f"fam{i}" for i in range(n_families)]
    
    return GeneContentData(
        tree=tree,
        presence_matrix=presence_matrix,
        taxon_names=taxon_names,
        family_names=family_names,
    )


def profile_global_rates():
    """Profile global rate estimation."""
    print("=" * 80)
    print("PROFILING: Global Rate Estimation")
    print("=" * 80)
    print()
    
    data = create_test_data(n_tips=50, n_families=500)
    print(f"Dataset: {data.n_taxa} taxa, {data.n_families} families")
    print()
    
    inference = GeneContentInference(data)
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = inference.fit_null()
    
    profiler.disable()
    
    # Print results
    gain = np.exp(result.parameters['log_gain'])
    loss = np.exp(result.parameters['log_loss'])
    print(f"Gain rate: {gain:.4f}")
    print(f"Loss rate: {loss:.4f}")
    print(f"Log-likelihood: {result.log_likelihood:.2f}")
    print(f"Function evals: {result.n_function_evals}")
    print()
    
    # Print profile stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    print("Profile Results (Top 30 by cumulative time):")
    print(s.getvalue())
    
    return profiler


def profile_retention_test():
    """Profile retention test."""
    print("\n" + "=" * 80)
    print("PROFILING: Retention Test")
    print("=" * 80)
    print()
    
    data = create_test_data(n_tips=50, n_families=500)
    print(f"Dataset: {data.n_taxa} taxa, {data.n_families} families")
    print()
    
    inference = GeneContentInference(data)
    
    # Test first 50 families
    test_families = set(data.family_names[:50])
    constraint = RetentionBiasConstraint(retained_families=test_families)
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    null_result, alt_result, lrt_result = inference.fit_and_test(constraint, alpha=0.05)
    
    profiler.disable()
    
    # Print results
    print(f"Null LL: {null_result.log_likelihood:.2f}")
    print(f"Alt LL: {alt_result.log_likelihood:.2f}")
    print(f"p-value: {lrt_result.pvalue:.4f}")
    print()
    
    # Print profile stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    print("Profile Results (Top 30 by cumulative time):")
    print(s.getvalue())
    
    return profiler


def profile_simulation():
    """Profile just the simulation code."""
    print("\n" + "=" * 80)
    print("PROFILING: Data Simulation")
    print("=" * 80)
    print()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    data = create_test_data(n_tips=100, n_families=1000)
    
    profiler.disable()
    
    print(f"Simulated: {data.n_taxa} taxa, {data.n_families} families")
    print()
    
    # Print profile stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("Profile Results (Top 20 by cumulative time):")
    print(s.getvalue())
    
    return profiler


if __name__ == "__main__":
    print("GeneContent Performance Profiling")
    print("=" * 80)
    print()
    
    # Profile each component
    profile_simulation()
    profile_global_rates()
    profile_retention_test()
    
    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print()
    print("Key findings will be in the cumulative time columns.")
    print("Look for:")
    print("  - log_likelihood() - main bottleneck")
    print("  - compute_likelihood() - pruning algorithm")
    print("  - expm() - transition matrix computation")
    print("  - minimize() - optimizer overhead")
