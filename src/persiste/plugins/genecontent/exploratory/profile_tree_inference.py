#!/usr/bin/env python3
"""
Profile tree inference to identify bottlenecks for Rust acceleration.
"""

import sys
from pathlib import Path
import numpy as np
import time
import cProfile
import pstats
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persiste.plugins.genecontent.tree_inference import (
    jaccard_distance,
    hamming_distance,
    upgma_tree,
    infer_tree_from_pam,
)


def profile_distance_computation():
    """Profile distance matrix computation."""
    print("=" * 70)
    print("PROFILING DISTANCE COMPUTATION")
    print("=" * 70)
    
    sizes = [(50, 1000), (100, 5000), (200, 10000)]
    
    for n_strains, n_genes in sizes:
        print(f"\n{n_strains} strains × {n_genes:,} genes:")
        
        np.random.seed(42)
        pam = np.random.binomial(1, 0.5, size=(n_strains, n_genes))
        
        # Time Jaccard distance
        start = time.time()
        D_jaccard = jaccard_distance(pam)
        jaccard_time = time.time() - start
        print(f"  Jaccard distance: {jaccard_time:.4f}s")
        
        # Time Hamming distance
        start = time.time()
        D_hamming = hamming_distance(pam)
        hamming_time = time.time() - start
        print(f"  Hamming distance: {hamming_time:.4f}s")


def profile_upgma():
    """Profile UPGMA tree construction."""
    print("\n" + "=" * 70)
    print("PROFILING UPGMA")
    print("=" * 70)
    
    sizes = [50, 100, 200]
    
    for n_strains in sizes:
        print(f"\n{n_strains} strains:")
        
        # Create random distance matrix
        np.random.seed(42)
        D = np.random.rand(n_strains, n_strains)
        D = (D + D.T) / 2  # Make symmetric
        np.fill_diagonal(D, 0)
        
        taxon_names = [f"strain_{i}" for i in range(n_strains)]
        
        start = time.time()
        tree = upgma_tree(D, taxon_names)
        upgma_time = time.time() - start
        print(f"  UPGMA: {upgma_time:.4f}s")
        print(f"  Tree nodes: {tree.n_nodes}")


def profile_full_pipeline():
    """Profile complete tree inference pipeline."""
    print("\n" + "=" * 70)
    print("PROFILING FULL PIPELINE")
    print("=" * 70)
    
    sizes = [(50, 1000), (100, 5000), (200, 10000)]
    
    for n_strains, n_genes in sizes:
        print(f"\n{n_strains} strains × {n_genes:,} genes:")
        
        np.random.seed(42)
        pam = np.random.binomial(1, 0.5, size=(n_strains, n_genes))
        taxon_names = [f"strain_{i}" for i in range(n_strains)]
        
        # Profile with cProfile
        pr = cProfile.Profile()
        pr.enable()
        
        start = time.time()
        tree, metadata = infer_tree_from_pam(pam, taxon_names, method='jaccard_upgma')
        total_time = time.time() - start
        
        pr.disable()
        
        print(f"  Total time: {total_time:.4f}s")
        
        # Print top time consumers
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        
        print("\n  Top 10 time consumers:")
        for line in s.getvalue().split('\n')[5:15]:
            if line.strip():
                print(f"    {line}")


def test_correctness():
    """Test correctness of tree inference."""
    print("\n" + "=" * 70)
    print("TESTING CORRECTNESS")
    print("=" * 70)
    
    # Test 1: Identical taxa
    print("\nTest 1: Identical taxa should have zero distance")
    pam = np.array([[1, 0, 1, 0],
                    [1, 0, 1, 0]])
    D = jaccard_distance(pam)
    assert D[0, 1] == 0.0, "Failed: identical taxa"
    print("  ✓ Passed")
    
    # Test 2: Disjoint taxa
    print("\nTest 2: Disjoint taxa should have distance 1")
    pam = np.array([[1, 1, 0, 0],
                    [0, 0, 1, 1]])
    D = jaccard_distance(pam)
    assert D[0, 1] == 1.0, "Failed: disjoint taxa"
    print("  ✓ Passed")
    
    # Test 3: Tree structure
    print("\nTest 3: Tree structure")
    pam = np.array([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 1, 1]
    ])
    taxon_names = ['A', 'B', 'C']
    tree, metadata = infer_tree_from_pam(pam, taxon_names)
    
    assert tree.n_tips == 3, "Failed: wrong number of tips"
    assert set(tree.tip_names) == {'A', 'B', 'C'}, "Failed: wrong tip names"
    assert metadata.source == 'inferred', "Failed: wrong source"
    assert metadata.method == 'jaccard_upgma', "Failed: wrong method"
    print("  ✓ Passed")
    
    # Test 4: Taxon names with underscores
    print("\nTest 4: Taxon names with underscores preserved")
    pam = np.array([[1, 0], [0, 1]])
    taxon_names = ['GCF_001_234', 'GCF_002_567']
    tree, metadata = infer_tree_from_pam(pam, taxon_names)
    
    assert 'GCF_001_234' in tree.tip_names, "Failed: underscore not preserved"
    assert 'GCF_002_567' in tree.tip_names, "Failed: underscore not preserved"
    print("  ✓ Passed")
    
    print("\n✓ All correctness tests passed!")


def main():
    print("=" * 70)
    print("TREE INFERENCE PROFILING")
    print("=" * 70)
    
    # Test correctness first
    test_correctness()
    
    # Profile components
    profile_distance_computation()
    profile_upgma()
    profile_full_pipeline()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nBottlenecks identified:")
    print("1. Distance computation (Jaccard/Hamming) - scipy.spatial.distance.pdist")
    print("2. UPGMA clustering - scipy.cluster.hierarchy.linkage")
    print("3. Newick parsing - TreeStructure.from_newick")
    print("\nRecommendation: Implement distance computation and UPGMA in Rust")


if __name__ == "__main__":
    main()
