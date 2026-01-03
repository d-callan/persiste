"""Tests for site patterns compression."""

import sys
import numpy as np


def _align_to_tree(order: list[str], tree_tip_names: list[str], alignment: np.ndarray) -> np.ndarray:
    """Reorder alignment rows to follow the TreeStructure tip ordering."""
    name_to_idx = {name: idx for idx, name in enumerate(order)}
    row_indices = [name_to_idx[name] for name in tree_tip_names]
    return alignment[row_indices]


def test_site_patterns_basic():
    """Test basic site patterns compression."""
    print("Testing basic site patterns compression...")
    
    from persiste.plugins.phylo.data.site_patterns import SitePatterns
    
    # Create alignment with repeated patterns
    # Sites 0,1,2 are identical: [0,0,0,0]
    # Sites 3,4 are identical: [1,2,1,2]
    # Site 5 is unique: [3,3,3,3]
    alignment = np.array([
        [0, 0, 0, 1, 1, 3],  # Taxon A
        [0, 0, 0, 2, 2, 3],  # Taxon B
        [0, 0, 0, 1, 1, 3],  # Taxon C
        [0, 0, 0, 2, 2, 3],  # Taxon D
    ])
    
    patterns = SitePatterns(alignment)
    
    print(f"  Original sites: {patterns.n_sites}")
    print(f"  Unique patterns: {patterns.n_unique}")
    print(f"  Compression ratio: {patterns.compression_ratio:.2f}x")
    
    assert patterns.n_sites == 6, "Should have 6 original sites"
    assert patterns.n_unique == 3, "Should have 3 unique patterns"
    assert patterns.compression_ratio == 2.0, "Should have 2x compression"
    
    # Check weights
    assert patterns.pattern_weights.sum() == 6, "Weights should sum to original site count"
    assert 3 in patterns.pattern_weights, "Should have pattern with weight 3"
    assert 2 in patterns.pattern_weights, "Should have pattern with weight 2"
    assert 1 in patterns.pattern_weights, "Should have pattern with weight 1"
    
    print("  ✓ Site patterns compression works")
    print(f"  ✓ {patterns}")


def test_site_patterns_all_unique():
    """Test compression when all sites are unique."""
    print("Testing all unique sites...")
    
    from persiste.plugins.phylo.data.site_patterns import SitePatterns
    
    # All sites different
    alignment = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ])
    
    patterns = SitePatterns(alignment)
    
    print(f"  Original sites: {patterns.n_sites}")
    print(f"  Unique patterns: {patterns.n_unique}")
    print(f"  Compression ratio: {patterns.compression_ratio:.2f}x")
    
    assert patterns.n_unique == 4, "All sites should be unique"
    assert patterns.compression_ratio == 1.0, "No compression possible"
    assert all(patterns.pattern_weights == 1), "All weights should be 1"
    
    print("  ✓ No compression when all sites unique")


def test_site_patterns_all_identical():
    """Test compression when all sites are identical."""
    print("Testing all identical sites...")
    
    from persiste.plugins.phylo.data.site_patterns import SitePatterns
    
    # All sites identical
    alignment = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
    ])
    
    patterns = SitePatterns(alignment)
    
    print(f"  Original sites: {patterns.n_sites}")
    print(f"  Unique patterns: {patterns.n_unique}")
    print(f"  Compression ratio: {patterns.compression_ratio:.2f}x")
    
    assert patterns.n_unique == 1, "Should have only 1 unique pattern"
    assert patterns.compression_ratio == 5.0, "Should have 5x compression"
    assert patterns.pattern_weights[0] == 5, "Single pattern should have weight 5"
    
    print("  ✓ Maximum compression when all sites identical")


def test_site_patterns_decompression():
    """Test decompression of site likelihoods."""
    print("Testing site likelihood decompression...")
    
    from persiste.plugins.phylo.data.site_patterns import SitePatterns
    
    # Create alignment with patterns
    alignment = np.array([
        [0, 0, 1, 1, 2],
        [0, 0, 1, 1, 2],
        [0, 0, 1, 1, 2],
        [0, 0, 1, 1, 2],
    ])
    
    patterns = SitePatterns(alignment)
    
    # Simulate pattern likelihoods
    pattern_likelihoods = np.array([-1.0, -2.0, -3.0])  # 3 unique patterns
    
    # Decompress to original sites
    site_likelihoods = patterns.decompress_site_likelihoods(pattern_likelihoods)
    
    print(f"  Pattern likelihoods: {pattern_likelihoods}")
    print(f"  Site likelihoods: {site_likelihoods}")
    
    assert len(site_likelihoods) == 5, "Should have 5 site likelihoods"
    
    # Sites 0,1 should have same likelihood (pattern 0)
    assert site_likelihoods[0] == site_likelihoods[1]
    
    # Sites 2,3 should have same likelihood (pattern 1)
    assert site_likelihoods[2] == site_likelihoods[3]
    
    print("  ✓ Decompression works correctly")


def test_site_patterns_with_likelihood():
    """Test site patterns with actual likelihood calculation."""
    print("Testing site patterns with likelihood...")
    
    from persiste.core.trees import TreeStructure
    from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    from persiste.plugins.phylo.data.site_patterns import SitePatterns
    
    # Setup
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick)
    
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)
    
    # Create alignment with repeated patterns
    atg_idx = codon_space.index("ATG")
    ttc_idx = codon_space.index("TTC")
    
    alignment = np.array([
        [atg_idx, atg_idx, atg_idx, ttc_idx, ttc_idx],  # A
        [atg_idx, atg_idx, atg_idx, ttc_idx, ttc_idx],  # B
        [atg_idx, atg_idx, atg_idx, ttc_idx, ttc_idx],  # C
        [atg_idx, atg_idx, atg_idx, ttc_idx, ttc_idx],  # D
    ])
    alignment = _align_to_tree(["A", "B", "C", "D"], tree.tip_names, alignment)
    
    # Compress patterns
    patterns = SitePatterns(alignment)
    
    print(f"  Original sites: {patterns.n_sites}")
    print(f"  Unique patterns: {patterns.n_unique}")
    print(f"  Weights: {patterns.pattern_weights}")
    
    # Compute likelihood with compressed alignment
    obs_model_compressed = PhyloCTMCObservationModel(
        graph,
        tree,
        patterns.compressed_alignment,
        site_weights=patterns.weights,
    )
    
    # Compute likelihood with original alignment
    obs_model_original = PhyloCTMCObservationModel(graph, tree, alignment)
    
    from persiste.core.data import ObservedTransitions
    dummy_data = ObservedTransitions(counts={}, exposure=1.0)
    
    log_lik_compressed = obs_model_compressed.log_likelihood(dummy_data, baseline, graph)
    log_lik_original = obs_model_original.log_likelihood(dummy_data, baseline, graph)
    
    print(f"  Log-likelihood (compressed): {log_lik_compressed:.6f}")
    print(f"  Log-likelihood (original):   {log_lik_original:.6f}")
    print(f"  Difference: {abs(log_lik_compressed - log_lik_original):.10f}")
    
    # Should be identical (within numerical precision)
    assert abs(log_lik_compressed - log_lik_original) < 1e-10, \
        "Compressed and original likelihoods should match"
    
    print("  ✓ Site patterns give identical likelihood")
    print("  ✓ Compression is lossless")


def test_site_patterns_summary():
    """Test site patterns summary statistics."""
    print("Testing site patterns summary...")
    
    from persiste.plugins.phylo.data.site_patterns import SitePatterns
    
    # Create alignment
    alignment = np.array([
        [0, 0, 0, 1, 1, 2, 3, 4],
        [0, 0, 0, 1, 1, 2, 3, 4],
        [0, 0, 0, 1, 1, 2, 3, 4],
        [0, 0, 0, 1, 1, 2, 3, 4],
    ])
    
    patterns = SitePatterns(alignment)
    summary = patterns.summary()
    
    print(f"  Summary: {summary}")
    
    assert summary['n_sites_original'] == 8
    assert summary['n_patterns_unique'] == 5
    assert summary['compression_ratio'] == 8.0 / 5.0
    assert summary['most_common_pattern_count'] == 3
    
    print("  ✓ Summary statistics correct")


def main():
    """Run all site patterns tests."""
    print("=" * 60)
    print("PERSISTE Phylo Plugin - Site Patterns Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_site_patterns_basic,
        test_site_patterns_all_unique,
        test_site_patterns_all_identical,
        test_site_patterns_decompression,
        test_site_patterns_with_likelihood,
        test_site_patterns_summary,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ Site patterns compression validated!")
        print("✓ Ready for production use")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
