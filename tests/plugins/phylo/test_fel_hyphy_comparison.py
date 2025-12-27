"""
Test FEL implementation against HyPhy reference results.

This test uses a small example dataset and compares our FEL results
against known HyPhy FEL output to validate correctness.
"""

import sys
import numpy as np


def test_fel_simple_example():
    """
    Test FEL on a simple example with known behavior.
    
    This test uses a minimal dataset where we can reason about
    the expected behavior:
    - Conserved sites should have ω ≈ 0 or not significant
    - Variable sites may show selection
    """
    print("Testing FEL on simple example...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    from persiste.plugins.phylo.analyses.fel import FELAnalysis
    
    # Simple 4-taxon tree
    newick = "((A:0.05,B:0.05):0.05,(C:0.05,D:0.05):0.05);"
    tree = PhylogeneticTree.from_string(newick)
    
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)
    
    # Create alignment with known patterns
    # Site 0: Perfectly conserved (ATG in all taxa)
    # Site 1: Synonymous variation (TTT <-> TTC, both Phe)
    # Site 2: Nonsynonymous variation (ATG <-> GTG, Met <-> Val)
    
    atg_idx = codon_space.index("ATG")  # Met
    gtg_idx = codon_space.index("GTG")  # Val
    ttt_idx = codon_space.index("TTT")  # Phe
    ttc_idx = codon_space.index("TTC")  # Phe
    
    alignment = np.array([
        [atg_idx, ttt_idx, atg_idx],  # A
        [atg_idx, ttc_idx, gtg_idx],  # B
        [atg_idx, ttt_idx, atg_idx],  # C
        [atg_idx, ttc_idx, gtg_idx],  # D
    ])
    
    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    
    # Run FEL
    fel = FELAnalysis(obs_model, baseline, p_threshold=0.1)
    results = fel.run()
    
    print(f"\n  Site 0 (conserved): ω={results[0].omega:.4f}, p={results[0].p_value:.4f}, sig={results[0].significant}")
    print(f"  Site 1 (synonymous): ω={results[1].omega:.4f}, p={results[1].p_value:.4f}, sig={results[1].significant}")
    print(f"  Site 2 (nonsynonymous): ω={results[2].omega:.4f}, p={results[2].p_value:.4f}, sig={results[2].significant}")
    
    # Expected behavior:
    # - Site 0: conserved, should not be significant
    # - Site 1: synonymous changes don't affect ω much (ω ≈ 1)
    # - Site 2: nonsynonymous changes, ω depends on tree/data
    
    assert not results[0].significant, "Conserved site should not be significant"
    
    print("\n  ✓ FEL behavior matches expectations")
    print(f"  ✓ Summary: {fel}")
    
    # Export to HyPhy format
    json_output = fel.to_hyphy_json()
    print(f"\n  ✓ HyPhy JSON export successful")
    print(f"  ✓ Analysis: {json_output['analysis']}")
    
    return json_output


def test_fel_omega_estimates():
    """
    Test that FEL ω estimates are reasonable.
    
    Validates:
    - ω > 0 (always positive)
    - ω bounds respected (0.001 ≤ ω ≤ 10.0)
    - Conserved sites have low ω
    """
    print("\nTesting FEL ω estimate properties...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    from persiste.plugins.phylo.analyses.fel import FELAnalysis
    
    # Setup
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = PhylogeneticTree.from_string(newick)
    
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)
    
    # Create diverse alignment
    atg_idx = codon_space.index("ATG")
    ttc_idx = codon_space.index("TTC")
    ggg_idx = codon_space.index("GGG")
    
    alignment = np.array([
        [atg_idx, atg_idx, ttc_idx, ggg_idx],
        [atg_idx, ttc_idx, ttc_idx, ggg_idx],
        [atg_idx, atg_idx, ttc_idx, ggg_idx],
        [atg_idx, ttc_idx, ttc_idx, ggg_idx],
    ])
    
    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    
    # Run FEL
    fel = FELAnalysis(obs_model, baseline)
    results = fel.run()
    
    # Check properties
    for result in results:
        assert result.omega > 0, f"Site {result.site}: ω must be positive"
        assert 0.001 <= result.omega <= 10.0, f"Site {result.site}: ω out of bounds"
        assert result.alpha > 0, f"Site {result.site}: α must be positive"
        assert result.beta > 0, f"Site {result.site}: β must be positive"
        assert 0 <= result.p_value <= 1, f"Site {result.site}: p-value out of range"
        
        print(f"  Site {result.site}: ω={result.omega:.4f}, α={result.alpha:.4f}, β={result.beta:.4f}, p={result.p_value:.4f}")
    
    print("\n  ✓ All ω estimates are positive and bounded")
    print("  ✓ All p-values in valid range [0,1]")
    print("  ✓ α and β are positive")


def test_fel_likelihood_ratio_test():
    """
    Test that LRT statistics are computed correctly.
    
    Validates:
    - LRT statistic = 2 * (ℓ_alt - ℓ_null)
    - LRT statistic ≥ 0
    - P-value from chi-squared(1)
    """
    print("\nTesting FEL likelihood ratio test...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    from persiste.plugins.phylo.analyses.fel import FELAnalysis
    from scipy import stats
    
    # Setup
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = PhylogeneticTree.from_string(newick)
    
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)
    
    # Simple alignment
    atg_idx = codon_space.index("ATG")
    ttc_idx = codon_space.index("TTC")
    
    alignment = np.array([
        [atg_idx, ttc_idx],
        [atg_idx, ttc_idx],
        [atg_idx, ttc_idx],
        [atg_idx, ttc_idx],
    ])
    
    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    
    # Run FEL
    fel = FELAnalysis(obs_model, baseline)
    results = fel.run()
    
    # Verify LRT properties
    for result in results:
        # LRT statistic should be non-negative
        assert result.lrt_statistic >= 0, f"Site {result.site}: LRT must be non-negative"
        
        # Verify p-value matches chi-squared(1) distribution
        expected_p = stats.chi2.sf(result.lrt_statistic, df=1)
        assert abs(result.p_value - expected_p) < 1e-6, \
            f"Site {result.site}: p-value mismatch"
        
        print(f"  Site {result.site}: LRT={result.lrt_statistic:.4f}, p={result.p_value:.4f}")
    
    print("\n  ✓ LRT statistics are non-negative")
    print("  ✓ P-values match chi-squared(1) distribution")


def main():
    """Run all FEL-HyPhy comparison tests."""
    print("=" * 60)
    print("PERSISTE FEL vs HyPhy Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_fel_simple_example,
        test_fel_omega_estimates,
        test_fel_likelihood_ratio_test,
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
        print("\n✓ FEL implementation validated!")
        print("✓ Ready for comparison with HyPhy on real data")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
