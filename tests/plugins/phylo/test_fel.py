"""Tests for FEL (Fixed Effects Likelihood) analysis."""

import sys
import numpy as np


def test_fel_basic():
    """Test basic FEL analysis on simple data."""
    print("Testing basic FEL analysis...")
    
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
    
    # Alignment: 3 sites
    # Site 0: conserved (all ATG) - expect ω ≈ 1 or low
    # Site 1: variable nonsynonymous - expect ω > 0
    # Site 2: conserved (all TTT) - expect ω ≈ 1 or low
    atg_idx = codon_space.index("ATG")
    ttc_idx = codon_space.index("TTC")  # Phe (different from Met)
    ttt_idx = codon_space.index("TTT")  # Phe
    
    alignment = np.array([
        [atg_idx, atg_idx, ttt_idx],  # A
        [atg_idx, ttc_idx, ttt_idx],  # B
        [atg_idx, atg_idx, ttt_idx],  # C
        [atg_idx, ttc_idx, ttt_idx],  # D
    ])
    
    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    
    # Run FEL
    fel = FELAnalysis(obs_model, baseline, p_threshold=0.1)
    results = fel.run()
    
    assert len(results) == 3, f"Expected 3 site results, got {len(results)}"
    
    print(f"  Site 0: ω={results[0].omega:.4f}, p={results[0].p_value:.4f}")
    print(f"  Site 1: ω={results[1].omega:.4f}, p={results[1].p_value:.4f}")
    print(f"  Site 2: ω={results[2].omega:.4f}, p={results[2].p_value:.4f}")
    
    # Site 0 and 2 are conserved, should have low ω or not significant
    # Site 1 has variation, may have different ω
    
    print("  ✓ FEL analysis runs successfully")
    print(f"  ✓ {fel}")


def test_fel_conserved_site():
    """Test FEL on perfectly conserved site."""
    print("Testing FEL on conserved site...")
    
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
    
    # Alignment: single conserved site (all ATG)
    atg_idx = codon_space.index("ATG")
    alignment = np.array([
        [atg_idx],
        [atg_idx],
        [atg_idx],
        [atg_idx],
    ])
    
    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    
    # Run FEL
    fel = FELAnalysis(obs_model, baseline)
    results = fel.run()
    
    result = results[0]
    
    print(f"  ω={result.omega:.4f}")
    print(f"  p-value={result.p_value:.4f}")
    print(f"  Significant: {result.significant}")
    
    # Conserved site should not show significant selection
    # (though ω estimate may vary due to limited data)
    
    print("  ✓ Conserved site analysis works")


def test_fel_summary():
    """Test FEL summary statistics."""
    print("Testing FEL summary...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    from persiste.plugins.phylo.analyses.fel import FELAnalysis
    
    # Setup with multiple sites
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = PhylogeneticTree.from_string(newick)
    
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)
    
    # Create alignment with 5 sites
    atg_idx = codon_space.index("ATG")
    alignment = np.array([
        [atg_idx, atg_idx, atg_idx, atg_idx, atg_idx],
        [atg_idx, atg_idx, atg_idx, atg_idx, atg_idx],
        [atg_idx, atg_idx, atg_idx, atg_idx, atg_idx],
        [atg_idx, atg_idx, atg_idx, atg_idx, atg_idx],
    ])
    
    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    
    # Run FEL
    fel = FELAnalysis(obs_model, baseline)
    fel.run()
    
    # Get summary
    summary = fel.summary()
    
    print(f"  Sites analyzed: {summary['n_sites']}")
    print(f"  Significant: {summary['n_significant']}")
    print(f"  Mean ω: {summary['mean_omega']:.4f}")
    print(f"  Median ω: {summary['median_omega']:.4f}")
    
    assert summary['n_sites'] == 5, "Should analyze 5 sites"
    assert 'mean_omega' in summary, "Summary should include mean_omega"
    assert 'n_significant' in summary, "Summary should include n_significant"
    
    print("  ✓ Summary statistics work")


def test_fel_hyphy_json():
    """Test FEL HyPhy JSON export."""
    print("Testing FEL HyPhy JSON export...")
    
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
    
    # Simple alignment
    atg_idx = codon_space.index("ATG")
    alignment = np.array([
        [atg_idx, atg_idx],
        [atg_idx, atg_idx],
        [atg_idx, atg_idx],
        [atg_idx, atg_idx],
    ])
    
    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    
    # Run FEL
    fel = FELAnalysis(obs_model, baseline)
    fel.run()
    
    # Export to HyPhy JSON
    json_output = fel.to_hyphy_json()
    
    assert 'analysis' in json_output, "JSON should have 'analysis' field"
    assert 'MLE' in json_output, "JSON should have 'MLE' field"
    assert 'summary' in json_output, "JSON should have 'summary' field"
    
    # Check MLE content structure
    mle_content = json_output['MLE']['content']
    assert '0' in mle_content, "Should have site 0 results"
    assert 'omega' in mle_content['0'], "Site results should have omega"
    assert 'alpha' in mle_content['0'], "Site results should have alpha"
    assert 'beta' in mle_content['0'], "Site results should have beta"
    assert 'p' in mle_content['0'], "Site results should have p-value"
    
    print("  ✓ HyPhy JSON export works")
    print(f"  ✓ Analysis: {json_output['analysis']}")
    print(f"  ✓ Sites: {json_output['input']['n_sites']}")


def test_fel_site_classification():
    """Test FEL site classification (positive/purifying)."""
    print("Testing FEL site classification...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    from persiste.plugins.phylo.analyses.fel import FELAnalysis, FELSiteResult
    
    # Create mock results to test classification
    # (In real analysis, these would come from fit_site_omega)
    
    results = [
        FELSiteResult(
            site=0, alpha=1.0, beta=2.0, omega=2.0,
            log_likelihood=-10.0, lrt_statistic=5.0,
            p_value=0.025, significant=True
        ),
        FELSiteResult(
            site=1, alpha=1.0, beta=0.1, omega=0.1,
            log_likelihood=-10.0, lrt_statistic=5.0,
            p_value=0.025, significant=True
        ),
        FELSiteResult(
            site=2, alpha=1.0, beta=1.0, omega=1.0,
            log_likelihood=-10.0, lrt_statistic=0.1,
            p_value=0.75, significant=False
        ),
    ]
    
    # Mock FEL object
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = PhylogeneticTree.from_string(newick)
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)
    atg_idx = codon_space.index("ATG")
    alignment = np.array([[atg_idx], [atg_idx], [atg_idx], [atg_idx]])
    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    
    fel = FELAnalysis(obs_model, baseline)
    fel.site_results = results
    
    # Test classification
    significant = fel.get_significant_sites()
    positive = fel.get_positively_selected_sites()
    negative = fel.get_negatively_selected_sites()
    
    assert len(significant) == 2, "Should have 2 significant sites"
    assert len(positive) == 1, "Should have 1 positively selected site"
    assert len(negative) == 1, "Should have 1 purifying selection site"
    
    assert positive[0].site == 0, "Site 0 should be positive (ω=2.0)"
    assert negative[0].site == 1, "Site 1 should be purifying (ω=0.1)"
    
    print("  ✓ Site classification works")
    print(f"  ✓ Positive selection: {len(positive)} sites")
    print(f"  ✓ Purifying selection: {len(negative)} sites")


def main():
    """Run all FEL tests."""
    print("=" * 60)
    print("PERSISTE Phylo Plugin - FEL Analysis Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_fel_basic,
        test_fel_conserved_site,
        test_fel_summary,
        test_fel_hyphy_json,
        test_fel_site_classification,
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
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
