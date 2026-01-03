"""Tests for FEL (Fixed Effects Likelihood) analysis."""

import sys

import numpy as np

from persiste.core.trees import TreeStructure
from persiste.plugins.phylo.analyses.fel import FELAnalysis, FELSiteResult
from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
from persiste.plugins.phylo.states.codons import CodonStateSpace
from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph


def _align_alignment_to_tree(
    tree: TreeStructure,
    taxa_order: list[str],
    alignment: np.ndarray,
) -> np.ndarray:
    """Reorder alignment rows to match the tree's internal tip ordering."""
    order_map = {name: idx for idx, name in enumerate(taxa_order)}
    indices = [order_map[name] for name in tree.tip_names]
    return alignment[indices]


def test_fel_basic() -> None:
    """Test basic FEL analysis on simple data."""
    print("Testing basic FEL analysis...")

    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick)

    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)

    atg_idx = codon_space.index("ATG")
    ttc_idx = codon_space.index("TTC")
    ttt_idx = codon_space.index("TTT")

    alignment = np.array(
        [
            [atg_idx, atg_idx, ttt_idx],
            [atg_idx, ttc_idx, ttt_idx],
            [atg_idx, atg_idx, ttt_idx],
            [atg_idx, ttc_idx, ttt_idx],
        ]
    )
    alignment = _align_alignment_to_tree(tree, ["A", "B", "C", "D"], alignment)

    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    fel = FELAnalysis(obs_model, baseline, p_threshold=0.1)
    results = fel.run()

    assert len(results) == 3, f"Expected 3 site results, got {len(results)}"
    print(f"  Site 0: ω={results[0].omega:.4f}, p={results[0].p_value:.4f}")
    print(f"  Site 1: ω={results[1].omega:.4f}, p={results[1].p_value:.4f}")
    print(f"  Site 2: ω={results[2].omega:.4f}, p={results[2].p_value:.4f}")
    print("  ✓ FEL analysis runs successfully")
    print(f"  ✓ {fel}")


def test_fel_conserved_site() -> None:
    """Test FEL on perfectly conserved site."""
    print("Testing FEL on conserved site...")

    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick)

    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)

    atg_idx = codon_space.index("ATG")
    alignment = np.array([[atg_idx], [atg_idx], [atg_idx], [atg_idx]])
    alignment = _align_alignment_to_tree(tree, ["A", "B", "C", "D"], alignment)

    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    fel = FELAnalysis(obs_model, baseline)
    results = fel.run()

    result = results[0]
    print(f"  ω={result.omega:.4f}")
    print(f"  p-value={result.p_value:.4f}")
    print(f"  Significant: {result.significant}")
    print("  ✓ Conserved site analysis works")


def test_fel_summary() -> None:
    """Test FEL summary statistics."""
    print("Testing FEL summary...")

    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick)

    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)

    atg_idx = codon_space.index("ATG")
    alignment = np.array(
        [
            [atg_idx, atg_idx, atg_idx, atg_idx, atg_idx],
            [atg_idx, atg_idx, atg_idx, atg_idx, atg_idx],
            [atg_idx, atg_idx, atg_idx, atg_idx, atg_idx],
            [atg_idx, atg_idx, atg_idx, atg_idx, atg_idx],
        ]
    )
    alignment = _align_alignment_to_tree(tree, ["A", "B", "C", "D"], alignment)

    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)

    fel = FELAnalysis(obs_model, baseline)
    fel.run()
    summary = fel.summary()

    print(f"  Sites analyzed: {summary['n_sites']}")
    print(f"  Significant: {summary['n_significant']}")
    print(f"  Mean ω: {summary['mean_omega']:.4f}")
    print(f"  Median ω: {summary['median_omega']:.4f}")

    assert summary["n_sites"] == 5, "Should analyze 5 sites"
    assert "mean_omega" in summary, "Summary should include mean_omega"
    assert "n_significant" in summary, "Summary should include n_significant"
    print("  ✓ Summary statistics work")


def test_fel_hyphy_json() -> None:
    """Test FEL HyPhy JSON export."""
    print("Testing FEL HyPhy JSON export...")

    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick)

    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)

    atg_idx = codon_space.index("ATG")
    alignment = np.array([[atg_idx, atg_idx]] * 4)
    alignment = _align_alignment_to_tree(tree, ["A", "B", "C", "D"], alignment)

    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)

    fel = FELAnalysis(obs_model, baseline)
    fel.run()
    json_output = fel.to_hyphy_json()

    assert "analysis" in json_output, "JSON should have 'analysis' field"
    assert "MLE" in json_output, "JSON should have 'MLE' field"
    assert "summary" in json_output, "JSON should have 'summary' field"

    mle_content = json_output["MLE"]["content"]
    assert "0" in mle_content, "Should have site 0 results"
    assert "omega" in mle_content["0"], "Site results should have omega"
    assert "alpha" in mle_content["0"], "Site results should have alpha"
    assert "beta" in mle_content["0"], "Site results should have beta"
    assert "p" in mle_content["0"], "Site results should have p-value"

    print("  ✓ HyPhy JSON export works")
    print(f"  ✓ Analysis: {json_output['analysis']}")
    print(f"  ✓ Sites: {json_output['input']['n_sites']}")


def test_fel_site_classification() -> None:
    """Test FEL site classification (positive/purifying)."""
    print("Testing FEL site classification...")

    results = [
        FELSiteResult(
            site=0,
            alpha=1.0,
            beta=2.0,
            omega=2.0,
            log_likelihood=-10.0,
            lrt_statistic=5.0,
            p_value=0.025,
            significant=True,
        ),
        FELSiteResult(
            site=1,
            alpha=1.0,
            beta=0.1,
            omega=0.1,
            log_likelihood=-10.0,
            lrt_statistic=5.0,
            p_value=0.025,
            significant=True,
        ),
        FELSiteResult(
            site=2,
            alpha=1.0,
            beta=1.0,
            omega=1.0,
            log_likelihood=-10.0,
            lrt_statistic=0.1,
            p_value=0.75,
            significant=False,
        ),
    ]

    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick)
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)
    atg_idx = codon_space.index("ATG")
    alignment = np.array([[atg_idx], [atg_idx], [atg_idx], [atg_idx]])
    alignment = _align_alignment_to_tree(tree, ["A", "B", "C", "D"], alignment)

    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)

    fel = FELAnalysis(obs_model, baseline)
    fel.site_results = results

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
