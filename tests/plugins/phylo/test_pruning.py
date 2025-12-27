"""Tests for Felsenstein pruning algorithm."""

import sys
import numpy as np


def test_simple_likelihood():
    """Test likelihood computation on a simple tree."""
    print("Testing simple likelihood computation...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    from persiste.plugins.phylo.observation.pruning import FelsensteinPruning
    
    # Simple 2-taxon tree: (A:0.1,B:0.1);
    newick = "(A:0.1,B:0.1);"
    tree = PhylogeneticTree.from_string(newick)
    
    # Simple 2-state model (like nucleotides A/C)
    n_states = 2
    
    # Jukes-Cantor rate matrix: equal rates between states
    # Q = [[-1, 1], [1, -1]]
    Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
    
    # Equal frequencies
    freqs = np.array([0.5, 0.5])
    
    # Data: both taxa in state 0
    # Shape: (n_taxa=2, n_sites=1)
    data = np.array([[0], [0]])
    
    pruning = FelsensteinPruning(tree.tree, n_states, lambda: Q)
    log_lik = pruning.compute_likelihood(data, Q, freqs)
    
    # When both taxa are in the same state, likelihood should be higher
    # than if they were in different states
    print(f"  Log-likelihood (same state): {log_lik:.6f}")
    
    # Data: taxa in different states
    data_diff = np.array([[0], [1]])
    log_lik_diff = pruning.compute_likelihood(data_diff, Q, freqs)
    
    print(f"  Log-likelihood (diff state): {log_lik_diff:.6f}")
    
    assert log_lik > log_lik_diff, "Same state should have higher likelihood"
    
    print("  ✓ Likelihood computation works")
    print("  ✓ Same state has higher likelihood than different state")


def test_jukes_cantor_likelihood():
    """Test likelihood with Jukes-Cantor model on 4-taxon tree."""
    print("Testing Jukes-Cantor likelihood...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    from persiste.plugins.phylo.observation.pruning import FelsensteinPruning
    
    # 4-taxon tree
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = PhylogeneticTree.from_string(newick)
    
    # 4-state Jukes-Cantor (nucleotides)
    n_states = 4
    
    # JC69: equal rates, rate = 1
    rate = 1.0
    Q = np.ones((n_states, n_states)) * rate / (n_states - 1)
    np.fill_diagonal(Q, -rate)
    
    # Equal frequencies
    freqs = np.ones(n_states) / n_states
    
    # Data: all taxa in state 0 (conserved site)
    data_conserved = np.array([[0], [0], [0], [0]])
    
    pruning = FelsensteinPruning(tree.tree, n_states, lambda: Q)
    log_lik_conserved = pruning.compute_likelihood(data_conserved, Q, freqs)
    
    # Data: all different states (variable site)
    data_variable = np.array([[0], [1], [2], [3]])
    log_lik_variable = pruning.compute_likelihood(data_variable, Q, freqs)
    
    print(f"  Log-likelihood (conserved): {log_lik_conserved:.6f}")
    print(f"  Log-likelihood (variable):  {log_lik_variable:.6f}")
    
    # Conserved sites should have higher likelihood
    assert log_lik_conserved > log_lik_variable, \
        "Conserved site should have higher likelihood"
    
    print("  ✓ JC69 likelihood computation works")
    print("  ✓ Conserved sites have higher likelihood")


def test_multiple_sites():
    """Test likelihood computation with multiple sites."""
    print("Testing multiple sites...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    from persiste.plugins.phylo.observation.pruning import FelsensteinPruning
    
    # Simple tree
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = PhylogeneticTree.from_string(newick)
    
    # 4-state model
    n_states = 4
    Q = np.ones((n_states, n_states)) * 0.25
    np.fill_diagonal(Q, -0.75)
    freqs = np.ones(n_states) / n_states
    
    # Data: 3 sites
    # Site 0: conserved (all 0)
    # Site 1: variable (0,1,2,3)
    # Site 2: partially conserved (0,0,1,1)
    data = np.array([
        [0, 0, 0],  # A
        [0, 1, 0],  # B
        [0, 2, 1],  # C
        [0, 3, 1],  # D
    ])
    
    pruning = FelsensteinPruning(tree.tree, n_states, lambda: Q)
    
    # Total likelihood
    log_lik_total = pruning.compute_likelihood(data, Q, freqs)
    
    # Per-site likelihoods
    site_liks = pruning.compute_site_likelihoods(data, Q, freqs)
    
    print(f"  Total log-likelihood: {log_lik_total:.6f}")
    print(f"  Site log-likelihoods: {site_liks}")
    
    # Total should equal sum of sites
    assert np.isclose(log_lik_total, site_liks.sum()), \
        "Total likelihood should equal sum of site likelihoods"
    
    # Site 0 (conserved) should have highest likelihood
    assert site_liks[0] > site_liks[1], "Conserved site should have highest likelihood"
    
    print("  ✓ Multiple sites work correctly")
    print("  ✓ Total = sum of site likelihoods")


def test_missing_data():
    """Test handling of missing/ambiguous data."""
    print("Testing missing data...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    from persiste.plugins.phylo.observation.pruning import FelsensteinPruning
    
    # Simple tree
    newick = "(A:0.1,B:0.1);"
    tree = PhylogeneticTree.from_string(newick)
    
    # 2-state model
    n_states = 2
    Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
    freqs = np.array([0.5, 0.5])
    
    # Data with missing value (-1)
    # A is in state 0, B is missing
    data_missing = np.array([[0], [-1]])
    
    pruning = FelsensteinPruning(tree.tree, n_states, lambda: Q)
    log_lik_missing = pruning.compute_likelihood(data_missing, Q, freqs)
    
    # Should not crash and should return finite likelihood
    assert np.isfinite(log_lik_missing), "Likelihood with missing data should be finite"
    
    # Compare to complete data
    data_complete = np.array([[0], [0]])
    log_lik_complete = pruning.compute_likelihood(data_complete, Q, freqs)
    
    print(f"  Log-likelihood (with missing): {log_lik_missing:.6f}")
    print(f"  Log-likelihood (complete):     {log_lik_complete:.6f}")
    
    # Missing data should have higher (less negative) likelihood
    # because it marginalizes over all states, making data less constraining
    assert log_lik_missing > log_lik_complete, \
        "Missing data should increase likelihood (marginalizes over uncertainty)"
    
    print("  ✓ Missing data handled correctly")
    print("  ✓ Missing data increases likelihood (less constraining)")


def test_codon_likelihood():
    """Test likelihood with codon model (61 states)."""
    print("Testing codon model likelihood...")
    
    from persiste.plugins.phylo.data.tree import PhylogeneticTree
    from persiste.plugins.phylo.observation.pruning import FelsensteinPruning
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    
    # Simple tree
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = PhylogeneticTree.from_string(newick)
    
    # Codon model
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)
    
    n_states = 61
    Q = baseline.build_rate_matrix(omega=1.0)
    freqs = codon_space.frequencies
    
    # Data: all taxa have codon ATG (Met, index varies)
    atg_idx = codon_space.index("ATG")
    data = np.array([[atg_idx], [atg_idx], [atg_idx], [atg_idx]])
    
    pruning = FelsensteinPruning(tree.tree, n_states, lambda: Q)
    log_lik = pruning.compute_likelihood(data, Q, freqs)
    
    print(f"  Log-likelihood (61-state codon model): {log_lik:.6f}")
    
    assert np.isfinite(log_lik), "Codon likelihood should be finite"
    assert log_lik < 0, "Log-likelihood should be negative"
    
    print("  ✓ Codon model (61 states) works")
    print(f"  ✓ MG94 rate matrix integrated with pruning")


def main():
    """Run all pruning tests."""
    print("=" * 60)
    print("PERSISTE Phylo Plugin - Pruning Algorithm Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_simple_likelihood,
        test_jukes_cantor_likelihood,
        test_multiple_sites,
        test_missing_data,
        test_codon_likelihood,
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
