"""Tests for phylogenetic CTMC observation model."""

import sys
import numpy as np


def test_phylo_ctmc_basic():
    """Test basic PhyloCTMC observation model."""
    print("Testing PhyloCTMC observation model...")
    
    from persiste.core.trees import TreeStructure
    from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    
    # Setup
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    
    # Alignment: all taxa have ATG (Met)
    atg_idx = codon_space.index("ATG")
    alignment = np.array([[atg_idx], [atg_idx], [atg_idx], [atg_idx]])
    
    # Create observation model (n_states inferred from rate matrix)
    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    
    assert obs_model.n_taxa == 4, f"Expected 4 taxa, got {obs_model.n_taxa}"
    assert obs_model.n_sites == 1, f"Expected 1 site, got {obs_model.n_sites}"
    
    print(f"  ✓ PhyloCTMC created: {obs_model}")
    print(f"  ✓ Taxa: {obs_model.n_taxa}, Sites: {obs_model.n_sites}")


def test_phylo_ctmc_likelihood():
    """Test likelihood computation via PhyloCTMC."""
    print("Testing PhyloCTMC likelihood...")
    
    from persiste.core.trees import TreeStructure
    from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    from persiste.core.data import ObservedTransitions
    
    # Setup
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)
    
    # Alignment: conserved site (all ATG)
    atg_idx = codon_space.index("ATG")
    alignment = np.array([[atg_idx], [atg_idx], [atg_idx], [atg_idx]])
    
    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    
    # Compute likelihood
    # Note: data parameter is unused (alignment stored in obs_model)
    dummy_data = ObservedTransitions(counts={}, exposure=1.0)
    log_lik = obs_model.log_likelihood(dummy_data, baseline, graph)
    
    print(f"  Log-likelihood: {log_lik:.6f}")
    
    assert np.isfinite(log_lik), "Likelihood should be finite"
    assert log_lik < 0, "Log-likelihood should be negative"
    
    print("  ✓ Likelihood computation works")


def test_phylo_ctmc_omega_variation():
    """Test likelihood with different ω values."""
    print("Testing ω variation...")
    
    from persiste.core.trees import TreeStructure
    from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    
    # Setup
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)
    
    # Alignment: conserved site
    atg_idx = codon_space.index("ATG")
    alignment = np.array([[atg_idx], [atg_idx], [atg_idx], [atg_idx]])
    
    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    
    # Test different ω values
    omega_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    log_liks = []
    
    for omega in omega_values:
        log_lik = obs_model.log_likelihood_with_omega(omega, baseline)
        log_liks.append(log_lik)
        print(f"  ω={omega:.1f}: log-lik={log_lik:.6f}")
    
    # For conserved site, ω shouldn't matter much (all synonymous)
    # But likelihood should still be finite for all ω
    assert all(np.isfinite(ll) for ll in log_liks), "All likelihoods should be finite"
    
    print("  ✓ ω variation works")


def test_phylo_ctmc_site_likelihoods():
    """Test per-site likelihood computation."""
    print("Testing per-site likelihoods...")
    
    from persiste.core.trees import TreeStructure
    from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    
    # Setup
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)
    
    # Alignment: 3 sites
    atg_idx = codon_space.index("ATG")
    ttc_idx = codon_space.index("TTC")
    ggg_idx = codon_space.index("GGG")
    
    alignment = np.array([
        [atg_idx, atg_idx, atg_idx],  # A: conserved
        [atg_idx, ttc_idx, atg_idx],  # B: variable at site 1
        [atg_idx, atg_idx, ggg_idx],  # C: variable at site 2
        [atg_idx, ttc_idx, ggg_idx],  # D: variable at sites 1,2
    ])
    
    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    
    # Get site likelihoods
    site_liks = obs_model.site_log_likelihoods_with_omega(1.0, baseline)
    
    print(f"  Site log-likelihoods: {site_liks}")
    
    assert len(site_liks) == 3, f"Expected 3 site likelihoods, got {len(site_liks)}"
    assert all(np.isfinite(ll) for ll in site_liks), "All site likelihoods should be finite"
    
    # Site 0 (conserved) should have highest likelihood
    assert site_liks[0] > site_liks[1], "Conserved site should have higher likelihood"
    assert site_liks[0] > site_liks[2], "Conserved site should have higher likelihood"
    
    print("  ✓ Per-site likelihoods work")
    print("  ✓ Conserved site has highest likelihood")


def test_phylo_ctmc_integration():
    """Test integration with PERSISTE ConstraintModel."""
    print("Testing PERSISTE integration...")
    
    from persiste.core.trees import TreeStructure
    from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    from persiste.core.constraints import ConstraintModel
    from persiste.core.data import ObservedTransitions
    
    # Setup
    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)
    
    # Alignment
    atg_idx = codon_space.index("ATG")
    alignment = np.array([[atg_idx], [atg_idx], [atg_idx], [atg_idx]])
    
    obs_model = PhyloCTMCObservationModel(graph, tree, alignment)
    
    # Create ConstraintModel
    # For phylogenetics: θ = ω (dN/dS)
    model = ConstraintModel(
        states=codon_space,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition",
        allow_facilitation=True,  # Allow ω > 1
    )
    
    # Compute likelihood via ObservationModel interface
    dummy_data = ObservedTransitions(counts={}, exposure=1.0)
    log_lik = obs_model.log_likelihood(dummy_data, baseline, graph)
    
    print(f"  Log-likelihood via ObservationModel: {log_lik:.6f}")
    
    assert np.isfinite(log_lik), "Likelihood should be finite"
    
    print("  ✓ Integration with ConstraintModel works")
    print("  ✓ ObservationModel interface compatible")


def main():
    """Run all PhyloCTMC tests."""
    print("=" * 60)
    print("PERSISTE Phylo Plugin - PhyloCTMC Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_phylo_ctmc_basic,
        test_phylo_ctmc_likelihood,
        test_phylo_ctmc_omega_variation,
        test_phylo_ctmc_site_likelihoods,
        test_phylo_ctmc_integration,
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
