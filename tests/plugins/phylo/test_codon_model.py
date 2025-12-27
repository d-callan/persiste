"""
Tests for codon model components.

Verifies:
- GeneticCode: codon→amino acid mapping, synonymous classification
- CodonStateSpace: 61 sense codons, proper indexing
- CodonTransitionGraph: single-nt changes, syn/nonsyn classification
- MG94Baseline: rate parameterization, κ and ω effects
- Integration with PERSISTE core (ConstraintModel)
"""

import sys
import numpy as np


def test_genetic_code():
    """Test genetic code basics."""
    print("Testing genetic code...")
    
    from persiste.plugins.phylo.states.genetic_code import GeneticCode
    
    # Universal code
    code = GeneticCode.universal()
    
    assert len(code.sense_codons) == 61, f"Expected 61 sense codons, got {len(code.sense_codons)}"
    assert len(code.stop_codons) == 3, f"Expected 3 stop codons, got {len(code.stop_codons)}"
    
    # Translation
    assert code.translate("ATG") == "M", "ATG should encode Met"
    assert code.translate("TAA") == "*", "TAA should be stop"
    
    # Synonymous check
    assert code.is_synonymous("TTT", "TTC"), "TTT and TTC both encode Phe"
    assert not code.is_synonymous("TTT", "TTA"), "TTT (Phe) and TTA (Leu) differ"
    
    # Single nucleotide change
    assert code.is_single_nucleotide_change("TTT", "TTC"), "TTT→TTC is single change"
    assert not code.is_single_nucleotide_change("TTT", "TAC"), "TTT→TAC is double change"
    
    # Transition/transversion
    assert code.is_transition("A", "G"), "A→G is transition"
    assert code.is_transition("C", "T"), "C→T is transition"
    assert code.is_transversion("A", "C"), "A→C is transversion"
    assert code.is_transversion("G", "T"), "G→T is transversion"
    
    print("  ✓ Universal genetic code (61 sense, 3 stop)")
    print("  ✓ Translation correct (ATG→M, TAA→*)")
    print("  ✓ Synonymous classification works")
    print("  ✓ Transition/transversion detection works")


def test_codon_state_space():
    """Test codon state space."""
    print("Testing codon state space...")
    
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    
    space = CodonStateSpace.universal()
    
    # Dimension
    assert space.dimension == 61, f"Expected 61 codons, got {space.dimension}"
    
    # Index ↔ codon mapping
    for i in range(space.dimension):
        codon = space.codon(i)
        assert space.index(codon) == i, f"Index mismatch for {codon}"
    
    # Amino acid lookup
    met_idx = space.index("ATG")
    assert space.amino_acid(met_idx) == "M", "ATG should give Met"
    
    # Synonymous check via indices
    ttt_idx = space.index("TTT")
    ttc_idx = space.index("TTC")
    tta_idx = space.index("TTA")
    
    assert space.is_synonymous(ttt_idx, ttc_idx), "TTT↔TTC should be synonymous"
    assert not space.is_synonymous(ttt_idx, tta_idx), "TTT↔TTA should be nonsynonymous"
    
    # Frequencies
    freqs = space.frequencies
    assert len(freqs) == 61, "Should have 61 frequencies"
    assert np.isclose(freqs.sum(), 1.0), "Frequencies should sum to 1"
    
    print(f"  ✓ CodonStateSpace dimension = {space.dimension}")
    print("  ✓ Index ↔ codon mapping bijective")
    print("  ✓ Amino acid lookup works")
    print("  ✓ Synonymous classification via indices works")


def test_codon_transition_graph():
    """Test codon transition graph."""
    print("Testing codon transition graph...")
    
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    
    graph = CodonTransitionGraph.universal()
    
    # Count edges
    counts = graph.count_edges()
    
    print(f"  Total edges: {counts['total']}")
    print(f"  Synonymous: {counts['synonymous']}")
    print(f"  Nonsynonymous: {counts['nonsynonymous']}")
    print(f"  Transitions: {counts['transition']}")
    print(f"  Transversions: {counts['transversion']}")
    
    # Basic properties
    assert counts['total'] > 0, "Should have edges"
    assert counts['synonymous'] + counts['nonsynonymous'] == counts['total']
    assert counts['transition'] + counts['transversion'] == counts['total']
    
    # Specific checks
    ttt_idx = graph.codon_space.index("TTT")
    ttc_idx = graph.codon_space.index("TTC")
    tac_idx = graph.codon_space.index("TAC")
    
    assert graph.allows(ttt_idx, ttc_idx), "TTT→TTC should be allowed (single nt)"
    assert not graph.allows(ttt_idx, tac_idx), "TTT→TAC should not be allowed (2 nt)"
    
    assert graph.is_synonymous(ttt_idx, ttc_idx), "TTT→TTC is synonymous"
    assert graph.is_transition(ttt_idx, ttc_idx), "TTT→TTC is transition (T→C)"
    
    print("  ✓ Graph structure correct")
    print("  ✓ Only single-nt changes allowed")
    print("  ✓ Synonymous/nonsynonymous classification correct")


def test_mg94_baseline():
    """Test MG94 baseline model."""
    print("Testing MG94 baseline...")
    
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    
    # Default (κ=1, ω=1, uniform frequencies)
    baseline = MG94Baseline.universal(kappa=1.0, omega=1.0)
    
    # Check rate structure
    ttt_idx = baseline.codon_space.index("TTT")
    ttc_idx = baseline.codon_space.index("TTC")
    tta_idx = baseline.codon_space.index("TTA")
    tac_idx = baseline.codon_space.index("TAC")
    
    # Single-nt changes should have positive rates
    rate_syn = baseline.get_rate(ttt_idx, ttc_idx)  # Synonymous
    rate_nonsyn = baseline.get_rate(ttt_idx, tta_idx)  # Nonsynonymous
    
    assert rate_syn > 0, "Synonymous rate should be positive"
    assert rate_nonsyn > 0, "Nonsynonymous rate should be positive"
    
    # Multi-nt changes should have zero rate
    rate_multi = baseline.get_rate(ttt_idx, tac_idx)
    assert rate_multi == 0, "Multi-nt change should have zero rate"
    
    # With κ > 1, transitions should be faster
    baseline_kappa = MG94Baseline.universal(kappa=2.0, omega=1.0)
    
    rate_ts = baseline_kappa.get_rate(ttt_idx, ttc_idx)  # Transition
    rate_tv = baseline_kappa.get_rate(ttt_idx, tta_idx)  # Transversion
    
    # TTT→TTC is T→C (transition), TTT→TTA is T→A (transversion)
    # Wait, TTT→TTA changes position 2 from T to A? Let me check.
    # TTT = T T T
    # TTA = T T A
    # Third position: T→A = transversion
    # So rate_tv should be lower than rate_ts
    
    print(f"  Rate TTT→TTC (transition): {rate_ts:.6f}")
    print(f"  Rate TTT→TTA (transversion): {rate_tv:.6f}")
    
    assert rate_ts > rate_tv, "With κ=2, transitions should be faster"
    
    # Rate matrix
    Q = baseline.build_rate_matrix(omega=1.0)
    assert Q.shape == (61, 61), "Rate matrix should be 61×61"
    assert np.allclose(Q.sum(axis=1), 0), "Rows should sum to 0"
    
    print("  ✓ MG94 rate structure correct")
    print("  ✓ κ affects transition/transversion ratio")
    print("  ✓ Rate matrix valid (rows sum to 0)")


def test_integration_with_constraint_model():
    """Test that MG94 integrates with PERSISTE ConstraintModel."""
    print("Testing integration with ConstraintModel...")
    
    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.core.constraints import ConstraintModel
    
    # Create codon model components
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)
    
    # Create ConstraintModel
    # θ = ω in phylogenetics
    model = ConstraintModel(
        states=codon_space,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition",
        allow_facilitation=True,  # Allow ω > 1 (positive selection)
    )
    
    # Get indices for test
    ttt_idx = codon_space.index("TTT")
    ttc_idx = codon_space.index("TTC")  # Synonymous to TTT
    tta_idx = codon_space.index("TTA")  # Nonsynonymous to TTT
    
    # Set ω = 1 for synonymous (no constraint)
    # Set ω = 0.5 for nonsynonymous (purifying selection)
    model.set_parameters(theta={
        (ttt_idx, ttc_idx): 1.0,  # Synonymous: θ = 1
        (ttt_idx, tta_idx): 0.5,  # Nonsynonymous: θ = ω = 0.5
    })
    
    # Check effective rates
    rate_syn = model.effective_rate(ttt_idx, ttc_idx)
    rate_nonsyn = model.effective_rate(ttt_idx, tta_idx)
    
    baseline_syn = baseline.get_rate(ttt_idx, ttc_idx)
    baseline_nonsyn = baseline.get_rate(ttt_idx, tta_idx)
    
    assert np.isclose(rate_syn, baseline_syn), "Synonymous rate should be unchanged (θ=1)"
    assert np.isclose(rate_nonsyn, 0.5 * baseline_nonsyn), "Nonsynonymous rate should be halved (θ=0.5)"
    
    print(f"  Synonymous: baseline={baseline_syn:.6f}, effective={rate_syn:.6f} (θ=1)")
    print(f"  Nonsynonymous: baseline={baseline_nonsyn:.6f}, effective={rate_nonsyn:.6f} (θ=0.5)")
    
    # Test positive selection (ω > 1)
    model.set_parameters(theta={
        (ttt_idx, tta_idx): 2.0,  # Positive selection: θ = ω = 2.0
    })
    
    rate_positive = model.effective_rate(ttt_idx, tta_idx)
    assert np.isclose(rate_positive, 2.0 * baseline_nonsyn), "Positive selection should double rate"
    
    print(f"  Positive selection: effective={rate_positive:.6f} (θ=2.0)")
    
    # Test facilitation policy
    model_no_facilitation = ConstraintModel(
        states=codon_space,
        baseline=baseline,
        graph=graph,
        constraint_structure="per_transition",
        allow_facilitation=False,  # Constraint-only
    )
    model_no_facilitation.set_parameters(theta={(ttt_idx, tta_idx): 2.0})
    
    rate_capped = model_no_facilitation.effective_rate(ttt_idx, tta_idx)
    assert np.isclose(rate_capped, baseline_nonsyn), "Without facilitation, θ=2 should be capped to θ=1"
    
    print("  ✓ ConstraintModel integrates with MG94")
    print("  ✓ θ = ω correctly modifies nonsynonymous rates")
    print("  ✓ allow_facilitation controls positive selection")


def main():
    """Run all codon model tests."""
    print("=" * 60)
    print("PERSISTE Phylo Plugin - Codon Model Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_genetic_code,
        test_codon_state_space,
        test_codon_transition_graph,
        test_mg94_baseline,
        test_integration_with_constraint_model,
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
