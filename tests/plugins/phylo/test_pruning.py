"""Tests for Felsenstein pruning algorithm."""

import sys

import numpy as np
from scipy.linalg import expm

from persiste.core.pruning import ArrayTipConditionalProvider, FelsensteinPruning
from persiste.core.trees import TreeStructure


class ConstantRateMatrixProvider:
    """Simple transition provider that exponentiates a fixed rate matrix."""

    def __init__(self, rate_matrix: np.ndarray, freqs: np.ndarray):
        self._rate_matrix = rate_matrix
        self.equilibrium_frequencies = freqs
        self._n_states = rate_matrix.shape[0]

    @property
    def n_states(self) -> int:
        return self._n_states

    def get_transition_matrix(self, branch_length: float) -> np.ndarray:
        if branch_length <= 0:
            return np.eye(self._n_states)
        return expm(self._rate_matrix * branch_length)


def _align_data_to_tree(
    tree: TreeStructure,
    taxa_order: list[str],
    alignment: np.ndarray,
) -> np.ndarray:
    """Reorder alignment rows to match the tree's tip ordering."""
    order_map = {name: i for i, name in enumerate(taxa_order)}
    indices = [order_map[name] for name in tree.tip_names]
    return alignment[indices]


def _run_pruning(
    tree: TreeStructure,
    data: np.ndarray,
    rate_matrix: np.ndarray,
    freqs: np.ndarray,
    *,
    return_per_site: bool = False,
) -> tuple[float, np.ndarray | None]:
    """Compute log-likelihood (and optionally per-site logs) via core pruning."""
    provider = ConstantRateMatrixProvider(rate_matrix, freqs)
    pruning = FelsensteinPruning(tree, rate_matrix.shape[0], use_jax=False)
    tip_provider = ArrayTipConditionalProvider(
        data=data,
        taxon_names=tree.tip_names,
        n_states=rate_matrix.shape[0],
    )
    result = pruning.compute_likelihood(
        transition_provider=provider,
        tip_provider=tip_provider,
        n_sites=data.shape[1],
        return_per_site=return_per_site,
    )
    return result.log_likelihood, result.site_log_likelihoods


def test_simple_likelihood() -> None:
    """Test likelihood computation on a simple tree."""
    print("Testing simple likelihood computation...")

    newick = "(A:0.1,B:0.1);"
    tree = TreeStructure.from_newick(newick)
    rate_matrix = np.array([[-1.0, 1.0], [1.0, -1.0]])
    freqs = np.array([0.5, 0.5])

    data = np.array([[0], [0]])
    data = _align_data_to_tree(tree, ["A", "B"], data)
    log_lik, _ = _run_pruning(tree, data, rate_matrix, freqs)
    print(f"  Log-likelihood (same state): {log_lik:.6f}")

    data_diff = np.array([[0], [1]])
    data_diff = _align_data_to_tree(tree, ["A", "B"], data_diff)
    log_lik_diff, _ = _run_pruning(tree, data_diff, rate_matrix, freqs)
    print(f"  Log-likelihood (diff state): {log_lik_diff:.6f}")

    assert log_lik > log_lik_diff, "Same state should have higher likelihood"
    print("  ✓ Likelihood computation works")


def test_jukes_cantor_likelihood() -> None:
    """Test likelihood with Jukes-Cantor model on 4-taxon tree."""
    print("Testing Jukes-Cantor likelihood...")

    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick)
    n_states = 4
    rate = 1.0
    rate_matrix = np.ones((n_states, n_states)) * rate / (n_states - 1)
    np.fill_diagonal(rate_matrix, -rate)
    freqs = np.ones(n_states) / n_states

    data_conserved = np.array([[0], [0], [0], [0]])
    data_conserved = _align_data_to_tree(tree, ["A", "B", "C", "D"], data_conserved)
    log_lik_conserved, _ = _run_pruning(tree, data_conserved, rate_matrix, freqs)

    data_variable = np.array([[0], [1], [2], [3]])
    data_variable = _align_data_to_tree(tree, ["A", "B", "C", "D"], data_variable)
    log_lik_variable, _ = _run_pruning(tree, data_variable, rate_matrix, freqs)
    print(f"  Log-likelihood (conserved): {log_lik_conserved:.6f}")
    print(f"  Log-likelihood (variable):  {log_lik_variable:.6f}")

    assert log_lik_conserved > log_lik_variable, "Conserved site should have higher likelihood"
    print("  ✓ JC69 likelihood computation works")


def test_multiple_sites() -> None:
    """Test likelihood computation with multiple sites."""
    print("Testing multiple sites...")

    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick)
    n_states = 4
    rate_matrix = np.ones((n_states, n_states)) * 0.25
    np.fill_diagonal(rate_matrix, -0.75)
    freqs = np.ones(n_states) / n_states

    data = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 2, 1],
            [0, 3, 1],
        ]
    )
    data = _align_data_to_tree(tree, ["A", "B", "C", "D"], data)

    log_lik_total, site_liks = _run_pruning(
        tree,
        data,
        rate_matrix,
        freqs,
        return_per_site=True,
    )
    assert site_liks is not None
    print(f"  Total log-likelihood: {log_lik_total:.6f}")
    print(f"  Site log-likelihoods: {site_liks}")

    assert np.isclose(log_lik_total, site_liks.sum()), "Total should equal sum of site likelihoods"
    assert site_liks[0] > site_liks[1], "Conserved site should have highest likelihood"
    print("  ✓ Multiple sites work correctly")


def test_missing_data() -> None:
    """Test handling of missing/ambiguous data."""
    print("Testing missing data...")

    newick = "(A:0.1,B:0.1);"
    tree = TreeStructure.from_newick(newick)
    rate_matrix = np.array([[-1.0, 1.0], [1.0, -1.0]])
    freqs = np.array([0.5, 0.5])

    data_missing = np.array([[0], [-1]])
    data_missing = _align_data_to_tree(tree, ["A", "B"], data_missing)
    log_lik_missing, _ = _run_pruning(tree, data_missing, rate_matrix, freqs)

    data_complete = np.array([[0], [0]])
    data_complete = _align_data_to_tree(tree, ["A", "B"], data_complete)
    log_lik_complete, _ = _run_pruning(tree, data_complete, rate_matrix, freqs)
    print(f"  Log-likelihood (missing): {log_lik_missing:.6f}")
    print(f"  Log-likelihood (complete): {log_lik_complete:.6f}")

    assert log_lik_missing > log_lik_complete, "Missing data should increase likelihood"
    print("  ✓ Missing data handled correctly")


def test_codon_likelihood() -> None:
    """Test likelihood with codon model (61 states)."""
    print("Testing codon model likelihood...")

    from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
    from persiste.plugins.phylo.states.codons import CodonStateSpace
    from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph

    newick = "((A:0.1,B:0.1):0.1,(C:0.1,D:0.1):0.1);"
    tree = TreeStructure.from_newick(newick)
    codon_space = CodonStateSpace.universal()
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(codon_space, graph, kappa=2.0, omega=1.0)

    rate_matrix = baseline.build_rate_matrix(omega=1.0)
    freqs = codon_space.frequencies

    atg_idx = codon_space.index("ATG")
    data = np.array([[atg_idx], [atg_idx], [atg_idx], [atg_idx]])
    data = _align_data_to_tree(tree, ["A", "B", "C", "D"], data)

    log_lik, _ = _run_pruning(tree, data, rate_matrix, freqs)
    print(f"  Log-likelihood (61-state codon model): {log_lik:.6f}")

    assert np.isfinite(log_lik), "Codon likelihood should be finite"
    assert log_lik < 0, "Log-likelihood should be negative"
    print("  ✓ Codon model (61 states) works")


def main() -> int:
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
        except Exception as exc:
            failed += 1
            print(f"  ✗ FAILED: {exc}")
            import traceback

            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
