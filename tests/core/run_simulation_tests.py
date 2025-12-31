"""
Simple test runner for simulation module (no pytest required).
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from persiste.core.trees import TreeStructure
from persiste.core.simulation import (
    simulate_binary_evolution,
    compute_equilibrium_frequencies,
    compute_stationary_frequency,
    compute_mean_transitions,
)


def test_equilibrium_frequencies():
    """Test equilibrium frequency calculations."""
    print("Testing equilibrium frequencies...")
    
    # Equal rates
    pi_0, pi_1 = compute_equilibrium_frequencies(1.0, 1.0)
    assert np.isclose(pi_0, 0.5) and np.isclose(pi_1, 0.5)
    
    # Gain dominated
    pi_0, pi_1 = compute_equilibrium_frequencies(3.0, 1.0)
    assert np.isclose(pi_1, 0.75) and np.isclose(pi_0, 0.25)
    
    # Loss dominated
    pi_0, pi_1 = compute_equilibrium_frequencies(1.0, 3.0)
    assert np.isclose(pi_0, 0.75) and np.isclose(pi_1, 0.25)
    
    # Sum to one
    for gain in [0.5, 1.0, 2.0, 5.0]:
        for loss in [0.5, 1.0, 2.0, 5.0]:
            pi_0, pi_1 = compute_equilibrium_frequencies(gain, loss)
            assert np.isclose(pi_0 + pi_1, 1.0)
    
    print("  ✓ All equilibrium frequency tests passed")


def test_simulation_basic():
    """Test basic simulation functionality."""
    print("Testing basic simulation...")
    
    newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    
    # Test shape
    n_sites = 50
    matrix = simulate_binary_evolution(tree, 1.0, 1.0, n_sites, rng)
    assert matrix.shape == (tree.n_tips, n_sites)
    assert matrix.dtype == np.int8
    
    # Test binary values
    assert np.all((matrix == 0) | (matrix == 1))
    
    # Test reproducibility
    rng1 = np.random.default_rng(123)
    matrix1 = simulate_binary_evolution(tree, 1.5, 2.0, 50, rng1)
    rng2 = np.random.default_rng(123)
    matrix2 = simulate_binary_evolution(tree, 1.5, 2.0, 50, rng2)
    assert np.array_equal(matrix1, matrix2)
    
    print("  ✓ All basic simulation tests passed")


def test_simulation_equilibrium():
    """Test that simulation approaches equilibrium."""
    print("Testing simulation equilibrium...")
    
    newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    
    gain_rate = 1.5
    loss_rate = 2.0
    expected_pi1 = gain_rate / (gain_rate + loss_rate)
    
    # Simulate many sites
    matrix = simulate_binary_evolution(tree, gain_rate, loss_rate, 10000, rng)
    observed_pi1 = matrix.mean()
    
    # Should be close to equilibrium (within 5%)
    assert abs(observed_pi1 - expected_pi1) < 0.05
    
    print(f"  Expected π₁: {expected_pi1:.3f}, Observed: {observed_pi1:.3f}")
    print("  ✓ Equilibrium test passed")


def test_site_specific_rates():
    """Test site-specific rate variation."""
    print("Testing site-specific rates...")
    
    newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    
    # First 10 sites have high gain, low loss
    site_specific_rates = {}
    for i in range(10):
        site_specific_rates[i] = (3.0, 0.5)  # High presence
    
    matrix = simulate_binary_evolution(
        tree, 0.5, 3.0, 50, rng,
        site_specific_rates=site_specific_rates
    )
    
    # First 10 sites should have higher presence
    presence_high = matrix[:, :10].mean()
    presence_low = matrix[:, 10:].mean()
    
    assert presence_high > presence_low
    print(f"  High-rate sites: {presence_high:.3f}, Low-rate sites: {presence_low:.3f}")
    print("  ✓ Site-specific rates test passed")


def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")
    
    # Single tip
    newick = "A:1.0;"
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    matrix = simulate_binary_evolution(tree, 1.0, 1.0, 10, rng)
    assert matrix.shape == (1, 10)
    
    # Very long branches (should reach equilibrium)
    newick = "((A:100.0,B:100.0):100.0,(C:100.0,D:100.0):100.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    gain_rate, loss_rate = 1.5, 2.0
    expected_pi1 = gain_rate / (gain_rate + loss_rate)
    matrix = simulate_binary_evolution(tree, gain_rate, loss_rate, 1000, rng)
    observed_pi1 = matrix.mean()
    assert abs(observed_pi1 - expected_pi1) < 0.05
    
    print("  ✓ All edge case tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Simulation Module Tests")
    print("=" * 60)
    print()
    
    try:
        test_equilibrium_frequencies()
        test_simulation_basic()
        test_simulation_equilibrium()
        test_site_specific_rates()
        test_edge_cases()
        
        print()
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        return False
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
