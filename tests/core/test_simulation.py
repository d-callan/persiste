"""
Unit tests for persiste.core.simulation module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from persiste.core.trees import TreeStructure
from persiste.core.simulation import (
    simulate_binary_evolution,
    compute_equilibrium_frequencies,
    compute_stationary_frequency,
    compute_mean_transitions,
)


class TestEquilibriumFrequencies:
    """Test equilibrium frequency calculations."""
    
    def test_equal_rates(self):
        """Test equilibrium with equal gain and loss rates."""
        pi_0, pi_1 = compute_equilibrium_frequencies(1.0, 1.0)
        assert np.isclose(pi_0, 0.5)
        assert np.isclose(pi_1, 0.5)
        assert np.isclose(pi_0 + pi_1, 1.0)
    
    def test_gain_dominated(self):
        """Test equilibrium with higher gain rate."""
        pi_0, pi_1 = compute_equilibrium_frequencies(3.0, 1.0)
        assert pi_1 > pi_0  # More presence
        assert np.isclose(pi_1, 0.75)
        assert np.isclose(pi_0, 0.25)
    
    def test_loss_dominated(self):
        """Test equilibrium with higher loss rate."""
        pi_0, pi_1 = compute_equilibrium_frequencies(1.0, 3.0)
        assert pi_0 > pi_1  # More absence
        assert np.isclose(pi_0, 0.75)
        assert np.isclose(pi_1, 0.25)
    
    def test_sum_to_one(self):
        """Test that frequencies sum to 1."""
        for gain in [0.5, 1.0, 2.0, 5.0]:
            for loss in [0.5, 1.0, 2.0, 5.0]:
                pi_0, pi_1 = compute_equilibrium_frequencies(gain, loss)
                assert np.isclose(pi_0 + pi_1, 1.0)
    
    def test_stationary_frequency(self):
        """Test stationary frequency matches pi_1."""
        gain, loss = 1.5, 2.0
        pi_0, pi_1 = compute_equilibrium_frequencies(gain, loss)
        pi_1_direct = compute_stationary_frequency(gain, loss)
        assert np.isclose(pi_1, pi_1_direct)


class TestBinaryEvolutionSimulation:
    """Test binary trait evolution simulation."""
    
    @pytest.fixture
    def simple_tree(self):
        """Create a simple 4-tip tree."""
        newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
        return TreeStructure.from_newick(newick, backend="simple")
    
    @pytest.fixture
    def rng(self):
        """Create a reproducible RNG."""
        return np.random.default_rng(42)
    
    def test_output_shape(self, simple_tree, rng):
        """Test output matrix has correct shape."""
        n_sites = 50
        matrix = simulate_binary_evolution(
            tree=simple_tree,
            gain_rate=1.0,
            loss_rate=1.0,
            n_sites=n_sites,
            rng=rng
        )
        
        assert matrix.shape == (simple_tree.n_tips, n_sites)
        assert matrix.dtype == np.int8
    
    def test_binary_values(self, simple_tree, rng):
        """Test that all values are 0 or 1."""
        matrix = simulate_binary_evolution(
            tree=simple_tree,
            gain_rate=1.0,
            loss_rate=1.0,
            n_sites=100,
            rng=rng
        )
        
        assert np.all((matrix == 0) | (matrix == 1))
    
    def test_reproducibility(self, simple_tree):
        """Test that same seed gives same results."""
        rng1 = np.random.default_rng(123)
        matrix1 = simulate_binary_evolution(
            tree=simple_tree,
            gain_rate=1.5,
            loss_rate=2.0,
            n_sites=50,
            rng=rng1
        )
        
        rng2 = np.random.default_rng(123)
        matrix2 = simulate_binary_evolution(
            tree=simple_tree,
            gain_rate=1.5,
            loss_rate=2.0,
            n_sites=50,
            rng=rng2
        )
        
        assert np.array_equal(matrix1, matrix2)
    
    def test_equilibrium_frequency(self, simple_tree, rng):
        """Test that simulated data approaches equilibrium frequency."""
        gain_rate = 1.5
        loss_rate = 2.0
        expected_pi1 = gain_rate / (gain_rate + loss_rate)
        
        # Simulate many sites
        matrix = simulate_binary_evolution(
            tree=simple_tree,
            gain_rate=gain_rate,
            loss_rate=loss_rate,
            n_sites=10000,
            rng=rng
        )
        
        # Check overall presence frequency
        observed_pi1 = matrix.mean()
        
        # Should be close to equilibrium (within 5%)
        assert abs(observed_pi1 - expected_pi1) < 0.05
    
    def test_site_specific_rates(self, simple_tree, rng):
        """Test site-specific rate variation."""
        # First 10 sites have high gain, low loss
        # Remaining sites have low gain, high loss
        site_specific_rates = {}
        for i in range(10):
            site_specific_rates[i] = (3.0, 0.5)  # High presence
        
        matrix = simulate_binary_evolution(
            tree=simple_tree,
            gain_rate=0.5,  # Default: low presence
            loss_rate=3.0,
            n_sites=50,
            rng=rng,
            site_specific_rates=site_specific_rates
        )
        
        # First 10 sites should have higher presence
        presence_high = matrix[:, :10].mean()
        presence_low = matrix[:, 10:].mean()
        
        assert presence_high > presence_low
    
    def test_custom_root_frequencies(self, simple_tree, rng):
        """Test custom root state frequencies."""
        # Force all roots to be present
        matrix = simulate_binary_evolution(
            tree=simple_tree,
            gain_rate=0.1,
            loss_rate=0.1,
            n_sites=100,
            rng=rng,
            root_frequencies=(0.0, 1.0)  # Always start present
        )
        
        # With low rates and short branches, should stay mostly present
        assert matrix.mean() > 0.5
    
    def test_zero_rates(self, simple_tree, rng):
        """Test behavior with zero rates (no evolution)."""
        # No gain, no loss = frozen state
        matrix = simulate_binary_evolution(
            tree=simple_tree,
            gain_rate=0.0,
            loss_rate=0.0,
            n_sites=100,
            rng=rng,
            root_frequencies=(0.5, 0.5)
        )
        
        # Each column should be constant (no change along tree)
        for col in range(matrix.shape[1]):
            unique_vals = np.unique(matrix[:, col])
            # All tips should have same state (inherited from root)
            assert len(unique_vals) == 1
    
    def test_high_rates_high_variance(self, simple_tree, rng):
        """Test that high rates lead to more variation."""
        # Low rates
        matrix_low = simulate_binary_evolution(
            tree=simple_tree,
            gain_rate=0.1,
            loss_rate=0.1,
            n_sites=1000,
            rng=np.random.default_rng(42)
        )
        
        # High rates
        matrix_high = simulate_binary_evolution(
            tree=simple_tree,
            gain_rate=5.0,
            loss_rate=5.0,
            n_sites=1000,
            rng=np.random.default_rng(43)
        )
        
        # High rates should have more variation across tips
        var_low = matrix_low.var(axis=0).mean()
        var_high = matrix_high.var(axis=0).mean()
        
        # High rates should have higher variance (more transitions)
        assert var_high >= var_low


class TestMeanTransitions:
    """Test mean transitions calculation."""
    
    def test_simple_tree(self):
        """Test mean transitions on simple tree."""
        newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
        tree = TreeStructure.from_newick(newick, backend="simple")
        
        gain_rate = 1.5
        loss_rate = 2.0
        total_rate = gain_rate + loss_rate
        
        mean_trans = compute_mean_transitions(tree, gain_rate, loss_rate)
        
        # Mean branch length is 1.0, so mean transitions = total_rate * 1.0
        assert np.isclose(mean_trans, total_rate)
    
    def test_varying_branch_lengths(self):
        """Test with varying branch lengths."""
        newick = "((A:0.5,B:2.0):1.0,(C:0.5,D:0.5):1.0);"
        tree = TreeStructure.from_newick(newick, backend="simple")
        
        gain_rate = 1.0
        loss_rate = 1.0
        total_rate = 2.0
        
        mean_trans = compute_mean_transitions(tree, gain_rate, loss_rate)
        
        # Should be proportional to mean branch length
        mean_bl = np.mean(tree.branch_lengths[tree.branch_lengths > 0])
        expected = total_rate * mean_bl
        
        assert np.isclose(mean_trans, expected)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_tip_tree(self):
        """Test simulation on single-tip tree."""
        newick = "A:1.0;"
        tree = TreeStructure.from_newick(newick, backend="simple")
        rng = np.random.default_rng(42)
        
        matrix = simulate_binary_evolution(
            tree=tree,
            gain_rate=1.0,
            loss_rate=1.0,
            n_sites=10,
            rng=rng
        )
        
        assert matrix.shape == (1, 10)
        assert np.all((matrix == 0) | (matrix == 1))
    
    def test_very_long_branches(self):
        """Test with very long branches (should reach equilibrium)."""
        newick = "((A:100.0,B:100.0):100.0,(C:100.0,D:100.0):100.0);"
        tree = TreeStructure.from_newick(newick, backend="simple")
        rng = np.random.default_rng(42)
        
        gain_rate = 1.5
        loss_rate = 2.0
        expected_pi1 = gain_rate / (gain_rate + loss_rate)
        
        matrix = simulate_binary_evolution(
            tree=tree,
            gain_rate=gain_rate,
            loss_rate=loss_rate,
            n_sites=1000,
            rng=rng
        )
        
        # Should be at equilibrium
        observed_pi1 = matrix.mean()
        assert abs(observed_pi1 - expected_pi1) < 0.05
    
    def test_very_short_branches(self):
        """Test with very short branches (minimal evolution)."""
        newick = "((A:0.001,B:0.001):0.001,(C:0.001,D:0.001):0.001);"
        tree = TreeStructure.from_newick(newick, backend="simple")
        rng = np.random.default_rng(42)
        
        matrix = simulate_binary_evolution(
            tree=tree,
            gain_rate=1.0,
            loss_rate=1.0,
            n_sites=100,
            rng=rng,
            root_frequencies=(0.0, 1.0)  # Start all present
        )
        
        # With very short branches, most should stay present
        assert matrix.mean() > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
