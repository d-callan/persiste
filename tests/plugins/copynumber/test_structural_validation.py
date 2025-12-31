"""
Tier 1: Structural / Sanity Validation

These tests verify basic mathematical correctness.
Not biological - just "this is not broken".
"""

import pytest
import numpy as np
from scipy.linalg import expm

from persiste.plugins.copynumber.states.cn_states import (
    CopyNumberState,
    get_sparse_transition_graph,
    validate_transition_matrix,
)
from persiste.plugins.copynumber.baselines.cn_baseline import (
    GlobalBaseline,
    HierarchicalBaseline,
)
from persiste.plugins.copynumber.constraints.cn_constraints import (
    DosageStabilityConstraint,
    AmplificationBiasConstraint,
    apply_constraint,
)


class TestRateMatrixIntegrity:
    """A. Rate matrix integrity tests."""
    
    def test_off_diagonal_nonnegative(self):
        """All off-diagonal rates must be ≥ 0."""
        baseline = GlobalBaseline()
        Q = baseline.build_rate_matrix()
        
        # Check off-diagonal
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert Q[i, j] >= 0, f"Q[{i},{j}] = {Q[i,j]} < 0"
    
    def test_row_sums_zero(self):
        """Row sums must equal zero (valid CTMC)."""
        baseline = GlobalBaseline()
        Q = baseline.build_rate_matrix()
        
        row_sums = Q.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0, atol=1e-10)
    
    def test_forbidden_transitions_zero(self):
        """Forbidden transitions must have exactly zero rate."""
        baseline = GlobalBaseline()
        Q = baseline.build_rate_matrix()
        
        allowed = get_sparse_transition_graph()
        
        # Check forbidden transitions
        forbidden = [
            (0, 2), (2, 0),  # absent ↔ low-multi
            (0, 3), (3, 0),  # absent ↔ high-multi
            (1, 3), (3, 1),  # single ↔ high-multi
        ]
        
        for i, j in forbidden:
            assert Q[i, j] == 0, f"Forbidden transition {i}→{j} has rate {Q[i,j]}"
            assert not allowed[(i, j)], f"Transition {i}→{j} incorrectly marked as allowed"
    
    def test_allowed_transitions_positive(self):
        """Allowed transitions must have positive rate."""
        baseline = GlobalBaseline()
        Q = baseline.build_rate_matrix()
        
        allowed_pairs = [
            (0, 1), (1, 0),  # absent ↔ single
            (1, 2), (2, 1),  # single ↔ low-multi
            (2, 3), (3, 2),  # low-multi ↔ high-multi
        ]
        
        for i, j in allowed_pairs:
            assert Q[i, j] > 0, f"Allowed transition {i}→{j} has rate {Q[i,j]}"
    
    def test_hierarchical_baseline_integrity(self):
        """Hierarchical baseline must also produce valid matrices."""
        baseline = HierarchicalBaseline(sigma=0.5)
        rng = np.random.default_rng(42)
        baseline.sample_family_rates(n_families=10, rng=rng)
        
        allowed = get_sparse_transition_graph()
        
        for fam_idx in range(10):
            Q = baseline.build_rate_matrix(family_idx=fam_idx)
            
            # Check validity
            validate_transition_matrix(Q, allowed)
            
            # Check row sums
            row_sums = Q.sum(axis=1)
            np.testing.assert_allclose(row_sums, 0, atol=1e-10)
    
    def test_constraint_preserves_validity(self):
        """Applying constraints must preserve rate matrix validity."""
        baseline = GlobalBaseline()
        Q_base = baseline.build_rate_matrix()
        
        constraint = DosageStabilityConstraint()
        allowed = get_sparse_transition_graph()
        
        # Test various theta values
        for theta in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            Q = apply_constraint(Q_base, constraint, theta)
            
            # Must still be valid
            validate_transition_matrix(Q, allowed)
            
            # Row sums still zero
            row_sums = Q.sum(axis=1)
            np.testing.assert_allclose(row_sums, 0, atol=1e-10)
            
            # Off-diagonal still non-negative
            for i in range(4):
                for j in range(4):
                    if i != j:
                        assert Q[i, j] >= 0, f"Q[{i},{j}] = {Q[i,j]} < 0 at θ={theta}"
    
    def test_transition_matrix_exponential(self):
        """Matrix exponential must produce valid probability matrix."""
        baseline = GlobalBaseline()
        Q = baseline.build_rate_matrix()
        
        # Test at various branch lengths
        for t in [0.1, 0.5, 1.0, 2.0]:
            P = expm(Q * t)
            
            # All entries non-negative
            assert np.all(P >= -1e-10), f"Negative probabilities at t={t}"
            
            # Rows sum to 1
            row_sums = P.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)
            
            # Diagonal dominance for small t
            if t < 0.5:
                for i in range(4):
                    assert P[i, i] > 0.5, f"P[{i},{i}] = {P[i,i]} at t={t}"


class TestLikelihoodMonotonicity:
    """B. Likelihood monotonicity tests."""
    
    def test_theta_zero_recovers_baseline(self):
        """Constraining θ=0 must recover baseline likelihood exactly."""
        baseline = GlobalBaseline()
        Q_base = baseline.build_rate_matrix()
        
        constraint = DosageStabilityConstraint()
        Q_constrained = apply_constraint(Q_base, constraint, theta=0.0)
        
        # Should be identical
        np.testing.assert_allclose(Q_base, Q_constrained, atol=1e-12)
    
    def test_constraint_at_zero_is_neutral(self):
        """All constraint types must be neutral at θ=0."""
        baseline = GlobalBaseline()
        Q_base = baseline.build_rate_matrix()
        
        constraints = [
            DosageStabilityConstraint(),
            AmplificationBiasConstraint(),
        ]
        
        for constraint in constraints:
            Q_constrained = apply_constraint(Q_base, constraint, theta=0.0)
            np.testing.assert_allclose(
                Q_base, Q_constrained, atol=1e-12,
                err_msg=f"{constraint.__class__.__name__} not neutral at θ=0"
            )
    
    def test_multiplier_interpretation(self):
        """Multipliers must follow exp(θ) correctly."""
        baseline = GlobalBaseline(gain_rate=0.1)
        Q_base = baseline.build_rate_matrix()
        
        constraint = DosageStabilityConstraint()
        
        # Test specific theta values
        theta = 0.5
        Q_constrained = apply_constraint(Q_base, constraint, theta)
        
        # Gain rate should be multiplied by exp(0.5)
        expected_gain = 0.1 * np.exp(0.5)
        actual_gain = Q_constrained[0, 1]
        
        np.testing.assert_allclose(actual_gain, expected_gain, rtol=1e-10)


class TestIdentifiabilitySmokeTest:
    """C. Identifiability smoke test."""
    
    def test_gain_vs_amplify_distinct(self):
        """Gain-only vs amplification-only must be distinguishable."""
        baseline = GlobalBaseline(
            gain_rate=0.1,
            loss_rate=0.1,
            amplify_rate=0.05,
            contract_rate=0.05,
        )
        Q_base = baseline.build_rate_matrix()
        
        # Modify only gain
        Q_gain = Q_base.copy()
        Q_gain[0, 1] *= 2.0  # Double gain rate
        Q_gain[0, 0] = -Q_gain[0, :].sum() + Q_gain[0, 0]
        
        # Modify only amplify
        Q_amp = Q_base.copy()
        Q_amp[1, 2] *= 2.0  # Double amplify rate
        Q_amp[1, 1] = -Q_amp[1, :].sum() + Q_amp[1, 1]
        
        # Matrices should be different
        assert not np.allclose(Q_gain, Q_amp), "Gain and amplify changes are confounded"
        
        # Specifically, different transitions affected
        assert Q_gain[0, 1] != Q_base[0, 1], "Gain not modified"
        assert Q_amp[1, 2] != Q_base[1, 2], "Amplify not modified"
        assert Q_gain[1, 2] == Q_base[1, 2], "Amplify incorrectly modified"
        assert Q_amp[0, 1] == Q_base[0, 1], "Gain incorrectly modified"
    
    def test_loss_vs_contract_distinct(self):
        """Loss vs contraction must be distinguishable."""
        baseline = GlobalBaseline()
        Q_base = baseline.build_rate_matrix()
        
        # Modify only loss
        Q_loss = Q_base.copy()
        Q_loss[1, 0] *= 2.0
        Q_loss[1, 1] = -Q_loss[1, :].sum() + Q_loss[1, 1]
        
        # Modify only contract
        Q_contract = Q_base.copy()
        Q_contract[2, 1] *= 2.0
        Q_contract[2, 2] = -Q_contract[2, :].sum() + Q_contract[2, 2]
        
        # Should be different
        assert not np.allclose(Q_loss, Q_contract), "Loss and contract are confounded"
        
        # Different transitions
        assert Q_loss[1, 0] != Q_base[1, 0]
        assert Q_contract[2, 1] != Q_base[2, 1]
        assert Q_loss[2, 1] == Q_base[2, 1]
        assert Q_contract[1, 0] == Q_base[1, 0]
    
    def test_constraint_types_produce_different_patterns(self):
        """Different constraint types must produce distinct rate patterns."""
        baseline = GlobalBaseline()
        Q_base = baseline.build_rate_matrix()
        
        theta = 0.5
        
        # Apply different constraints
        dosage_constraint = DosageStabilityConstraint()
        Q_dosage = apply_constraint(Q_base, dosage_constraint, theta)
        
        amp_constraint = AmplificationBiasConstraint()
        Q_amp = apply_constraint(Q_base, amp_constraint, theta)
        
        # Should be different
        assert not np.allclose(Q_dosage, Q_amp), "Constraints produce identical patterns"
        
        # Dosage affects all transitions
        assert Q_dosage[0, 1] != Q_base[0, 1]  # gain
        assert Q_dosage[1, 0] != Q_base[1, 0]  # loss
        assert Q_dosage[1, 2] != Q_base[1, 2]  # amplify
        
        # Amplification only affects amplify transitions
        assert Q_amp[1, 2] != Q_base[1, 2]  # amplify affected
        assert Q_amp[0, 1] == Q_base[0, 1]  # gain not affected
        assert Q_amp[1, 0] == Q_base[1, 0]  # loss not affected


class TestStateBinning:
    """Test copy number binning logic."""
    
    def test_binning_diploid(self):
        """Test binning for diploid organism."""
        assert CopyNumberState.from_raw_count(0, ploidy=2) == CopyNumberState.ABSENT
        assert CopyNumberState.from_raw_count(1, ploidy=2) == CopyNumberState.SINGLE
        assert CopyNumberState.from_raw_count(2, ploidy=2) == CopyNumberState.SINGLE
        assert CopyNumberState.from_raw_count(3, ploidy=2) == CopyNumberState.LOW_MULTI
        assert CopyNumberState.from_raw_count(4, ploidy=2) == CopyNumberState.LOW_MULTI
        assert CopyNumberState.from_raw_count(5, ploidy=2) == CopyNumberState.LOW_MULTI
        assert CopyNumberState.from_raw_count(6, ploidy=2) == CopyNumberState.HIGH_MULTI
        assert CopyNumberState.from_raw_count(10, ploidy=2) == CopyNumberState.HIGH_MULTI
    
    def test_binning_haploid(self):
        """Test binning for haploid organism."""
        assert CopyNumberState.from_raw_count(0, ploidy=1) == CopyNumberState.ABSENT
        assert CopyNumberState.from_raw_count(1, ploidy=1) == CopyNumberState.SINGLE
        assert CopyNumberState.from_raw_count(2, ploidy=1) == CopyNumberState.LOW_MULTI
        assert CopyNumberState.from_raw_count(3, ploidy=1) == CopyNumberState.LOW_MULTI
        assert CopyNumberState.from_raw_count(4, ploidy=1) == CopyNumberState.LOW_MULTI
        assert CopyNumberState.from_raw_count(5, ploidy=1) == CopyNumberState.HIGH_MULTI
    
    def test_bin_matrix(self):
        """Test matrix binning."""
        raw_counts = np.array([
            [0, 2, 4, 8],
            [1, 3, 5, 10],
        ])
        
        binned = CopyNumberState.bin_matrix(raw_counts, ploidy=2)
        
        expected = np.array([
            [0, 1, 2, 3],
            [1, 2, 2, 3],
        ])
        
        np.testing.assert_array_equal(binned, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
