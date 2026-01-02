import numpy as np
import pytest

from persiste.core.trees import build_star_tree
from persiste.plugins.copynumber.baselines.cn_baseline import (
    GlobalBaseline,
    HierarchicalBaseline,
    create_baseline,
)
from persiste.plugins.copynumber.constraints.cn_constraints import (
    AmplificationBiasConstraint,
    DosageStabilityConstraint,
    apply_constraint,
    create_constraint,
)
from persiste.plugins.copynumber.data.cn_data import CopyNumberData
from persiste.plugins.copynumber.observation.cn_observation import (
    DeterministicBinObservation,
    create_observation_model,
)
from persiste.plugins.copynumber.cn_interface import compute_family_likelihood


class TestBaselines:
    def test_create_baseline_global(self):
        baseline = create_baseline(
            'global', gain_rate=0.2, loss_rate=0.3, amplify_rate=0.1, contract_rate=0.05
        )
        assert isinstance(baseline, GlobalBaseline)
        Q = baseline.build_rate_matrix()
        # Off-diagonal rates should match inputs
        assert pytest.approx(Q[0, 1]) == 0.2
        assert pytest.approx(Q[1, 0]) == 0.3
        assert pytest.approx(Q[1, 2]) == 0.1
        # Rows should sum to zero (diagonal negative)
        assert pytest.approx(Q[0].sum()) == 0.0
        assert Q[0, 0] < 0

    def test_hierarchical_requires_sampling(self):
        baseline = HierarchicalBaseline(sigma=0.2)
        with pytest.raises(ValueError, match="family_idx required"):
            baseline.get_baseline_rates()
        baseline.sample_family_rates(n_families=2, rng=np.random.default_rng(0))
        rates = baseline.get_baseline_rates(family_idx=1)
        assert (0, 1) in rates and rates[(0, 1)] > 0


class TestConstraints:
    def test_constraint_factory_and_apply(self):
        baseline = create_baseline('global')
        Q_base = baseline.build_rate_matrix()
        constraint = create_constraint('dosage_stability')
        theta = -0.5
        Q_constrained = apply_constraint(Q_base, constraint, theta)
        multiplier = np.exp(theta)
        assert pytest.approx(Q_constrained[1, 2]) == pytest.approx(Q_base[1, 2] * multiplier)
        # Gain/loss also affected for dosage stability
        assert pytest.approx(Q_constrained[0, 1]) == pytest.approx(Q_base[0, 1] * multiplier)
        # Diagonals rebalanced
        assert pytest.approx(Q_constrained[1].sum()) == 0.0

    def test_amplification_bias_directionality(self):
        baseline = create_baseline('global')
        Q_base = baseline.build_rate_matrix()
        constraint = AmplificationBiasConstraint()
        theta = 0.4
        Q_constrained = apply_constraint(Q_base, constraint, theta)
        amp_mult = np.exp(theta)
        contract_mult = np.exp(-theta)
        assert pytest.approx(Q_constrained[1, 2]) == pytest.approx(Q_base[1, 2] * amp_mult)
        assert pytest.approx(Q_constrained[2, 1]) == pytest.approx(Q_base[2, 1] * contract_mult)
        # Transitions not mentioned remain equal
        assert pytest.approx(Q_constrained[0, 1]) == pytest.approx(Q_base[0, 1])


class TestObservationAndData:
    def test_deterministic_observation_one_hot(self):
        obs = DeterministicBinObservation()
        likelihood = obs.get_tip_likelihood(2)
        assert np.all(likelihood >= 0)
        assert likelihood[2] == 1.0
        assert likelihood.sum() == 1.0
        matrix = obs.get_tip_likelihoods_matrix(np.array([0, 1, 2]))
        assert matrix.shape == (3, 4)
        assert np.all(matrix[np.arange(3), np.array([0, 1, 2])] == 1.0)

    def test_copy_number_data_validation(self):
        cn_matrix = np.array([[0, 1], [2, 3]], dtype=int)
        data = CopyNumberData(
            cn_matrix=cn_matrix,
            family_names=['fam1', 'fam2'],
            taxon_names=['tax1', 'tax2'],
            ploidy=2,
        )
        assert data.n_families == 2
        assert data.n_taxa == 2
        np.testing.assert_array_equal(data.get_family_data(1), np.array([2, 3]))
        stats = data.summary_statistics()
        assert stats['n_families'] == 2 and stats['n_taxa'] == 2

        with pytest.raises(ValueError, match="cn_matrix values"):
            CopyNumberData(
                cn_matrix=np.array([[0, 4]]),
                family_names=['fam'],
                taxon_names=['tax1', 'tax2'],
            )

        with pytest.raises(ValueError, match="family_names"):
            CopyNumberData(
                cn_matrix=cn_matrix,
                family_names=['fam1'],
                taxon_names=['tax1', 'tax2'],
            )


class TestComputeFamilyLikelihood:
    def test_compute_family_likelihood_simple_tree(self):
        tree = build_star_tree(['tax1', 'tax2'], branch_length=0.1)
        baseline = create_baseline('global', gain_rate=0.2, loss_rate=0.2, amplify_rate=0.1, contract_rate=0.1)
        Q = baseline.build_rate_matrix()
        obs_model = DeterministicBinObservation()
        family_data = np.array([1, 1])
        ll = compute_family_likelihood(family_data, tree, Q, obs_model)
        assert np.isfinite(ll)
        assert ll <= 0


class TestLikelihoodRatio:
    class DummyResult:
        def __init__(self, ll, params):
            self.log_likelihood = ll
            self.n_params = params
            self.baseline_type = 'global'
            self.constraint_type = None
            self.theta = None

        def compare_to(self, other):
            delta_ll = self.log_likelihood - other.log_likelihood
            delta_params = self.n_params - other.n_params
            from scipy.stats import chi2
            lrt = 2 * delta_ll
            p_value = 1 - chi2.cdf(lrt, df=delta_params)
            return {
                'delta_log_likelihood': delta_ll,
                'delta_params': delta_params,
                'delta_aic': 0.0,
                'delta_bic': 0.0,
                'lrt_statistic': lrt,
                'p_value': p_value,
            }

    def test_likelihood_ratio_wrapper(self):
        from persiste.plugins.copynumber.cn_interface import likelihood_ratio_test

        null = self.DummyResult(ll=-100.0, params=4)
        alt = self.DummyResult(ll=-90.0, params=5)
        stats = likelihood_ratio_test(alt, null, verbose=False)
        assert stats['delta_log_likelihood'] == 10.0
        assert stats['lrt_statistic'] == 20.0
        assert stats['p_value'] < 0.01
