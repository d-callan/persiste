import math

import numpy as np
import pytest

from persiste.plugins.genecontent.baselines.gene_baseline import (
    FixedRates,
    GlobalRates,
    HierarchicalRates,
    RateParameters,
)


def test_rate_parameters_reject_negative_rates():
    with pytest.raises(ValueError, match="gain_rate must be non-negative"):
        RateParameters(gain_rate=-0.1, loss_rate=0.2, family_id="fam")

    with pytest.raises(ValueError, match="loss_rate must be non-negative"):
        RateParameters(gain_rate=0.1, loss_rate=-0.2, family_id="fam")


def test_rate_parameters_transition_probability_matches_closed_form():
    params = RateParameters(gain_rate=0.5, loss_rate=0.3, family_id="famA")

    prob_zero = params.transition_probability(0.0)
    np.testing.assert_allclose(prob_zero, np.eye(2))

    t = 2.0
    λ = 0.5
    μ = 0.3
    total = λ + μ
    exp_term = math.exp(-total * t)
    expected = np.array([
        [(μ + λ * exp_term) / total, (λ - λ * exp_term) / total],
        [(μ - μ * exp_term) / total, (λ + μ * exp_term) / total],
    ])

    prob_t = params.transition_probability(t)
    np.testing.assert_allclose(prob_t, expected)


def test_hierarchical_rates_sampling_is_deterministic_with_seed():
    hierarchy = HierarchicalRates(
        mu_gain=-1.0,
        sigma_gain=0.0,
        mu_loss=-2.0,
        sigma_loss=0.0,
        essential_genes={"fam2"},
    )

    rng = np.random.default_rng(0)
    hierarchy._sample_rates(["fam1", "fam2"], rng=rng)

    fam1 = hierarchy.get_rates("fam1")
    np.testing.assert_allclose(fam1.gain_rate, math.exp(-1.0))
    np.testing.assert_allclose(fam1.loss_rate, math.exp(-2.0))

    fam2 = hierarchy.get_rates("fam2")
    assert fam2.loss_rate == pytest.approx(1e-10)


def test_hierarchical_rates_log_prior_matches_manual_calculation():
    hierarchy = HierarchicalRates(
        mu_gain=-1.0,
        sigma_gain=0.5,
        mu_loss=-1.5,
        sigma_loss=0.25,
    )

    hierarchy.set_rates("famA", gain_rate=math.exp(-1.0), loss_rate=math.exp(-1.5))
    hierarchy.set_rates("famB", gain_rate=math.exp(-0.5), loss_rate=math.exp(-2.0))

    log_prior = hierarchy.log_prior()

    expected = 0.0
    for rates in hierarchy.family_rates.values():
        log_gain = math.log(rates.gain_rate + 1e-10)
        log_loss = math.log(rates.loss_rate + 1e-10)
        expected -= 0.5 * ((log_gain - hierarchy.mu_gain) / hierarchy.sigma_gain) ** 2
        expected -= 0.5 * ((log_loss - hierarchy.mu_loss) / hierarchy.sigma_loss) ** 2

    assert log_prior == pytest.approx(expected)


def test_fixed_rates_use_defaults_for_unknown_families():
    baseline = FixedRates(
        rates={"fam_known": (0.2, 0.1)},
        default_gain=0.05,
        default_loss=0.07,
    )

    known = baseline.get_rates("fam_known")
    assert known.gain_rate == pytest.approx(0.2)
    assert known.loss_rate == pytest.approx(0.1)

    unknown = baseline.get_rates("fam_unknown")
    assert unknown.gain_rate == pytest.approx(0.05)
    assert unknown.loss_rate == pytest.approx(0.07)

    assert baseline.n_parameters() == 2 * len(baseline.rates) + 2


def test_global_rates_return_same_parameters_for_all_families():
    baseline = GlobalRates(gain_rate=0.3, loss_rate=0.12)

    fam_a = baseline.get_rates("famA")
    fam_b = baseline.get_rates("famB")

    assert fam_a.gain_rate == pytest.approx(0.3)
    assert fam_a.loss_rate == pytest.approx(0.12)
    assert fam_b.gain_rate == fam_a.gain_rate
    assert fam_b.loss_rate == fam_a.loss_rate
    assert baseline.n_parameters() == 2
