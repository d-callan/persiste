import math
import pytest

from persiste.plugins.genecontent.constraints.gene_constraint import (
    ConstraintEffect,
    EnvironmentalGradientConstraint,
    GenomicClusterConstraint,
    NullConstraint,
    PathwayCoherenceConstraint,
    PerFamilyConstraint,
    RetentionBiasConstraint,
)


def test_constraint_effect_properties():
    effect = ConstraintEffect(gain_effect=math.log(2.0), loss_effect=math.log(0.5), family_id="fam")
    assert effect.gain_multiplier == pytest.approx(2.0)
    assert effect.loss_multiplier == pytest.approx(0.5)
    assert not effect.is_neutral()
    assert ConstraintEffect().is_neutral()


def test_null_constraint_returns_neutral_effect_and_zero_params():
    constraint = NullConstraint()
    effect = constraint.get_effect("famA")
    assert effect.is_neutral()
    assert constraint.get_parameters() == {}
    assert constraint.n_parameters() == 0


def test_per_family_constraint_round_trip_parameters():
    constraint = PerFamilyConstraint(
        effects={
            "fam1": (0.1, -0.2),
            "fam2": (0.0, 0.0),
        },
        regularization=0.5,
    )

    effect_fam1 = constraint.get_effect("fam1")
    assert effect_fam1.gain_effect == pytest.approx(0.1)
    assert effect_fam1.loss_effect == pytest.approx(-0.2)

    effect_unknown = constraint.get_effect("unseen")
    assert effect_unknown.is_neutral()

    params = constraint.get_parameters()
    assert params == {"fam1_gain": 0.1, "fam1_loss": -0.2, "fam2_gain": 0.0, "fam2_loss": 0.0}

    constraint.set_parameters({"fam3_gain": 0.5, "fam3_loss": -0.1})
    effect_fam3 = constraint.get_effect("fam3")
    assert effect_fam3.gain_effect == pytest.approx(0.5)
    assert effect_fam3.loss_effect == pytest.approx(-0.1)

    penalty = constraint.log_prior()
    expected_penalty = -0.5 * constraint.regularization * sum(
        gain ** 2 + loss ** 2 for gain, loss in constraint.effects.values()
    )
    assert penalty == pytest.approx(expected_penalty)


def test_retention_bias_constraint_applies_loss_reduction():
    retained = {"famA", "famB"}
    constraint = RetentionBiasConstraint(retained_families=retained, retention_strength=-1.2)

    retained_effect = constraint.get_effect("famA")
    assert retained_effect.loss_effect == pytest.approx(-1.2)
    assert retained_effect.gain_effect == 0.0

    neutral_effect = constraint.get_effect("famX")
    assert neutral_effect.is_neutral()

    params = constraint.get_parameters()
    assert params == {"retention_strength": -1.2}

    constraint.set_parameters({"retention_strength": -0.7})
    assert constraint.retention_strength == pytest.approx(-0.7)
    assert constraint.n_parameters() == 1

    log_prior = constraint.log_prior()
    z = (constraint.retention_strength - constraint.prior_mean) / constraint.prior_std
    expected = -0.5 * z ** 2 - math.log(constraint.prior_std * math.sqrt(2 * math.pi))
    assert log_prior == pytest.approx(expected)


def test_pathway_coherence_constraint():
    path_map = {"path1": {"famA", "famB"}, "path2": {"famB", "famC"}}
    effs = {"path1": (0.5, -0.5), "path2": (0.1, -0.1)}
    constraint = PathwayCoherenceConstraint(pathway_map=path_map, pathway_effects=effs)

    # famA is in path1
    eff_a = constraint.get_effect("famA")
    assert eff_a.gain_effect == pytest.approx(0.5)
    assert eff_a.loss_effect == pytest.approx(-0.5)

    # famB is in both path1 and path2
    eff_b = constraint.get_effect("famB")
    assert eff_b.gain_effect == pytest.approx(0.6)
    assert eff_b.loss_effect == pytest.approx(-0.6)

    # Round trip parameters
    params = constraint.get_parameters()
    assert params["pathway_path1_gain"] == 0.5
    constraint.set_parameters({"pathway_path1_gain": 0.8})
    assert constraint.pathway_effects["path1"][0] == 0.8


def test_environmental_gradient_constraint():
    constraint = EnvironmentalGradientConstraint(
        metadata_key="temp", gain_slope=0.1, loss_slope=-0.2, reference_value=20.0
    )

    # Test effect at 30 degrees (diff = 10)
    context = {"temp": 30.0}
    eff = constraint.get_effect("famA", context=context)
    assert eff.gain_effect == pytest.approx(1.0)
    assert eff.loss_effect == pytest.approx(-2.0)

    # Test parameters
    params = constraint.get_parameters()
    assert params == {"gain_slope": 0.1, "loss_slope": -0.2}


def test_genomic_cluster_constraint():
    clusters = {"cluster1": {"fam1", "fam2"}}
    constraint = GenomicClusterConstraint(clusters=clusters, coordinated_loss_bias=-0.8)

    eff_fam1 = constraint.get_effect("fam1")
    assert eff_fam1.loss_effect == pytest.approx(-0.8)

    eff_neutral = constraint.get_effect("famX")
    assert eff_neutral.is_neutral()
