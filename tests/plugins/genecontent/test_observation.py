import numpy as np
import pytest

from persiste.plugins.genecontent.observation.gene_observation import (
    GeneContentObservationModel,
    TipObservations,
)
from persiste.plugins.genecontent.states.gene_state import GeneFamilyVector


def make_tip_observations():
    observations = {
        "A": GeneFamilyVector({"og1": True, "og2": False, "og3": True}),
        "B": GeneFamilyVector({"og1": False, "og2": True, "og3": False}),
    }
    return TipObservations(
        observations=observations,
        family_ids=["og1", "og2", "og3"],
        taxon_ids=["A", "B"],
    )


def test_tip_observations_to_matrix_round_trip():
    tips = make_tip_observations()
    matrix = tips.to_matrix()
    reconstructed = TipObservations.from_matrix(matrix, tips.taxon_ids, tips.family_ids)
    np.testing.assert_array_equal(reconstructed.to_matrix(), matrix)
    assert reconstructed.taxon_ids == tips.taxon_ids
    assert reconstructed.family_ids == tips.family_ids


def test_tip_observations_summary_calculates_counts():
    tips = make_tip_observations()
    summary = tips.summary()
    assert summary["n_taxa"] == 2
    assert summary["n_families"] == 3
    assert summary["mean_genes_per_taxon"] == pytest.approx(1.5)
    assert summary["mean_taxa_per_gene"] == pytest.approx(1.0)


def test_gene_content_observation_model_runs_numpy():
    tips = make_tip_observations()
    fake_tree = None
    model = GeneContentObservationModel(graph=None, tree=fake_tree, tip_observations=tips, use_rust=False)
    # smoke test for presence matrix cache
    assert model._presence().shape == (tips.n_taxa, tips.n_families)
