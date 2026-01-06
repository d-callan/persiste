from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

import persiste.core.tree_building as tree_building
from persiste.core.tree_building import (
    hamming_distance,
    infer_tree_from_binary_matrix,
    jaccard_distance,
    TreeInferenceMetadata,
    upgma_tree,
)


def test_jaccard_distance_matches_scipy():
    binary = np.array([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 1],
    ])

    expected = squareform(pdist(binary, metric="jaccard"))
    result = jaccard_distance(binary, use_rust=False)

    np.testing.assert_allclose(result, expected)


def test_hamming_distance_matches_scipy():
    binary = np.array([
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 1],
    ])

    expected = squareform(pdist(binary, metric="hamming"))
    result = hamming_distance(binary, use_rust=False)

    np.testing.assert_allclose(result, expected)


def test_upgma_tree_preserves_taxon_names():
    distance_matrix = np.array(
        [
            [0.0, 0.2, 0.4],
            [0.2, 0.0, 0.4],
            [0.4, 0.4, 0.0],
        ]
    )
    taxon_names = ["A_strain", "B_strain", "C_strain"]

    tree = upgma_tree(distance_matrix, taxon_names)

    assert tree.n_tips == 3
    assert set(tree.tip_names) == set(taxon_names)
    assert np.all(tree.branch_lengths >= 0.0)


def test_upgma_tree_tip_branch_lengths_positive_for_non_degenerate_input():
    distance_matrix = np.array(
        [
            [0.0, 0.2, 0.5],
            [0.2, 0.0, 0.4],
            [0.5, 0.4, 0.0],
        ]
    )
    taxon_names = ["A_strain", "B_strain", "C_strain"]

    tree = upgma_tree(distance_matrix, taxon_names)

    tip_branch_lengths = tree.branch_lengths[np.array(tree.tip_indices, dtype=int)]
    assert np.all(np.isfinite(tip_branch_lengths))
    assert np.all(tip_branch_lengths > 0.0)
    assert float(np.sum(tree.branch_lengths)) > 0.0


def test_infer_tree_metadata_records_method():
    rng = np.random.default_rng(0)
    binary = rng.integers(0, 2, size=(5, 20))
    taxon_names = [f"taxon_{i}" for i in range(5)]

    tree, metadata = infer_tree_from_binary_matrix(
        binary,
        taxon_names,
        method="hamming_upgma",
    )

    assert isinstance(tree.tip_names, list)
    assert isinstance(metadata, TreeInferenceMetadata)
    assert metadata.method == "hamming_upgma"
    assert metadata.distance_metric == "hamming"
    assert metadata.n_taxa == 5
    assert metadata.n_features == 20


@pytest.mark.parametrize(
    ("func_name", "metric", "rust_attr"),
    [
        ("jaccard_distance", "jaccard", "compute_jaccard_distance"),
        ("hamming_distance", "hamming", "compute_hamming_distance"),
    ],
)
def test_distance_functions_respect_use_rust_flag(func_name, metric, rust_attr, monkeypatch):
    binary = np.array(
        [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    func = getattr(tree_building, func_name)

    monkeypatch.setattr(tree_building, "RUST_AVAILABLE", True)
    calls = {"rust": 0, "pdist": 0}

    def fake_rust_impl(matrix):
        calls["rust"] += 1
        assert matrix.dtype == np.uint8
        return np.full((matrix.shape[0], matrix.shape[0]), 0.5, dtype=float)

    fake_rust = SimpleNamespace(**{rust_attr: fake_rust_impl})
    monkeypatch.setattr(tree_building, "persiste_rust", fake_rust, raising=False)

    result = func(binary.copy(), use_rust=True)
    assert calls["rust"] == 1
    assert result.shape == (binary.shape[0], binary.shape[0])

    original_pdist = tree_building.pdist
    original_squareform = tree_building.squareform

    def spy_pdist(matrix, metric_arg=None, *args, **kwargs):
        calls["pdist"] += 1
        # pdist may pass metric via positional or keyword argument
        kw_metric = kwargs.pop("metric", None)
        effective_metric = metric_arg if metric_arg is not None else kw_metric
        assert effective_metric == metric
        chosen_metric = effective_metric if effective_metric is not None else metric
        return original_pdist(matrix, chosen_metric, *args, **kwargs)

    def spy_squareform(values):
        return original_squareform(values)

    monkeypatch.setattr(tree_building, "pdist", spy_pdist)
    monkeypatch.setattr(tree_building, "squareform", spy_squareform)

    fallback_result = func(binary.copy(), use_rust=False)
    assert calls["pdist"] == 1
    assert calls["rust"] == 1  # no additional Rust call
    assert fallback_result.shape == (binary.shape[0], binary.shape[0])


def test_neighbor_joining_warns_and_calls_upgma(monkeypatch):
    distance_matrix = np.array(
        [
            [0.0, 0.2, 0.4],
            [0.2, 0.0, 0.4],
            [0.4, 0.4, 0.0],
        ]
    )
    taxon_names = ["A", "B", "C"]
    sentinel = object()

    def fake_upgma(matrix, names):
        np.testing.assert_allclose(matrix, distance_matrix)
        assert names == taxon_names
        return sentinel

    monkeypatch.setattr(tree_building, "upgma_tree", fake_upgma)

    with pytest.warns(UserWarning, match="Neighbor-Joining not yet implemented"):
        result = tree_building.neighbor_joining_tree(distance_matrix, taxon_names)

    assert result is sentinel


def test_infer_tree_neighbor_joining_records_metadata():
    binary = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ]
    )
    taxon_names = ["A", "B", "C", "D"]

    with pytest.warns(UserWarning, match="Neighbor-Joining not yet implemented"):
        tree, metadata = infer_tree_from_binary_matrix(
            binary,
            taxon_names,
            method="jaccard_nj",
        )

    assert tree.n_tips == len(taxon_names)
    assert metadata.method == "jaccard_nj"
    assert metadata.clustering_method == "neighbor_joining"
    assert metadata.distance_metric == "jaccard"
