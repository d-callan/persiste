import numpy as np
import pytest

from persiste.core.pruning_rust import compute_likelihoods_batch
from persiste.core import pruning_rust as pruning_mod
from persiste.core.trees import build_star_tree


@pytest.fixture()
def pruning_inputs():
    tree = build_star_tree(["taxon_a", "taxon_b", "taxon_c"], branch_length=0.5)
    presence = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
    ], dtype=np.int8)
    gain_rates = np.array([0.8, 1.2], dtype=float)
    loss_rates = np.array([1.5, 0.7], dtype=float)
    taxon_names = tree.tip_names
    return tree, presence, gain_rates, loss_rates, taxon_names


def test_compute_likelihoods_batch_falls_back_to_numpy(monkeypatch, pruning_inputs):
    tree, presence, gain_rates, loss_rates, taxon_names = pruning_inputs

    expected = np.array([-1.0, -2.0])
    calls = {
        "numpy": 0,
        "rust": 0,
    }

    def fake_numpy(*args, **kwargs):
        calls["numpy"] += 1
        return expected

    def fake_rust(*args, **kwargs):
        calls["rust"] += 1
        return np.array([42.0])

    monkeypatch.setattr(pruning_mod, "RUST_AVAILABLE", False, raising=False)
    monkeypatch.setattr(pruning_mod, "_compute_likelihoods_numpy", fake_numpy)
    monkeypatch.setattr(pruning_mod, "_compute_likelihoods_rust", fake_rust)

    result = compute_likelihoods_batch(
        tree,
        presence,
        gain_rates,
        loss_rates,
        taxon_names,
        use_rust=True,
    )

    np.testing.assert_allclose(result, expected)
    assert calls["numpy"] == 1
    assert calls["rust"] == 0


def test_compute_likelihoods_batch_uses_rust_when_available(monkeypatch, pruning_inputs):
    tree, presence, gain_rates, loss_rates, taxon_names = pruning_inputs

    expected = np.array([-0.1, -0.2])
    calls = {
        "numpy": 0,
        "rust": 0,
    }

    def fake_numpy(*args, **kwargs):
        calls["numpy"] += 1
        return np.array([99.0])

    def fake_rust(*args, **kwargs):
        calls["rust"] += 1
        return expected

    monkeypatch.setattr(pruning_mod, "RUST_AVAILABLE", True, raising=False)
    monkeypatch.setattr(pruning_mod, "_compute_likelihoods_numpy", fake_numpy)
    monkeypatch.setattr(pruning_mod, "_compute_likelihoods_rust", fake_rust)

    result = compute_likelihoods_batch(
        tree,
        presence,
        gain_rates,
        loss_rates,
        taxon_names,
        use_rust=True,
    )

    np.testing.assert_allclose(result, expected)
    assert calls["rust"] == 1
    assert calls["numpy"] == 0
