import numpy as np
import pandas as pd
import pytest

from persiste.core.trees import TreeStructure
from persiste.plugins.genecontent import pam_interface


def test_load_pam_rows_csv(tmp_path):
    df = pd.DataFrame(
        data=[[1, 0, 1], [0, 1, 1]],
        index=["strainA", "strainB"],
        columns=["gene1", "gene2", "gene3"],
    )
    csv_path = tmp_path / "pam.csv"
    df.to_csv(csv_path)

    pam, taxa, genes = pam_interface.load_pam(csv_path)

    assert pam.shape == (2, 3)
    assert taxa == ["strainA", "strainB"]
    assert genes == ["gene1", "gene2", "gene3"]
    assert pam.dtype == int


def test_load_pam_columns_tsv(tmp_path):
    df = pd.DataFrame(
        data=[[1, 0], [0, 1], [1, 1]],
        index=["gene1", "gene2", "gene3"],
        columns=["strain_alpha", "strain_beta"],
    )
    tsv_path = tmp_path / "pam.tsv"
    df.to_csv(tsv_path, sep="\t")

    pam, taxa, genes = pam_interface.load_pam(tsv_path)

    assert pam.shape == (2, 3)
    assert taxa == ["strain_alpha", "strain_beta"]
    assert genes == ["gene1", "gene2", "gene3"]
    # Verify orientation detection flipped the matrix
    assert np.array_equal(pam, df.values.T.astype(int))


def test_load_pam_invalid_extension(tmp_path):
    path = tmp_path / "pam.json"
    path.write_text("{}")

    with pytest.raises(ValueError, match="Unsupported file format"):
        pam_interface.load_pam(path)


def test_fit_infers_tree_and_returns_metadata():
    pam = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=int,
    )
    taxa = ["A", "B", "C"]
    genes = ["g1", "g2", "g3", "g4"]

    result = pam_interface.fit(
        pam,
        taxon_names=taxa,
        gene_names=genes,
        verbose=False,
    )

    assert result.tree_metadata.source == "inferred"
    assert result.tree_metadata.method == "jaccard_upgma"
    assert result.data.tree.n_tips == len(taxa)
    assert result.data.n_families == len(genes)
    assert result.log_likelihood <= 0  # log-likelihood should be finite


def test_fit_respects_provided_tree():
    pam = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=int,
    )
    taxa = ["X", "Y"]
    genes = ["g1", "g2"]
    tree = TreeStructure.from_newick("(X:0.5,Y:0.5);")

    result = pam_interface.fit(
        pam,
        tree=tree,
        taxon_names=taxa,
        gene_names=genes,
        verbose=False,
    )

    assert result.tree_metadata.source == "provided"
    assert result.data.tree is tree
    assert result.gain_rate > 0
    assert result.loss_rate > 0
