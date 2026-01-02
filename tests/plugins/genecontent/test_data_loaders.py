from pathlib import Path

import numpy as np

from persiste.plugins.genecontent.data.loaders import (
    load_gene_matrix,
    load_metadata,
)


def write_tmp(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def test_load_gene_matrix_supports_varied_truthy_values(tmp_path):
    tsv = tmp_path / "matrix.tsv"
    write_tmp(
        tsv,
        """
        taxon\tOG1\tOG2\tOG3
        A\t1\t0\tpresent
        B\ttrue\tno\t0
        C\tyes\t1\t1
        """,
    )

    observations = load_gene_matrix(tsv)

    assert observations.n_taxa == 3
    assert observations.family_ids == ["OG1", "OG2", "OG3"]
    assert observations.taxon_ids == ["A", "B", "C"]

    matrix = observations.to_matrix()
    expected = np.array(
        [
            [1, 0, 1],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(matrix, expected)


def test_load_metadata_parses_fields_and_preserves_whitespace(tmp_path):
    metadata_file = tmp_path / "metadata.tsv"
    write_tmp(
        metadata_file,
        """
        taxon\thost\tniche
        A\thuman\tblood
        B\tmosquito\tgut
        """,
    )

    metadata = load_metadata(metadata_file)

    assert metadata == {
        "A": {"host": "human", "niche": "blood"},
        "B": {"host": "mosquito", "niche": "gut"},
    }
