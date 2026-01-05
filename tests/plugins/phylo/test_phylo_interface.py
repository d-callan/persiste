"""Tests for the high-level phylogenetic interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from persiste.core.trees import TreeStructure
from persiste.plugins.phylo import (
    CodonStateSpace,
    PhyloModelConfig,
    build_phylo_components,
    fit_global_omega,
    load_codon_alignment,
    sequences_to_codon_alignment,
)


def _example_sequences() -> dict[str, str]:
    return {
        "A": "ATGATGATG",
        "B": "ATGTTTATG",
        "C": "ATGATGTTT",
        "D": "ATGTTTTTT",
    }


def test_sequences_to_codon_alignment_respects_taxon_order():
    codon_space = CodonStateSpace.universal()
    sequences = _example_sequences()
    taxa_order = ["D", "A"]

    alignment, ordered = sequences_to_codon_alignment(sequences, codon_space, taxa_order)

    assert ordered == taxa_order
    assert alignment.shape == (2, 3)
    assert alignment.dtype == np.int_

    d_row = alignment[0]
    a_row = alignment[1]
    assert (d_row == a_row).sum() < d_row.size  # rows should differ


def test_load_codon_alignment_reads_fasta(tmp_path: Path):
    fasta_path = tmp_path / "alignment.fasta"
    sequences = _example_sequences()
    with fasta_path.open("w", encoding="utf-8") as handle:
        for label, seq in sequences.items():
            handle.write(f">{label}\n{seq}\n")

    codon_space = CodonStateSpace.universal()
    alignment, taxa = load_codon_alignment(fasta_path, codon_space=codon_space)

    assert set(taxa) == set(sequences.keys())
    assert alignment.shape == (len(taxa), len(sequences["A"]) // 3)


def test_fit_global_omega_runs_constraint_inference():
    tree = TreeStructure.from_newick(
        "((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);",
        backend="simple",
    )
    codon_space = CodonStateSpace.universal()
    sequences = _example_sequences()
    alignment, taxa = sequences_to_codon_alignment(sequences, codon_space)

    # Ensure tree tip order matches alignment order
    tip_order = tree.tip_names
    reindexed_alignment, _ = sequences_to_codon_alignment(
        sequences,
        codon_space,
        taxa_order=tip_order,
    )

    result = fit_global_omega(
        tree=tree,
        alignment=reindexed_alignment,
        config=PhyloModelConfig(initial_omega=1.0),
        inference_kwargs={"options": {"maxiter": 5}},
    )

    assert "omega" in result.parameters
    assert result.metadata["success"] is True
    assert result.log_likelihood <= 0.0

    components = build_phylo_components(
        tree=tree,
        alignment=reindexed_alignment,
        config=PhyloModelConfig(initial_omega=1.0),
    )
    assert {"codon_space", "graph", "baseline", "observation_model"} <= components.keys()
