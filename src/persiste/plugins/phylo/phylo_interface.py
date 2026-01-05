"""
High-level helpers for running phylogenetic inference via ConstraintInference.

These utilities mirror the assembly interface pattern: they wire together the
codon state space, MG94 baseline, PhyloCTMC observation model, and the shared
ConstraintInference engine so callers can focus on inputs (tree + alignment)
instead of low-level plumbing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from persiste.core.data import ObservedTransitions
from persiste.core.inference import ConstraintInference, ConstraintResult
from persiste.core.trees import TreeStructure, load_tree
from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
from persiste.plugins.phylo.constraints.omega_constraint import OmegaConstraint
from persiste.plugins.phylo.observation.phylo_ctmc import PhyloCTMCObservationModel
from persiste.plugins.phylo.states.codons import CodonStateSpace
from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph

__all__ = [
    "PhyloModelConfig",
    "PhyloData",
    "build_phylo_components",
    "fit_global_omega",
    "load_tree",
    "load_codon_alignment",
    "sequences_to_codon_alignment",
]


@dataclass
class PhyloModelConfig:
    """Configuration for codon-level phylogenetic inference."""

    genetic_code: str = "Universal"
    kappa: float = 1.0
    initial_omega: float = 1.0


@dataclass
class PhyloData:
    """Simple container bundling a parsed tree and codon alignment."""

    tree: TreeStructure
    alignment: np.ndarray
    taxa: Sequence[str]


def sequences_to_codon_alignment(
    sequences: Mapping[str, str],
    codon_space: CodonStateSpace,
    taxa_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert codon FASTA sequences into the integer alignment used by MG94 models.

    Args:
        sequences: Mapping from taxon name to nucleotide sequence (multiples of 3).
        codon_space: CodonStateSpace defining valid codons.
        taxa_order: Optional ordering of taxa (defaults to insertion order).

    Returns:
        (alignment, ordered_taxa) tuple where alignment is (n_taxa, n_sites).
    """

    if not sequences:
        raise ValueError("sequences mapping cannot be empty")

    if taxa_order is None:
        taxa_order = list(sequences.keys())

    first_seq = next(iter(sequences.values()))
    if len(first_seq) % 3 != 0:
        raise ValueError("Sequences must have length divisible by 3 (codon alignment).")

    n_sites = len(first_seq) // 3
    alignment = np.zeros((len(taxa_order), n_sites), dtype=int)

    for taxon_idx, taxon in enumerate(taxa_order):
        try:
            seq = sequences[taxon].upper()
        except KeyError as exc:
            raise KeyError(f"Taxon '{taxon}' missing from sequences mapping.") from exc

        if len(seq) != len(first_seq):
            raise ValueError(f"Sequence length mismatch for taxon '{taxon}'.")

        if len(seq) % 3 != 0:
            raise ValueError(f"Sequence length for '{taxon}' is not divisible by 3.")

        for site_idx in range(n_sites):
            codon = seq[site_idx * 3 : site_idx * 3 + 3]
            alignment[taxon_idx, site_idx] = codon_space.index(codon)

    return alignment, list(taxa_order)


def load_codon_alignment(
    fasta_path: str | Path,
    *,
    codon_space: CodonStateSpace,
    taxa_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Load a codon alignment from FASTA and convert it into codon indices.

    Args:
        fasta_path: Path to codon-aligned FASTA file.
        codon_space: CodonStateSpace defining codon indices.
        taxa_order: Optional ordering; defaults to FASTA order.

    Returns:
        (alignment, taxa) tuple ready for PhyloCTMCObservationModel.
    """

    sequences: Dict[str, str] = {}
    current_label: str | None = None
    fasta_path = Path(fasta_path)

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_label = line[1:].strip()
                if not current_label:
                    raise ValueError("Encountered FASTA entry with empty label.")
                if current_label in sequences:
                    raise ValueError(f"Duplicate FASTA label '{current_label}'.")
                sequences[current_label] = ""
                continue

            if current_label is None:
                raise ValueError("FASTA content missing label before sequence data.")
            sequences[current_label] += line.replace(" ", "").upper()

    return sequences_to_codon_alignment(sequences, codon_space, taxa_order)


def build_phylo_components(
    *,
    tree: TreeStructure,
    alignment: np.ndarray,
    config: PhyloModelConfig | None = None,
) -> dict[str, Any]:
    """
    Construct codon space, graph, baseline, and observation model objects.
    """

    cfg = config or PhyloModelConfig()
    codon_space = CodonStateSpace.from_genetic_code(cfg.genetic_code)
    graph = CodonTransitionGraph(codon_space)
    baseline = MG94Baseline(
        codon_space=codon_space,
        graph=graph,
        kappa=cfg.kappa,
        omega=cfg.initial_omega,
    )
    obs_model = PhyloCTMCObservationModel(graph=graph, tree=tree, alignment=alignment)

    return {
        "codon_space": codon_space,
        "graph": graph,
        "baseline": baseline,
        "observation_model": obs_model,
    }


def fit_global_omega(
    *,
    tree: TreeStructure,
    alignment: np.ndarray,
    config: PhyloModelConfig | None = None,
    inference_kwargs: dict[str, Any] | None = None,
) -> ConstraintResult:
    """
    Fit a single ω (global dN/dS) parameter using ConstraintInference.

    Args:
        tree: TreeStructure describing phylogeny with branch lengths.
        alignment: (n_taxa, n_sites) codon index matrix matching tree tip order.
        config: Optional PhyloModelConfig overriding genetic code/kappa/initial ω.
        inference_kwargs: Extra arguments forwarded to ConstraintInference.fit().

    Returns:
        ConstraintResult with the fitted ω and diagnostics.
    """

    components = build_phylo_components(tree=tree, alignment=alignment, config=config)
    baseline = components["baseline"]
    graph = components["graph"]
    obs_model = components["observation_model"]
    cfg = config or PhyloModelConfig()

    constraint = OmegaConstraint(
        baseline=baseline,
        graph=graph,
        omega=cfg.initial_omega,
    )

    engine = ConstraintInference(constraint, obs_model)
    data = ObservedTransitions(counts={}, exposure=1.0)

    kwargs = dict(inference_kwargs or {})
    return engine.fit(data, method="MLE", **kwargs)
