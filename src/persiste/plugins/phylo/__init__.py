"""
PERSISTE Phylogenetics Plugin.

Provides HyPhy-compatible selection analysis using the PERSISTE framework.

Key components:
- CodonStateSpace: 61 sense codons with genetic code metadata
- CodonTransitionGraph: Single-nucleotide codon changes
- MG94Baseline: Muse-Gaut 1994 codon model (synonymous rates)
- PhyloCTMCObservationModel: Phylogenetic likelihood via Felsenstein pruning

Tree handling:
- Uses core.trees.TreeStructure for tree representation
- Deprecated: phylo.data.tree.PhylogeneticTree (use core.trees instead)

Analyses:
- FEL: Fixed Effects Likelihood (site-specific ω)
- SLAC: Single Likelihood Ancestor Counting
- MEME: Mixed Effects Model of Evolution
- FUBAR: Fast Unconstrained Bayesian AppRoximation
- BUSTED: Branch-Site Unrestricted Statistical Test
- aBSREL: Adaptive Branch-Site Random Effects Likelihood
- RELAX: Test for relaxed/intensified selection
"""

from persiste.plugins.phylo.plugin import PhyloPlugin
from persiste.plugins.phylo.states.genetic_code import GeneticCode
from persiste.plugins.phylo.states.codons import CodonStateSpace
from persiste.plugins.phylo.transitions.codon_graph import CodonTransitionGraph
from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
from persiste.plugins.phylo.phylo_interface import (
    PhyloModelConfig,
    build_phylo_components,
    fit_global_omega,
    load_codon_alignment,
    sequences_to_codon_alignment,
)

# Re-export core tree utilities for convenience
from persiste.core.trees import TreeStructure, load_tree

__all__ = [
    "PhyloPlugin",
    "GeneticCode",
    "CodonStateSpace",
    "CodonTransitionGraph",
    "MG94Baseline",
    "TreeStructure",
    "load_tree",
    "PhyloModelConfig",
    "build_phylo_components",
    "fit_global_omega",
    "load_codon_alignment",
    "sequences_to_codon_alignment",
]
