"""
PERSISTE Phylogenetics Plugin.

Provides HyPhy-compatible selection analysis using the PERSISTE framework.

Key components:
- CodonStateSpace: 61 sense codons with genetic code metadata
- CodonTransitionGraph: Single-nucleotide codon changes
- MG94Baseline: Muse-Gaut 1994 codon model (synonymous rates)
- PhyloCTMCObservationModel: Phylogenetic likelihood via Felsenstein pruning

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

__all__ = [
    "PhyloPlugin",
    "GeneticCode",
    "CodonStateSpace", 
    "CodonTransitionGraph",
    "MG94Baseline",
]
