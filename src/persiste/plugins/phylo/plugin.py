"""Phylogenetics plugin for PERSISTE."""

from typing import Dict, Any
from persiste.plugins.base import PluginBase


class PhyloPlugin(PluginBase):
    """
    Phylogenetics plugin providing HyPhy-compatible selection analysis.
    
    Maps HyPhy concepts to PERSISTE abstractions:
    - ω (dN/dS) → θ (constraint parameter)
    - MG94/GY94 → Baseline (synonymous rate model)
    - Codon model → StateSpace + TransitionGraph
    - Phylogenetic likelihood → ObservationModel (CTMC)
    
    The key insight: ω IS θ for the phylogenetics domain.
    - ω = 1: neutral (no constraint)
    - ω < 1: purifying selection (constraint/suppression)
    - ω > 1: positive selection (facilitation)
    """
    
    @property
    def name(self) -> str:
        return "phylo"
    
    @property
    def version(self) -> str:
        return "0.1.0"
    
    @property
    def state_spaces(self) -> Dict[str, type]:
        from persiste.plugins.phylo.states.codons import CodonStateSpace
        from persiste.plugins.phylo.states.nucleotides import NucleotideStateSpace
        
        return {
            "codons": CodonStateSpace,
            "nucleotides": NucleotideStateSpace,
        }
    
    @property
    def baselines(self) -> Dict[str, type]:
        from persiste.plugins.phylo.baselines.mg94 import MG94Baseline
        
        return {
            "MG94": MG94Baseline,
        }
    
    @property
    def analyses(self) -> Dict[str, type]:
        # Will be populated as analyses are implemented
        return {}
    
    @property
    def loaders(self) -> Dict[str, type]:
        # Will be populated with FASTA, NEXUS loaders
        return {}
    
    @property
    def priors(self) -> Dict[str, Any]:
        return {
            "omega": {
                "distribution": "gamma",
                "shape": 1.0,
                "rate": 1.0,
                "description": "Prior on ω (dN/dS)",
            },
            "kappa": {
                "distribution": "lognormal", 
                "mean": 1.0,
                "sd": 1.25,
                "description": "Prior on κ (transition/transversion ratio)",
            },
        }
