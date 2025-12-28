"""
GeneContent Plugin for PERSISTE.

Detects selective retention, loss, or gain of gene families
relative to a neutral gene-content evolutionary process.

Scientific question (v1):
    Which gene families are preferentially retained, lost, or gained
    relative to a neutral gene-content evolutionary process?
"""

from .states.gene_state import GenePresenceState, GeneFamilyVector
from .baselines.gene_baseline import GeneContentBaseline
from .constraints.gene_constraint import GeneContentConstraint
from .observation.gene_observation import GeneContentObservation
from .data.loaders import load_gene_matrix, load_tree, load_metadata
from .inference.gene_inference import (
    GeneContentData,
    GeneContentModel,
    GeneContentInference,
)

__all__ = [
    'GenePresenceState',
    'GeneFamilyVector',
    'GeneContentBaseline',
    'GeneContentConstraint',
    'GeneContentObservation',
    'load_gene_matrix',
    'load_tree',
    'load_metadata',
    'GeneContentData',
    'GeneContentModel',
    'GeneContentInference',
]
