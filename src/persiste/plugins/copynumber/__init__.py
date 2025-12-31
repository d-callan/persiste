"""
CopyNumberDynamics Plugin

Models gene family copy number evolution as a small-state CTMC.

Biological Question:
    How do different lineages regulate gene dosage over evolutionary time?

Key Features:
    - Binned copy number states (absent, single, low-multi, high-multi)
    - Sparse transition graph (gradual dosage change)
    - Hierarchical baseline rates (per-family variation)
    - Biologically meaningful constraints:
        * Dosage stability (buffering vs volatility)
        * Amplification bias (adaptive CNV)
        * Host-conditioned volatility (lineage-specific)

Design Philosophy:
    - Complements GeneContent (presence/absence)
    - Captures dosage regulation, not just retention
    - Reuses PersiSTE inference machinery
    - Statistically honest (no spurious volatility)

Example:
    >>> from persiste.plugins.copynumber import fit
    >>> result = fit(
    ...     cn_matrix=cn_data,
    ...     family_names=families,
    ...     taxon_names=taxa,
    ...     tree=tree,
    ...     constraint_type='dosage_stability'
    ... )
    >>> result.print_summary()
"""

from persiste.plugins.copynumber.states.cn_states import CopyNumberState
from persiste.plugins.copynumber.cn_interface import fit

__all__ = [
    'CopyNumberState',
    'fit',
]
