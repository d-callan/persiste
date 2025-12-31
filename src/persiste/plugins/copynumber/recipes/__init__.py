"""
Analysis recipes for CopyNumberDynamics plugin.

These recipes provide standard workflows for common CN evolution questions.
Each recipe is self-contained and produces interpretable results.
"""

from .recipe_0_null import null_cn_dynamics
from .recipe_1_dosage_stability import dosage_stability_scan
from .recipe_2_amplification_bias import amplification_bias_test
from .recipe_3_lineage_volatility import lineage_volatility_test

__all__ = [
    'null_cn_dynamics',
    'dosage_stability_scan',
    'amplification_bias_test',
    'lineage_volatility_test',
]
