"""
Tier 2: Optional Deep Diagnostics (Recipes) for Assembly Plugin.

These recipes provide comprehensive robustness analysis for assembly
constraint inference. They are opt-in, expensive, and runnable post-hoc.

Each recipe returns a DiagnosticReport subclass with:
- severity: 'ok', 'warning', 'fail'
- recommendation: Actionable next step
- print_summary(): Human-readable output
"""

from persiste.plugins.assembly.recipes.base import DiagnosticReport
from persiste.plugins.assembly.recipes.recipe_0_null_resampling import (
    NullResamplingReport,
    null_resampling_diagnostic,
)
from persiste.plugins.assembly.recipes.recipe_1_profile_likelihood import (
    ProfileLikelihoodReport,
    profile_likelihood_sweep,
)
from persiste.plugins.assembly.recipes.recipe_2_baseline_perturbation import (
    BaselinePerturbationReport,
    baseline_perturbation_sensitivity,
)

__all__ = [
    "DiagnosticReport",
    "NullResamplingReport",
    "null_resampling_diagnostic",
    "ProfileLikelihoodReport",
    "profile_likelihood_sweep",
    "BaselinePerturbationReport",
    "baseline_perturbation_sensitivity",
]
