"""High-level entry points for GeneContent recipes.

This package exposes two recipe families:

- :mod:`.heterogeneity` — strain-focused diagnostics and stratified regime workflows
- :mod:`.selective` — constraint-based hypothesis tests and baseline diagnostics

Users should import from ``persiste.plugins.genecontent.recipes`` for a single,
self-describing interface, while developers can add new functionality by placing
heterogeneity routines under ``heterogeneity.py`` and selective tests under
``selective.py``.
"""

from __future__ import annotations

from .heterogeneity import (
    HeterogeneityScanResult,
    StratifiedRegimeResult,
    run_heterogeneity_diagnostic,
    strain_heterogeneity_scan,
    stratified_regime_modeling,
)
from .selective import (
    ConstraintTestResult,
    run_baseline_diagnostics,
    test_environmental_gradient,
    test_pathway_retention,
    test_selective_hypothesis,
)

__all__ = [
    "ConstraintTestResult",
    "HeterogeneityScanResult",
    "StratifiedRegimeResult",
    "run_baseline_diagnostics",
    "run_heterogeneity_diagnostic",
    "strain_heterogeneity_scan",
    "stratified_regime_modeling",
    "test_environmental_gradient",
    "test_pathway_retention",
    "test_selective_hypothesis",
]
