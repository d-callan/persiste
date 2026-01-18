"""
Tier 1: Automatic Safety Checks for Assembly Plugin.

These checks run automatically during inference to assess whether
constraint inference is interpretable. They are fast, conservative,
and non-blocking.

Safety checks assess interpretability, not truth:
- Passing checks ≠ guaranteed correct inference
- Failing checks ≠ analysis is useless
- They are guardrails, not judges
"""

from persiste.plugins.assembly.safety.baseline_check import (
    BaselineSanityResult,
    check_baseline_sanity,
)
from persiste.plugins.assembly.safety.cache_reliability import (
    CacheReliabilityResult,
    check_cache_reliability,
)
from persiste.plugins.assembly.safety.identifiability import (
    IdentifiabilityResult,
    check_identifiability,
)
from persiste.plugins.assembly.safety.safety_report import (
    SafetyReport,
    run_safety_checks,
)

__all__ = [
    "BaselineSanityResult",
    "check_baseline_sanity",
    "IdentifiabilityResult",
    "check_identifiability",
    "CacheReliabilityResult",
    "check_cache_reliability",
    "SafetyReport",
    "run_safety_checks",
]
