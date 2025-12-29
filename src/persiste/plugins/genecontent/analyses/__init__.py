"""
Standard gene content analyses.

Provides opinionated, easy-to-use analysis recipes for common biological questions.
Think RELAX / BUSTED / aBSREL - named analyses, not just flags.
"""

from .standard_analyses import (
    GeneContentAnalysis,
    GlobalRatesResult,
    RetentionTestResult,
    BranchShiftResult,
    AssociationTestResult,
    ExploratoryScreeningResult,
)

__all__ = [
    'GeneContentAnalysis',
    'GlobalRatesResult',
    'RetentionTestResult',
    'BranchShiftResult',
    'AssociationTestResult',
    'ExploratoryScreeningResult',
]
