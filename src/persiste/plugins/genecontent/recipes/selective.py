"""Constraint-focused analysis recipes for gene content evolution.

These functions test selective hypotheses by pairing constraints with
likelihood-ratio comparisons against the null model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constraints.gene_constraint import (
    EnvironmentalGradientConstraint,
    GeneContentConstraint,
    PathwayCoherenceConstraint,
)
from ..inference.gene_inference import BaselineDiagnostics, ComparisonResult
from ..pam_interface import PAMAnalysisResult, fit


@dataclass
class ConstraintTestResult:
    """Result of a formal constraint hypothesis test."""

    comparison: ComparisonResult
    null_result: PAMAnalysisResult
    alt_result: PAMAnalysisResult

    def print_summary(self):
        """Print test summary using ComparisonResult's guidance."""
        self.comparison.print_report()
        print("\nEstimated Parameters:")
        params = {
            k: v
            for k, v in self.comparison.alt_result.parameters.items()
            if k not in ("log_gain", "log_loss")
        }
        for name, val in params.items():
            print(f"  {name}: {val:.4f}")
        print("=" * 70)


def run_baseline_diagnostics(
    pam: np.ndarray,
    taxon_names: list[str],
    gene_names: list[str],
    tree_method: str = "jaccard_upgma",
) -> BaselineDiagnostics:
    """Check if the input data and baseline model estimates are trustworthy."""

    res = fit(pam, taxon_names=taxon_names, gene_names=gene_names, tree_method=tree_method)
    return res.inference.get_baseline_diagnostics()


def test_selective_hypothesis(
    pam: np.ndarray,
    taxon_names: list[str],
    gene_names: list[str],
    constraint: GeneContentConstraint,
    tree_method: str = "jaccard_upgma",
    verbose: bool = True,
) -> ConstraintTestResult:
    """Compare a constrained model against the null using an LRT."""

    res = fit(
        pam=pam,
        taxon_names=taxon_names,
        gene_names=gene_names,
        tree_method=tree_method,
        verbose=False,
    )

    comparison = res.inference.compare_to_null(constraint, verbose=verbose)

    alt_res = fit(
        pam=pam,
        taxon_names=taxon_names,
        gene_names=gene_names,
        tree_method=tree_method,
        constraint=constraint,
        verbose=False,
    )

    return ConstraintTestResult(
        comparison=comparison,
        null_result=res,
        alt_result=alt_res,
    )


def test_pathway_retention(
    pam: np.ndarray,
    taxon_names: list[str],
    gene_names: list[str],
    pathway_map: dict[str, set[str]],
    tree_method: str = "jaccard_upgma",
) -> ConstraintTestResult:
    """Convenience recipe for testing coordinated pathway retention."""

    constraint = PathwayCoherenceConstraint(pathway_map=pathway_map)
    initial_params = {f"pathway_{path_id}_loss": -0.1 for path_id in pathway_map}
    constraint.set_parameters(initial_params)
    return test_selective_hypothesis(pam, taxon_names, gene_names, constraint, tree_method)


def test_environmental_gradient(
    pam: np.ndarray,
    taxon_names: list[str],
    gene_names: list[str],
    metadata_key: str,
    tree_method: str = "jaccard_upgma",
) -> ConstraintTestResult:
    """Convenience recipe for testing environmental gradient effects."""

    constraint = EnvironmentalGradientConstraint(metadata_key=metadata_key)
    return test_selective_hypothesis(pam, taxon_names, gene_names, constraint, tree_method)


__all__ = [
    "ConstraintTestResult",
    "run_baseline_diagnostics",
    "test_selective_hypothesis",
    "test_pathway_retention",
    "test_environmental_gradient",
]
