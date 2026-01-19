import numpy as np
import pytest
from persiste.plugins.genecontent.recipes import (
    ConstraintTestResult,
    run_heterogeneity_diagnostic,
    test_selective_hypothesis,
)
from persiste.plugins.genecontent.constraints.gene_constraint import RetentionBiasConstraint
from persiste.plugins.genecontent.recipes import HeterogeneityScanResult

@pytest.fixture
def toy_data():
    """Create a small toy PAM and taxon/gene names."""
    # 10 taxa, 50 genes
    np.random.seed(42)
    pam = np.random.binomial(1, 0.2, size=(10, 50))
    taxon_names = [f"strain_{i}" for i in range(10)]
    gene_names = [f"gene_{i}" for i in range(50)]
    return pam, taxon_names, gene_names

def test_run_heterogeneity_diagnostic(toy_data):
    pam, taxa, genes = toy_data
    result = run_heterogeneity_diagnostic(pam, taxa, genes)
    assert isinstance(result, HeterogeneityScanResult)
    assert hasattr(result, "parameter_shifts")
    assert "λ (remove top 10%)" in result.parameter_shifts

def test_test_selective_hypothesis(toy_data):
    pam, taxa, genes = toy_data
    # Create a simple retention constraint
    constraint = RetentionBiasConstraint(
        retained_families={genes[0], genes[1]},
        retention_strength=-1.0
    )

    result = test_selective_hypothesis(pam, taxa, genes, constraint, verbose=False)

    assert isinstance(result, ConstraintTestResult)
    # Access LRT statistic via the comparison object
    # Note: Using max(0.0, ...) to match internal ConstraintTestResult logic if needed,
    # but here we just check it exists and is a number.
    assert hasattr(result.comparison.lrt_result, "statistic")
    assert 0 <= result.comparison.lrt_result.pvalue <= 1.0
    assert result.comparison.lrt_result.df == 1

    # Test summary printing doesn't crash
    result.print_summary()
