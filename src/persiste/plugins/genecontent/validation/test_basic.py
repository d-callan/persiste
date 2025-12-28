#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Basic validation test for GeneContent plugin."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

import numpy as np
from persiste.core.trees import TreeStructure
from persiste.plugins.genecontent.inference.gene_inference import (
    GeneContentData,
    GeneContentModel,
    GeneContentInference,
)
from persiste.plugins.genecontent.constraints.gene_constraint import (
    NullConstraint,
    PerFamilyConstraint,
)

def test_basic_likelihood():
    """Test basic likelihood computation."""
    print("Test 1: Basic likelihood computation")
    
    newick = "(A:1.0,B:1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    data = np.array([[1], [0]])
    taxon_names = ["A", "B"]
    
    gene_data = GeneContentData(
        tree=tree,
        presence_matrix=data,
        taxon_names=taxon_names,
        family_names=["fam1"],
    )
    
    constraint = NullConstraint()
    model = GeneContentModel(
        data=gene_data,
        constraint=constraint,
        gain_rate=1.0,
        loss_rate=2.0,
    )
    
    log_lik = model.log_likelihood()
    print(f"  Log-likelihood: {log_lik:.6f}")
    print("  PASS")
    return True

def test_mle_fitting():
    """Test MLE parameter fitting."""
    print("\nTest 2: MLE parameter fitting")
    
    newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    
    # Create simple data
    data = np.array([
        [1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0],
    ])
    taxon_names = ["A", "B", "C", "D"]
    family_names = [f"fam{i}" for i in range(5)]
    
    gene_data = GeneContentData(
        tree=tree,
        presence_matrix=data,
        taxon_names=taxon_names,
        family_names=family_names,
    )
    
    constraint = NullConstraint()
    inference = GeneContentInference(gene_data, constraint)
    
    result = inference.fit_mle(verbose=False)
    print(f"  Gain rate: {result.gain_rate:.4f}")
    print(f"  Loss rate: {result.loss_rate:.4f}")
    print(f"  Log-likelihood: {result.log_likelihood:.4f}")
    print("  PASS")
    return True

def test_constraint_fitting():
    """Test constraint parameter fitting."""
    print("\nTest 3: Constraint parameter fitting")
    
    newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    
    data = np.array([
        [1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0],
    ])
    taxon_names = ["A", "B", "C", "D"]
    family_names = [f"fam{i}" for i in range(5)]
    
    gene_data = GeneContentData(
        tree=tree,
        presence_matrix=data,
        taxon_names=taxon_names,
        family_names=family_names,
    )
    
    constrained_families = ["fam0", "fam1"]
    constraint = PerFamilyConstraint(constrained_families)
    inference = GeneContentInference(gene_data, constraint)
    
    result = inference.fit_mle(verbose=False)
    print(f"  Gain rate: {result.gain_rate:.4f}")
    print(f"  Loss rate: {result.loss_rate:.4f}")
    print(f"  Constraint theta: {result.constraint_params[0]:.4f}")
    print(f"  Log-likelihood: {result.log_likelihood:.4f}")
    print("  PASS")
    return True

def test_likelihood_ratio_test():
    """Test likelihood ratio test."""
    print("\nTest 4: Likelihood ratio test")
    
    newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    
    data = np.array([
        [1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0],
    ])
    taxon_names = ["A", "B", "C", "D"]
    family_names = [f"fam{i}" for i in range(5)]
    
    gene_data = GeneContentData(
        tree=tree,
        presence_matrix=data,
        taxon_names=taxon_names,
        family_names=family_names,
    )
    
    constrained_families = ["fam0", "fam1"]
    constraint = PerFamilyConstraint(constrained_families)
    inference = GeneContentInference(gene_data, constraint)
    
    lrt_result = inference.fit_and_test(verbose=False)
    print(f"  LR statistic: {lrt_result.lr_statistic:.4f}")
    print(f"  p-value: {lrt_result.p_value:.4f}")
    print(f"  Significant: {lrt_result.is_significant}")
    print("  PASS")
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("GENECONTENT PLUGIN BASIC VALIDATION")
    print("=" * 70)
    print()
    
    tests = [
        test_basic_likelihood,
        test_mle_fitting,
        test_constraint_fitting,
        test_likelihood_ratio_test,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    sys.exit(0 if failed == 0 else 1)
