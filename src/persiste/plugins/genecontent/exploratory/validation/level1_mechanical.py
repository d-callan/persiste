#!/usr/bin/env python
"""Level 1: Mechanical Correctness Tests"""

import numpy as np
from scipy.linalg import expm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from persiste.core.trees import TreeStructure
from persiste.plugins.genecontent.inference.gene_inference import (
    GeneContentData,
    GeneContentModel,
)
from persiste.plugins.genecontent.constraints.gene_constraint import NullConstraint


class MechanicalTests:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def log(self, message, status="INFO"):
        prefix = {"PASS": "[PASS]", "FAIL": "[FAIL]", "INFO": "[INFO]"}[status]
        line = "{0} {1}".format(prefix, message)
        print(line)
        self.results.append(line)
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
    
    def test_single_branch_analytical(self):
        self.log("Test 1.1: Likelihood computation works", "INFO")
        
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
        
        gain_rate = 1.0
        loss_rate = 2.0
        
        constraint = NullConstraint()
        model = GeneContentModel(
            data=gene_data,
            constraint=constraint,
        )
        
        parameters = {
            'log_gain': np.log(gain_rate),
            'log_loss': np.log(loss_rate),
        }
        computed_log_lik = model.log_likelihood(parameters)
        
        # Check that likelihood is finite and reasonable
        if np.isfinite(computed_log_lik) and computed_log_lik < 0:
            self.log("  Log-likelihood: {0:.6f}".format(computed_log_lik), "INFO")
            self.log("  Likelihood is finite and negative (correct)", "PASS")
            return True
        else:
            self.log("  Log-likelihood: {0:.6f}".format(computed_log_lik), "INFO")
            self.log("  Likelihood should be finite and negative", "FAIL")
            return False
    
    def test_gain_only_limit(self):
        self.log("Test 1.2: Gain-only limit", "INFO")
        
        newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
        tree = TreeStructure.from_newick(newick, backend="simple")
        data = np.array([[1], [1], [1], [1]])
        taxon_names = ["A", "B", "C", "D"]
        
        gene_data = GeneContentData(
            tree=tree,
            presence_matrix=data,
            taxon_names=taxon_names,
            family_names=["fam1"],
        )
        
        gain_rate = 1.0
        loss_rate = 1e-10
        
        constraint = NullConstraint()
        model = GeneContentModel(
            data=gene_data,
            constraint=constraint,
        )
        
        parameters = {
            'log_gain': np.log(gain_rate),
            'log_loss': np.log(loss_rate),
        }
        log_lik = model.log_likelihood(parameters)
        
        parameters_high_loss = {
            'log_gain': np.log(gain_rate),
            'log_loss': np.log(10.0),
        }
        log_lik_high_loss = model.log_likelihood(parameters_high_loss)
        
        if log_lik > log_lik_high_loss:
            self.log("  Gain-only LL:  {0:.4f}".format(log_lik), "INFO")
            self.log("  High-loss LL:  {0:.4f}".format(log_lik_high_loss), "INFO")
            self.log("  Gain-only has higher likelihood (correct)", "PASS")
            return True
        else:
            self.log("  Gain-only LL:  {0:.4f}".format(log_lik), "INFO")
            self.log("  High-loss LL:  {0:.4f}".format(log_lik_high_loss), "INFO")
            self.log("  Gain-only should have higher likelihood", "FAIL")
            return False
    
    def test_loss_only_limit(self):
        self.log("Test 1.3: Loss-only limit", "INFO")
        
        newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
        tree = TreeStructure.from_newick(newick, backend="simple")
        data = np.array([[0], [0], [0], [0]])
        taxon_names = ["A", "B", "C", "D"]
        
        gene_data = GeneContentData(
            tree=tree,
            presence_matrix=data,
            taxon_names=taxon_names,
            family_names=["fam1"],
        )
        
        gain_rate = 1e-10
        loss_rate = 1.0
        
        constraint = NullConstraint()
        model = GeneContentModel(
            data=gene_data,
            constraint=constraint,
        )
        
        parameters = {
            'log_gain': np.log(gain_rate),
            'log_loss': np.log(loss_rate),
        }
        log_lik = model.log_likelihood(parameters)
        
        parameters_high_gain = {
            'log_gain': np.log(10.0),
            'log_loss': np.log(loss_rate),
        }
        log_lik_high_gain = model.log_likelihood(parameters_high_gain)
        
        if log_lik > log_lik_high_gain:
            self.log("  Loss-only LL:  {0:.4f}".format(log_lik), "INFO")
            self.log("  High-gain LL:  {0:.4f}".format(log_lik_high_gain), "INFO")
            self.log("  Loss-only has higher likelihood (correct)", "PASS")
            return True
        else:
            self.log("  Loss-only LL:  {0:.4f}".format(log_lik), "INFO")
            self.log("  High-gain LL:  {0:.4f}".format(log_lik_high_gain), "INFO")
            self.log("  Loss-only should have higher likelihood", "FAIL")
            return False
    
    def test_branch_length_scaling(self):
        self.log("Test 1.4: Branch length scaling", "INFO")
        
        newick1 = "(A:1.0,B:1.0);"
        tree1 = TreeStructure.from_newick(newick1, backend="simple")
        
        newick2 = "(A:2.0,B:2.0);"
        tree2 = TreeStructure.from_newick(newick2, backend="simple")
        
        data = np.array([[1], [0]])
        taxon_names = ["A", "B"]
        
        gene_data1 = GeneContentData(
            tree=tree1,
            presence_matrix=data,
            taxon_names=taxon_names,
            family_names=["fam1"],
        )
        
        gene_data2 = GeneContentData(
            tree=tree2,
            presence_matrix=data,
            taxon_names=taxon_names,
            family_names=["fam1"],
        )
        
        gain_rate = 1.0
        loss_rate = 2.0
        
        constraint = NullConstraint()
        
        model1 = GeneContentModel(
            data=gene_data1,
            constraint=constraint,
        )
        
        model2 = GeneContentModel(
            data=gene_data2,
            constraint=constraint,
        )
        
        parameters = {
            'log_gain': np.log(gain_rate),
            'log_loss': np.log(loss_rate),
        }
        log_lik1 = model1.log_likelihood(parameters)
        log_lik2 = model2.log_likelihood(parameters)
        
        if log_lik1 != log_lik2:
            self.log("  Unit branches:    {0:.6f}".format(log_lik1), "INFO")
            self.log("  Doubled branches: {0:.6f}".format(log_lik2), "INFO")
            self.log("  Likelihoods differ (correct)", "PASS")
            return True
        else:
            self.log("  Unit branches:    {0:.6f}".format(log_lik1), "INFO")
            self.log("  Doubled branches: {0:.6f}".format(log_lik2), "INFO")
            self.log("  Likelihoods should differ", "FAIL")
            return False
    
    def run_all(self):
        print("=" * 70)
        print("LEVEL 1: MECHANICAL CORRECTNESS TESTS")
        print("=" * 70)
        print()
        
        self.test_single_branch_analytical()
        print()
        
        self.test_gain_only_limit()
        print()
        
        self.test_loss_only_limit()
        print()
        
        self.test_branch_length_scaling()
        print()
        
        print("=" * 70)
        print("RESULTS: {0} passed, {1} failed".format(self.passed, self.failed))
        print("=" * 70)
        
        return self.passed, self.failed, self.results


if __name__ == "__main__":
    tests = MechanicalTests()
    passed, failed, results = tests.run_all()
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / "level1_mechanical_{0}.txt".format(timestamp)
    
    with open(output_file, "w") as f:
        f.write("\n".join(results))
        f.write("\n\nRESULTS: {0} passed, {1} failed\n".format(passed, failed))
    
    print("\nResults saved to: {0}".format(output_file))
    
    sys.exit(0 if failed == 0 else 1)
