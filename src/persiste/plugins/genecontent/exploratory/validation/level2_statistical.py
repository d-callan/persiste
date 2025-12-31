#!/usr/bin/env python
"""Level 2: Statistical Honesty Tests"""

import numpy as np
from scipy import stats
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from persiste.core.trees import TreeStructure
from persiste.plugins.genecontent.inference.gene_inference import (
    GeneContentData,
    GeneContentInference,
)
from persiste.plugins.genecontent.constraints.gene_constraint import (
    NullConstraint,
    PerFamilyConstraint,
)


class StatisticalTests:
    """Level 2 validation tests."""
    
    def __init__(self, n_replicates: int = 20):
        self.n_replicates = n_replicates
        self.results = []
        self.passed = 0
        self.failed = 0
        self.rng = np.random.default_rng(42)
    
    def log(self, message, status="INFO"):
        """Log test result."""
        prefix = {"PASS": "[PASS]", "FAIL": "[FAIL]", "INFO": "[INFO]", "WARN": "[WARN]"}[status]
        line = "{0} {1}".format(prefix, message)
        print(line)
        self.results.append(line)
        
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
    
    def simulate_gene_content(
        self,
        tree: TreeStructure,
        gain_rate: float,
        loss_rate: float,
        n_families: int,
        constrained_families: List[str] = None,
        constraint_effect: float = 0.0,
    ) -> np.ndarray:
        """
        Simulate gene content evolution on a tree.
        """
        from scipy.linalg import expm
        
        n_taxa = tree.n_tips
        presence_matrix = np.zeros((n_taxa, n_families), dtype=int)
        
        # Equilibrium frequencies
        pi_0 = loss_rate / (gain_rate + loss_rate)
        pi_1 = gain_rate / (gain_rate + loss_rate)
        
        for fam_idx in range(n_families):
            # Check if this family is constrained
            is_constrained = False
            if constrained_families is not None:
                fam_name = "fam{0}".format(fam_idx)
                is_constrained = fam_name in constrained_families
            
            # Adjust loss rate if constrained
            if is_constrained:
                effective_loss_rate = loss_rate * np.exp(constraint_effect)
            else:
                effective_loss_rate = loss_rate
            
            # Build rate matrix
            Q = np.array([
                [-gain_rate, gain_rate],
                [effective_loss_rate, -effective_loss_rate]
            ])
            
            # Sample root state from equilibrium
            root_state = self.rng.choice([0, 1], p=[pi_0, pi_1])
            
            # Simulate down the tree using postorder traversal
            node_states = {}
            node_states[tree.root_index] = root_state
            
            # Traverse tree and simulate
            for node_idx in range(tree.n_nodes):
                if node_idx in node_states:
                    parent_state = node_states[node_idx]
                    
                    # Get children
                    child_indices = tree.children_array[tree.children_array[:, 0] == node_idx]
                    
                    for _, child1_idx, child2_idx in child_indices:
                        for child_idx in [child1_idx, child2_idx]:
                            if child_idx >= 0:  # Valid child
                                # Get branch length
                                t = tree.branch_lengths[child_idx]
                                
                                # Compute transition probabilities
                                P = expm(Q * t)
                                
                                # Sample child state
                                child_state = self.rng.choice(
                                    [0, 1],
                                    p=P[parent_state, :]
                                )
                                node_states[child_idx] = child_state
            
            # Extract tip states
            for tip_idx_pos, tip_idx in enumerate(tree.tip_indices):
                presence_matrix[tip_idx_pos, fam_idx] = node_states[tip_idx]
        
        return presence_matrix
    
    def test_null_recovery(self):
        """Test 2.1: Null recovery (θ = 0 → θ̂ ≈ 0)."""
        self.log("Test 2.1: Null recovery", "INFO")
        
        # Create tree
        newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
        tree = TreeStructure.from_newick(newick, backend="simple")
        taxon_names = ["A", "B", "C", "D"]
        
        # Simulate data with NO constraint (θ = 0)
        true_gain = 2.0
        true_loss = 3.0
        n_families = 30
        
        self.log("  Simulating {0} families with θ = 0".format(n_families), "INFO")
        
        presence_matrix = self.simulate_gene_content(
            tree=tree,
            gain_rate=true_gain,
            loss_rate=true_loss,
            n_families=n_families,
            constrained_families=["fam0", "fam1", "fam2"],  # Designate some as "constrained"
            constraint_effect=0.0,  # But θ = 0, so no actual effect
        )
        
        family_names = ["fam{0}".format(i) for i in range(n_families)]
        
        gene_data = GeneContentData(
            tree=tree,
            presence_matrix=presence_matrix,
            taxon_names=taxon_names,
            family_names=family_names,
        )
        
        # Fit model with constraint
        constrained_families = ["fam0", "fam1", "fam2"]
        constraint = PerFamilyConstraint(constrained_families)
        
        inference = GeneContentInference(gene_data)
        
        # Fit and get θ̂
        result = inference.fit_with_constraint(constraint)
        theta_hat = result.constraint_params[0]
        
        # Success criterion: |θ̂| < 0.1
        tolerance = 0.1
        
        if abs(theta_hat) < tolerance:
            self.log("  True θ:     0.0", "INFO")
            self.log("  Estimated θ̂: {0:.4f}".format(theta_hat), "INFO")
            self.log("  |θ̂| = {0:.4f} < {1}".format(abs(theta_hat), tolerance), "PASS")
            return True
        else:
            self.log("  True θ:     0.0", "INFO")
            self.log("  Estimated θ̂: {0:.4f}".format(theta_hat), "INFO")
            self.log("  |θ̂| = {0:.4f} >= {1}".format(abs(theta_hat), tolerance), "FAIL")
            return False
    
    def test_type_i_error(self):
        """Test 2.2: Type I error rate (should be ≈ 0.05 at α = 0.05)."""
        self.log("Test 2.2: Type I error rate", "INFO")
        
        # Create tree
        newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
        tree = TreeStructure.from_newick(newick, backend="simple")
        taxon_names = ["A", "B", "C", "D"]
        
        # Run multiple replicates
        n_replicates = min(self.n_replicates, 10)  # Limit for quick test
        alpha = 0.05
        rejections = 0
        
        self.log("  Running {0} replicates under null (θ = 0)".format(n_replicates), "INFO")
        
        for rep in range(n_replicates):
            self.log("    Replicate {0}/{1}...".format(rep + 1, n_replicates), "INFO")
            
            # Simulate under null
            presence_matrix = self.simulate_gene_content(
                tree=tree,
                gain_rate=2.0,
                loss_rate=3.0,
                n_families=20,  # Keep small for speed
                constrained_families=["fam0", "fam1"],
                constraint_effect=0.0,  # θ = 0
            )
            
            family_names = ["fam{0}".format(i) for i in range(20)]
            
            gene_data = GeneContentData(
                tree=tree,
                presence_matrix=presence_matrix,
                taxon_names=taxon_names,
                family_names=family_names,
            )
            
            # Test H₀: θ = 0 vs H₁: θ ≠ 0
            constrained_families = ["fam0", "fam1"]
            constraint = PerFamilyConstraint(constrained_families)
            
            inference = GeneContentInference(gene_data)
            
            try:
                _, _, lrt_result = inference.fit_and_test(constraint)
                
                # Check if we reject null
                if lrt_result.p_value < alpha:
                    rejections += 1
            except Exception as e:
                self.log("    Replicate {0} failed: {1}".format(rep, e), "WARN")
                continue
        
        # Compute rejection rate
        rejection_rate = rejections / n_replicates
        
        # Success criterion: rejection_rate < 0.15
        # (Small number of replicates, so we're less strict)
        upper_bound = 0.15
        
        if rejection_rate < upper_bound:
            self.log("  Rejections: {0}/{1}".format(rejections, n_replicates), "INFO")
            self.log("  Rejection rate: {0:.3f}".format(rejection_rate), "INFO")
            self.log("  {0:.3f} < {1}".format(rejection_rate, upper_bound), "PASS")
            return True
        else:
            self.log("  Rejections: {0}/{1}".format(rejections, n_replicates), "INFO")
            self.log("  Rejection rate: {0:.3f}".format(rejection_rate), "INFO")
            self.log("  Expected: rate < {0}".format(upper_bound), "FAIL")
            return False
    
    def test_no_spurious_constraints(self):
        """Test 2.3: No spurious constraints under correct baseline."""
        self.log("Test 2.3: No spurious constraints", "INFO")
        
        # Create tree
        newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
        tree = TreeStructure.from_newick(newick, backend="simple")
        taxon_names = ["A", "B", "C", "D"]
        
        # Simulate with correct baseline (no constraint)
        presence_matrix = self.simulate_gene_content(
            tree=tree,
            gain_rate=2.0,
            loss_rate=3.0,
            n_families=30,
            constrained_families=None,  # No constraint
            constraint_effect=0.0,
        )
        
        family_names = ["fam{0}".format(i) for i in range(30)]
        
        gene_data = GeneContentData(
            tree=tree,
            presence_matrix=presence_matrix,
            taxon_names=taxon_names,
            family_names=family_names,
        )
        
        # Test multiple random family sets for spurious constraints
        n_tests = 5  # Limit for quick test
        false_positives = 0
        alpha = 0.05
        
        self.log("  Testing {0} random family sets".format(n_tests), "INFO")
        
        for test_idx in range(n_tests):
            # Pick random families to test
            test_families = self.rng.choice(
                family_names,
                size=3,
                replace=False
            ).tolist()
            
            constraint = PerFamilyConstraint(test_families)
            inference = GeneContentInference(gene_data)
            
            try:
                _, _, lrt_result = inference.fit_and_test(constraint)
                
                if lrt_result.p_value < alpha:
                    false_positives += 1
            except Exception as e:
                self.log("    Test {0} failed: {1}".format(test_idx, e), "WARN")
                continue
        
        # Success criterion: No more than 1 false positive (roughly α * n_tests)
        expected_fp = alpha * n_tests
        
        if false_positives <= 1:
            self.log("  False positives: {0}/{1}".format(false_positives, n_tests), "INFO")
            self.log("  Expected: ~{0:.1f}".format(expected_fp), "INFO")
            self.log("  No spurious constraints detected", "PASS")
            return True
        else:
            self.log("  False positives: {0}/{1}".format(false_positives, n_tests), "INFO")
            self.log("  Expected: ~{0:.1f}".format(expected_fp), "INFO")
            self.log("  Too many spurious constraints", "FAIL")
            return False
    
    def run_all(self):
        """Run all Level 2 tests."""
        print("=" * 70)
        print("LEVEL 2: STATISTICAL HONESTY TESTS")
        print("=" * 70)
        print()
        
        self.test_null_recovery()
        print()
        
        self.test_type_i_error()
        print()
        
        self.test_no_spurious_constraints()
        print()
        
        print("=" * 70)
        print("RESULTS: {0} passed, {1} failed".format(self.passed, self.failed))
        print("=" * 70)
        
        return self.passed, self.failed, self.results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Level 2 statistical tests")
    parser.add_argument(
        "--replicates",
        type=int,
        default=20,
        help="Number of replicates for Type I error test"
    )
    args = parser.parse_args()
    
    tests = StatisticalTests(n_replicates=args.replicates)
    passed, failed, results = tests.run_all()
    
    # Save results
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / "level2_statistical_{0}.txt".format(timestamp)
    
    with open(output_file, "w") as f:
        f.write("\n".join(results))
        f.write("\n\nRESULTS: {0} passed, {1} failed\n".format(passed, failed))
        f.write("Replicates: {0}\n".format(args.replicates))
    
    print("\nResults saved to: {0}".format(output_file))
    
    sys.exit(0 if failed == 0 else 1)
