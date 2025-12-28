#!/usr/bin/env python
"""Complete GeneContent plugin validation suite"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from persiste.core.trees import TreeStructure
from persiste.plugins.genecontent.inference.gene_inference import (
    GeneContentData,
    GeneContentModel,
    GeneContentInference,
)
from persiste.plugins.genecontent.constraints.gene_constraint import (
    NullConstraint,
    RetentionBiasConstraint,
)


class ValidationSuite:
    """GeneContent plugin validation suite"""

    def __init__(self, output_dir=None):
        """Initialize validation suite"""
        self.rng = np.random.default_rng(42)
        self.results = []
        self.passed = 0
        self.failed = 0
        
        if output_dir is None:
            output_dir = Path(__file__).parent / "outputs"
            output_dir.mkdir(exist_ok=True)
        self.output_dir = output_dir

    def log(self, message, status="INFO"):
        """Log message with status"""
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
        retained_families: Set[str] = None,
        retention_strength: float = 0.0,
    ) -> Tuple[np.ndarray, List[str]]:
        """Simulate gene content evolution on a tree"""
        from scipy.linalg import expm
        
        n_taxa = tree.n_tips
        presence_matrix = np.zeros((n_taxa, n_families), dtype=int)
        family_names = ["fam{0}".format(i) for i in range(n_families)]
        
        # Equilibrium frequencies
        pi_0 = loss_rate / (gain_rate + loss_rate)
        pi_1 = gain_rate / (gain_rate + loss_rate)
        
        for fam_idx, fam_name in enumerate(family_names):
            # Check if this family is retained
            is_retained = False
            if retained_families is not None:
                is_retained = fam_name in retained_families
            
            # Apply retention bias if needed
            if is_retained:
                effective_loss_rate = loss_rate * np.exp(retention_strength)
            else:
                effective_loss_rate = loss_rate
            
            # Build rate matrix
            Q = np.array([
                [-gain_rate, gain_rate],
                [effective_loss_rate, -effective_loss_rate]
            ])
            
            # Sample root state from equilibrium
            root_state = self.rng.choice([0, 1], p=[pi_0, pi_1])
            
            # Simulate down the tree
            node_states = {tree.root_index: root_state}
            
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
        
        return presence_matrix, family_names

    #### LEVEL 1: MECHANICAL CORRECTNESS TESTS ####

    def test_likelihood_computation(self):
        """Test 1.1: Likelihood computation works"""
        self.log("Test 1.1: Likelihood computation", "INFO")
        
        # Create tree
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
        
        # Initialize model
        constraint = NullConstraint()
        model = GeneContentModel(
            data=gene_data,
            constraint=constraint,
        )
        
        # Compute likelihood
        params = {
            'log_gain': np.log(1.0),
            'log_loss': np.log(2.0),
        }
        log_lik = model.log_likelihood(params)
        
        if np.isfinite(log_lik) and log_lik < 0:
            self.log("  Log-likelihood: {0:.6f}".format(log_lik), "INFO")
            self.log("  Likelihood computation works correctly", "PASS")
            return True
        else:
            self.log("  Log-likelihood: {0:.6f}".format(log_lik), "INFO")
            self.log("  Likelihood should be finite and negative", "FAIL")
            return False

    def test_branch_length_scaling(self):
        """Test 1.2: Branch length scaling works correctly"""
        self.log("Test 1.2: Branch length scaling", "INFO")
        
        # Create trees with different branch lengths
        newick1 = "(A:1.0,B:1.0);"
        tree1 = TreeStructure.from_newick(newick1, backend="simple")
        
        newick2 = "(A:2.0,B:2.0);"
        tree2 = TreeStructure.from_newick(newick2, backend="simple")
        
        # Same data for both trees
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
        
        # Create models
        model1 = GeneContentModel(gene_data1, constraint=NullConstraint())
        model2 = GeneContentModel(gene_data2, constraint=NullConstraint())
        
        # Compute likelihoods with same parameters
        params = {
            'log_gain': np.log(1.0),
            'log_loss': np.log(2.0),
        }
        
        log_lik1 = model1.log_likelihood(params)
        log_lik2 = model2.log_likelihood(params)
        
        if log_lik1 != log_lik2:
            self.log("  Unit branches:    {0:.6f}".format(log_lik1), "INFO")
            self.log("  Doubled branches: {0:.6f}".format(log_lik2), "INFO")
            self.log("  Branch length scaling works correctly", "PASS")
            return True
        else:
            self.log("  Unit branches:    {0:.6f}".format(log_lik1), "INFO")
            self.log("  Doubled branches: {0:.6f}".format(log_lik2), "INFO")
            self.log("  Branch length scaling should affect likelihood", "FAIL")
            return False

    #### LEVEL 2: STATISTICAL HONESTY TESTS ####

    def test_null_recovery(self):
        """Test 2.1: Null recovery (θ = 0 → θ̂ ≈ 0)"""
        self.log("Test 2.1: Null recovery", "INFO")
        
        # Create tree
        newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
        tree = TreeStructure.from_newick(newick, backend="simple")
        taxon_names = ["A", "B", "C", "D"]
        
        # Simulate data with NO retention bias (θ = 0)
        true_gain = 2.0
        true_loss = 3.0
        n_families = 30
        
        self.log("  Simulating {0} families with θ = 0".format(n_families), "INFO")
        retained_families = {"fam0", "fam1", "fam2"}
        
        presence_matrix, family_names = self.simulate_gene_content(
            tree=tree,
            gain_rate=true_gain,
            loss_rate=true_loss,
            n_families=n_families,
            retained_families=retained_families,
            retention_strength=0.0,  # No actual effect (θ = 0)
        )
        
        gene_data = GeneContentData(
            tree=tree,
            presence_matrix=presence_matrix,
            taxon_names=taxon_names,
            family_names=family_names,
        )
        
        # Fit model with retention constraint
        constraint = RetentionBiasConstraint(retained_families=retained_families)
        inference = GeneContentInference(gene_data)
        
        # Fit and get θ̂
        result = inference.fit_with_constraint(constraint)
        theta_hat = result.parameters['retention_strength']
        
        # Success criterion: |θ̂| < 0.2
        tolerance = 0.2
        
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
    
    def test_parameter_recovery(self):
        """Test 2.2: Parameter recovery (θ ≠ 0 → θ̂ ≈ θ)"""
        self.log("Test 2.2: Parameter recovery", "INFO")
        
        # Create tree
        newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
        tree = TreeStructure.from_newick(newick, backend="simple")
        taxon_names = ["A", "B", "C", "D"]
        
        # True retention strength (negative = reduced loss rate)
        true_theta = -1.0
        
        # Simulate data WITH retention bias
        retained_families = {"fam0", "fam1", "fam2"}
        presence_matrix, family_names = self.simulate_gene_content(
            tree=tree,
            gain_rate=2.0,
            loss_rate=3.0,
            n_families=30,
            retained_families=retained_families,
            retention_strength=true_theta,
        )
        
        gene_data = GeneContentData(
            tree=tree,
            presence_matrix=presence_matrix,
            taxon_names=taxon_names,
            family_names=family_names,
        )
        
        # Fit model with retention constraint
        constraint = RetentionBiasConstraint(retained_families=retained_families)
        inference = GeneContentInference(gene_data)
        
        result = inference.fit_with_constraint(constraint)
        theta_hat = result.parameters['retention_strength']
        
        # Success criterion: same sign and |θ̂ - θ|/|θ| < 0.5
        relative_error = abs(theta_hat - true_theta) / abs(true_theta)
        correct_sign = (theta_hat * true_theta) > 0
        
        if correct_sign and relative_error < 0.5:
            self.log("  True θ:     {0:.4f}".format(true_theta), "INFO")
            self.log("  Estimated θ̂: {0:.4f}".format(theta_hat), "INFO")
            self.log("  Relative error: {0:.4f} < 0.5".format(relative_error), "PASS")
            return True
        else:
            self.log("  True θ:     {0:.4f}".format(true_theta), "INFO")
            self.log("  Estimated θ̂: {0:.4f}".format(theta_hat), "INFO")
            self.log("  Relative error: {0:.4f} >= 0.5 or wrong sign".format(relative_error), "FAIL")
            return False

    #### LEVEL 3: IDENTIFIABILITY TESTS ####
    
    def test_profile_likelihood(self):
        """Test 3.1: Profile likelihood shows curvature"""
        self.log("Test 3.1: Profile likelihood curvature", "INFO")
        
        # Create tree
        newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
        tree = TreeStructure.from_newick(newick, backend="simple")
        taxon_names = ["A", "B", "C", "D"]
        
        # True retention strength
        true_theta = -1.0
        
        # Simulate data WITH retention bias
        retained_families = {"fam0", "fam1", "fam2"}
        presence_matrix, family_names = self.simulate_gene_content(
            tree=tree,
            gain_rate=2.0,
            loss_rate=3.0,
            n_families=30,
            retained_families=retained_families,
            retention_strength=true_theta,
        )
        
        gene_data = GeneContentData(
            tree=tree,
            presence_matrix=presence_matrix,
            taxon_names=taxon_names,
            family_names=family_names,
        )
        
        # Compute profile likelihood
        theta_grid = np.linspace(-3.0, 1.0, 21)
        profile_lls = []
        
        # Fit null model first to get good gain/loss rates
        inference = GeneContentInference(gene_data)
        null_result = inference.fit_null()
        gain_rate = np.exp(null_result.parameters['log_gain'])
        loss_rate = np.exp(null_result.parameters['log_loss'])
        
        # Fix gain/loss at null MLE, vary retention strength
        for theta in theta_grid:
            constraint = RetentionBiasConstraint(
                retained_families=retained_families,
                retention_strength=theta,
            )
            
            model = GeneContentModel(
                data=gene_data,
                constraint=constraint,
            )
            
            parameters = {
                'log_gain': null_result.parameters['log_gain'],
                'log_loss': null_result.parameters['log_loss'],
                'retention_strength': theta,
            }
            
            ll = model.log_likelihood(parameters)
            profile_lls.append(ll)
        
        # Find maximum
        profile_lls = np.array(profile_lls)
        max_idx = np.argmax(profile_lls)
        theta_mle = theta_grid[max_idx]
        
        # Check for curvature
        left_idx = max(0, max_idx - 3)
        right_idx = min(len(theta_grid) - 1, max_idx + 3)
        
        has_curvature = (
            profile_lls[max_idx] > profile_lls[left_idx] and
            profile_lls[max_idx] > profile_lls[right_idx]
        )
        
        # Save profile likelihood plot
        plt.figure(figsize=(10, 6))
        plt.plot(theta_grid, profile_lls, 'b-', linewidth=2)
        plt.axvline(true_theta, color='r', linestyle='--', label='True θ = {0}'.format(true_theta))
        plt.axvline(theta_mle, color='g', linestyle='--', label='MLE θ̂ = {0:.2f}'.format(theta_mle))
        plt.xlabel('Retention strength (θ)', fontsize=12)
        plt.ylabel('Log-likelihood', fontsize=12)
        plt.title('Profile Likelihood for Retention Strength', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.output_dir / "profile_likelihood_{0}.png".format(timestamp)
        plt.savefig(plot_file)
        plt.close()
        
        if has_curvature:
            self.log("  True θ:  {0:.2f}".format(true_theta), "INFO")
            self.log("  MLE θ̂:   {0:.2f}".format(theta_mle), "INFO")
            self.log("  Profile likelihood shows clear curvature", "PASS")
            self.log("  Plot saved to: {0}".format(plot_file), "INFO")
            return True
        else:
            self.log("  True θ:  {0:.2f}".format(true_theta), "INFO")
            self.log("  MLE θ̂:   {0:.2f}".format(theta_mle), "INFO")
            self.log("  Profile likelihood is too flat", "FAIL")
            self.log("  Plot saved to: {0}".format(plot_file), "INFO")
            return False

    #### LEVEL 4: BASELINE SENSITIVITY TESTS ####
    
    def test_baseline_misspecification(self):
        """Test 4.1: Robustness to baseline misspecification"""
        self.log("Test 4.1: Baseline misspecification", "INFO")
        
        # Create tree
        newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
        tree = TreeStructure.from_newick(newick, backend="simple")
        taxon_names = ["A", "B", "C", "D"]
        
        # True parameters
        true_gain = 2.0
        true_loss = 3.0
        true_theta = -1.0
        
        # Simulate data with retention bias
        retained_families = {"fam0", "fam1", "fam2"}
        presence_matrix, family_names = self.simulate_gene_content(
            tree=tree,
            gain_rate=true_gain,
            loss_rate=true_loss,
            n_families=30,
            retained_families=retained_families,
            retention_strength=true_theta,
        )
        
        gene_data = GeneContentData(
            tree=tree,
            presence_matrix=presence_matrix,
            taxon_names=taxon_names,
            family_names=family_names,
        )
        
        # Test with slightly misspecified baseline
        # Instead of letting optimizer find gain/loss, fix at wrong values
        constraint = RetentionBiasConstraint(retained_families=retained_families)
        model = GeneContentModel(data=gene_data, constraint=constraint)
        
        # Wrong gain/loss rates (30% error)
        wrong_gain = true_gain * 1.3
        wrong_loss = true_loss * 0.7
        
        parameters = {
            'log_gain': np.log(wrong_gain),
            'log_loss': np.log(wrong_loss),
            'retention_strength': true_theta,
        }
        
        # Wrong baseline log-likelihood
        wrong_ll = model.log_likelihood(parameters)
        
        # Optimize retention strength with wrong baseline
        inference = GeneContentInference(gene_data)
        
        # First create a simpler API for testing baseline sensitivity
        class SimpleModel:
            def __init__(self, model, fixed_gain, fixed_loss):
                self.model = model
                self.fixed_gain = fixed_gain
                self.fixed_loss = fixed_loss
                
            def log_likelihood(self, theta):
                params = {
                    'log_gain': np.log(self.fixed_gain),
                    'log_loss': np.log(self.fixed_loss),
                    'retention_strength': theta,
                }
                return self.model.log_likelihood(params)
        
        # Create a simple grid search function
        def grid_search(simple_model, min_val=-3.0, max_val=1.0, n_steps=41):
            theta_grid = np.linspace(min_val, max_val, n_steps)
            lls = [simple_model.log_likelihood(theta) for theta in theta_grid]
            max_idx = np.argmax(lls)
            return theta_grid[max_idx]
        
        # Grid search for theta with wrong baseline
        simple_model = SimpleModel(model, wrong_gain, wrong_loss)
        theta_wrong = grid_search(simple_model)
        
        # Check bias
        bias = abs(theta_wrong - true_theta)
        relative_bias = bias / abs(true_theta)
        
        if relative_bias < 0.3:
            self.log("  True θ:      {0:.4f}".format(true_theta), "INFO")
            self.log("  Misspecified θ̂: {0:.4f}".format(theta_wrong), "INFO")
            self.log("  Relative bias: {0:.4f} < 0.3".format(relative_bias), "PASS")
            return True
        else:
            self.log("  True θ:      {0:.4f}".format(true_theta), "INFO")
            self.log("  Misspecified θ̂: {0:.4f}".format(theta_wrong), "INFO")
            self.log("  Relative bias: {0:.4f} >= 0.3".format(relative_bias), "FAIL")
            return False

    def run_all(self):
        """Run all validation tests"""
        self.log("=" * 80, "INFO")
        self.log("GENECONTENT PLUGIN VALIDATION SUITE", "INFO")
        self.log("=" * 80, "INFO")
        
        # Level 1: Mechanical Correctness
        self.log("\nLEVEL 1: MECHANICAL CORRECTNESS", "INFO")
        self.log("=" * 80, "INFO")
        
        self.test_likelihood_computation()
        print()
        self.test_branch_length_scaling()
        print()
        
        # Level 2: Statistical Honesty
        self.log("\nLEVEL 2: STATISTICAL HONESTY", "INFO")
        self.log("=" * 80, "INFO")
        
        self.test_null_recovery()
        print()
        self.test_parameter_recovery()
        print()
        
        # Level 3: Identifiability
        self.log("\nLEVEL 3: IDENTIFIABILITY", "INFO")
        self.log("=" * 80, "INFO")
        
        self.test_profile_likelihood()
        print()
        
        # Level 4: Baseline Sensitivity
        self.log("\nLEVEL 4: BASELINE SENSITIVITY", "INFO")
        self.log("=" * 80, "INFO")
        
        self.test_baseline_misspecification()
        print()
        
        # Summary
        self.log("\n" + "=" * 80, "INFO")
        self.log("VALIDATION SUMMARY", "INFO")
        self.log("=" * 80, "INFO")
        self.log("Total tests: {0}".format(self.passed + self.failed), "INFO")
        self.log("Passed:      {0}".format(self.passed), "INFO")
        self.log("Failed:      {0}".format(self.failed), "INFO")
        
        if self.failed == 0:
            self.log("\n[PASS] ALL VALIDATION TESTS PASSED", "INFO")
            self.log("System is ready for real data.", "INFO")
        else:
            self.log("\n[FAIL] {0} TESTS FAILED".format(self.failed), "INFO")
            self.log("Review failures before proceeding to real data.", "INFO")
        
        # Save validation report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / "validation_report_{0}.txt".format(timestamp)
        
        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("GENECONTENT PLUGIN VALIDATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write("Generated: {0}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            f.write("=" * 80 + "\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write("Total tests: {0}\n".format(self.passed + self.failed))
            f.write("Passed:      {0}\n".format(self.passed))
            f.write("Failed:      {0}\n".format(self.failed))
            f.write("\n")
            
            if self.failed == 0:
                f.write("[PASS] ALL VALIDATION TESTS PASSED\n\n")
                f.write("The GeneContent plugin has passed validation.\n")
                f.write("The system is ready for real data analysis.\n\n")
                f.write("Validation completed for all requested levels:\n")
                f.write("1. Mechanical Correctness: Likelihood computation works correctly\n")
                f.write("2. Statistical Honesty: Null recovery and parameter recovery confirmed\n")
                f.write("3. Identifiability: Profile likelihood shows clear curvature\n")
                f.write("4. Baseline Sensitivity: Robust to baseline misspecification\n\n")
                f.write("Next steps:\n")
                f.write("1. Run inference on real datasets\n")
                f.write("2. Compare results with existing methods (Count, GLOOME, BadiRate)\n")
                f.write("3. Document findings for publication\n")
            else:
                f.write("[FAIL] {0} TESTS FAILED\n\n".format(self.failed))
                f.write("Review failures before proceeding to real data.\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            for line in self.results:
                f.write(line + "\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        self.log("\nValidation report saved to: {0}".format(report_file), "INFO")
        
        return self.passed, self.failed


if __name__ == "__main__":
    validation = ValidationSuite()
    passed, failed = validation.run_all()
    sys.exit(0 if failed == 0 else 1)
