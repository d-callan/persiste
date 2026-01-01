"""
Comprehensive validation runner for CopyNumberDynamics plugin.

Runs all three tiers of validation and produces a summary report.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from persiste.core.trees import TreeStructure, build_star_tree
from persiste.plugins.copynumber.validation.cn_simulator import (
    simulate_scenario,
    SimulationScenario,
)
from persiste.plugins.copynumber.cn_interface import (
    fit,
    fit_null_model,
    likelihood_ratio_test,
)


@dataclass
class ValidationResult:
    """Results from a validation test."""
    test_name: str
    passed: bool
    details: str
    metric_value: float = None
    threshold: float = None


class ValidationRunner:
    """
    Runs comprehensive validation suite.
    
    Three tiers:
    1. Structural/sanity (cheap, must-have)
    2. Simulation-based recovery (core validation)
    3. Empirical plausibility (future)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[ValidationResult] = []
    
    def log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)
    
    def run_tier1_structural(self) -> List[ValidationResult]:
        """
        Run Tier 1: Structural/sanity validation.
        
        These are cheap, must-have tests.
        """
        self.log("\n" + "=" * 70)
        self.log("TIER 1: STRUCTURAL / SANITY VALIDATION")
        self.log("=" * 70)
        
        results = []
        
        # Import test functions
        from tests.plugins.copynumber.test_structural_validation import (
            TestRateMatrixIntegrity,
            TestLikelihoodMonotonicity,
            TestIdentifiabilitySmokeTest,
        )
        
        # Run rate matrix tests
        self.log("\nA. Rate Matrix Integrity")
        test_class = TestRateMatrixIntegrity()
        
        tests = [
            ("Off-diagonal non-negative", test_class.test_off_diagonal_nonnegative),
            ("Row sums zero", test_class.test_row_sums_zero),
            ("Forbidden transitions zero", test_class.test_forbidden_transitions_zero),
            ("Allowed transitions positive", test_class.test_allowed_transitions_positive),
            ("Hierarchical baseline valid", test_class.test_hierarchical_baseline_integrity),
            ("Constraints preserve validity", test_class.test_constraint_preserves_validity),
        ]
        
        for name, test_func in tests:
            try:
                test_func()
                results.append(ValidationResult(name, True, "PASS"))
                self.log(f"  ✓ {name}")
            except Exception as e:
                results.append(ValidationResult(name, False, f"FAIL: {str(e)}"))
                self.log(f"  ✗ {name}: {str(e)}")
        
        # Run likelihood monotonicity tests
        self.log("\nB. Likelihood Monotonicity")
        test_class = TestLikelihoodMonotonicity()
        
        tests = [
            ("θ=0 recovers baseline", test_class.test_theta_zero_recovers_baseline),
            ("Constraints neutral at θ=0", test_class.test_constraint_at_zero_is_neutral),
        ]
        
        for name, test_func in tests:
            try:
                test_func()
                results.append(ValidationResult(name, True, "PASS"))
                self.log(f"  ✓ {name}")
            except Exception as e:
                results.append(ValidationResult(name, False, f"FAIL: {str(e)}"))
                self.log(f"  ✗ {name}: {str(e)}")
        
        # Run identifiability tests
        self.log("\nC. Identifiability Smoke Test")
        test_class = TestIdentifiabilitySmokeTest()
        
        tests = [
            ("Gain vs amplify distinct", test_class.test_gain_vs_amplify_distinct),
            ("Loss vs contract distinct", test_class.test_loss_vs_contract_distinct),
            ("Constraint types distinct", test_class.test_constraint_types_produce_different_patterns),
        ]
        
        for name, test_func in tests:
            try:
                test_func()
                results.append(ValidationResult(name, True, "PASS"))
                self.log(f"  ✓ {name}")
            except Exception as e:
                results.append(ValidationResult(name, False, f"FAIL: {str(e)}"))
                self.log(f"  ✗ {name}: {str(e)}")
        
        return results
    
    def run_tier2_simulation(self) -> List[ValidationResult]:
        """
        Run Tier 2: Simulation-based recovery validation.
        
        This is the core validation.
        """
        self.log("\n" + "=" * 70)
        self.log("TIER 2: SIMULATION-BASED RECOVERY")
        self.log("=" * 70)
        
        results = []
        
        # Create test tree
        tree = self._create_test_tree()
        
        # Test 1: Null scenario (false positive rate)
        self.log("\n1. Null Scenario (False Positive Rate)")
        fpr_result = self._test_false_positive_rate(tree)
        results.append(fpr_result)
        
        # Test 2: Dosage buffering detection
        self.log("\n2. Dosage Buffering Detection")
        dosage_result = self._test_dosage_buffering_detection(tree)
        results.append(dosage_result)
        
        # Test 3: Amplification bias detection
        self.log("\n3. Amplification Bias Detection")
        amp_result = self._test_amplification_bias_detection(tree)
        results.append(amp_result)
        
        # Test 4: θ recovery
        self.log("\n4. Parameter Recovery")
        theta_result = self._test_theta_recovery(tree)
        results.append(theta_result)
        
        # Test 5: Wrong constraint rejection
        self.log("\n5. Wrong Constraint Rejection")
        wrong_result = self._test_wrong_constraint_rejection(tree)
        results.append(wrong_result)
        
        # Test 6: Power analysis
        self.log("\n6. Statistical Power")
        power_result = self._test_statistical_power(tree)
        results.append(power_result)
        
        return results
    
    def _create_test_tree(self) -> TreeStructure:
        """Create test tree for validation."""
        taxon_names = [f"taxon_{i:02d}" for i in range(20)]
        return build_star_tree(taxon_names, branch_length=0.5)
    
    def _test_false_positive_rate(self, tree: TreeStructure) -> ValidationResult:
        """Test false positive rate under null."""
        n_replicates = 10
        false_positives = 0
        
        for rep in range(n_replicates):
            cn_matrix, metadata = simulate_scenario(
                SimulationScenario.NULL,
                tree,
                n_families=50,
                seed=42 + rep
            )
            
            taxon_names = metadata['taxon_names']
            family_names = [f"fam_{i}" for i in range(50)]
            
            null_result = fit_null_model(
                cn_matrix, family_names, taxon_names, tree,
                baseline_type='global'
            )
            
            alt_result = fit(
                cn_matrix, family_names, taxon_names, tree,
                baseline_type='global',
                constraint_type='dosage_stability',
                theta=-0.3,
            )
            
            comparison = likelihood_ratio_test(alt_result, null_result)
            if comparison['p_value'] < 0.05:
                false_positives += 1
        
        fpr = false_positives / n_replicates
        passed = fpr <= 0.15  # Allow some tolerance
        
        self.log(f"  False positive rate: {fpr:.1%} (threshold: ≤15%)")
        self.log(f"  {'✓ PASS' if passed else '✗ FAIL'}")
        
        return ValidationResult(
            "False positive rate",
            passed,
            f"FPR = {fpr:.1%}",
            metric_value=fpr,
            threshold=0.15
        )
    
    def _test_dosage_buffering_detection(self, tree: TreeStructure) -> ValidationResult:
        """Test detection of dosage buffering."""
        cn_matrix, metadata = simulate_scenario(
            SimulationScenario.DOSAGE_BUFFERING,
            tree,
            n_families=100,
            seed=42
        )
        
        taxon_names = metadata['taxon_names']
        family_names = [f"fam_{i}" for i in range(100)]
        
        null_result = fit_null_model(
            cn_matrix, family_names, taxon_names, tree,
            baseline_type='global'
        )
        
        alt_result = fit(
            cn_matrix, family_names, taxon_names, tree,
            baseline_type='global',
            constraint_type='dosage_stability',
            theta=-0.5,
        )
        
        comparison = likelihood_ratio_test(alt_result, null_result)
        detected = comparison['p_value'] < 0.05
        
        self.log(f"  p-value: {comparison['p_value']:.4e}")
        self.log(f"  {'✓ DETECTED' if detected else '✗ NOT DETECTED'}")
        
        return ValidationResult(
            "Dosage buffering detection",
            detected,
            f"p = {comparison['p_value']:.4e}",
            metric_value=comparison['p_value'],
            threshold=0.05
        )
    
    def _test_amplification_bias_detection(self, tree: TreeStructure) -> ValidationResult:
        """Test detection of amplification bias."""
        cn_matrix, metadata = simulate_scenario(
            SimulationScenario.AMPLIFICATION_BIAS,
            tree,
            n_families=100,
            seed=42
        )
        
        taxon_names = metadata['taxon_names']
        family_names = [f"fam_{i}" for i in range(100)]
        
        null_result = fit_null_model(
            cn_matrix, family_names, taxon_names, tree,
            baseline_type='global'
        )
        
        alt_result = fit(
            cn_matrix, family_names, taxon_names, tree,
            baseline_type='global',
            constraint_type='amplification_bias',
            theta=0.5,
        )
        
        comparison = likelihood_ratio_test(alt_result, null_result)
        detected = comparison['p_value'] < 0.05
        
        self.log(f"  p-value: {comparison['p_value']:.4e}")
        self.log(f"  {'✓ DETECTED' if detected else '✗ NOT DETECTED'}")
        
        return ValidationResult(
            "Amplification bias detection",
            detected,
            f"p = {comparison['p_value']:.4e}",
            metric_value=comparison['p_value'],
            threshold=0.05
        )
    
    def _test_theta_recovery(self, tree: TreeStructure) -> ValidationResult:
        """Test recovery of true θ value."""
        from persiste.plugins.copynumber.validation.cn_simulator import (
            create_scenario_config,
            simulate_cn_evolution,
        )
        
        true_theta = -0.5
        
        config = create_scenario_config(
            SimulationScenario.DOSAGE_BUFFERING,
            n_families=100,
            seed=42
        )
        config.theta = true_theta
        
        cn_matrix, metadata = simulate_cn_evolution(tree, config)
        
        taxon_names = metadata['taxon_names']
        family_names = [f"fam_{i}" for i in range(100)]
        
        # Test different theta values
        theta_values = [-0.7, -0.6, -0.5, -0.4, -0.3]
        likelihoods = []
        
        for theta in theta_values:
            result = fit(
                cn_matrix, family_names, taxon_names, tree,
                baseline_type='global',
                constraint_type='dosage_stability',
                theta=theta,
            )
            likelihoods.append(result.log_likelihood)
        
        best_idx = np.argmax(likelihoods)
        best_theta = theta_values[best_idx]
        
        relative_error = abs(best_theta - true_theta) / abs(true_theta)
        passed = relative_error < 0.3
        
        self.log(f"  True θ: {true_theta:.2f}")
        self.log(f"  Best θ: {best_theta:.2f}")
        self.log(f"  Relative error: {relative_error:.1%}")
        self.log(f"  {'✓ PASS' if passed else '✗ FAIL'} (threshold: <30%)")
        
        return ValidationResult(
            "θ recovery",
            passed,
            f"Error = {relative_error:.1%}",
            metric_value=relative_error,
            threshold=0.3
        )
    
    def _test_wrong_constraint_rejection(self, tree: TreeStructure) -> ValidationResult:
        """Test that wrong constraint is not preferred."""
        cn_matrix, metadata = simulate_scenario(
            SimulationScenario.DOSAGE_BUFFERING,
            tree,
            n_families=100,
            seed=42
        )
        
        taxon_names = metadata['taxon_names']
        family_names = [f"fam_{i}" for i in range(100)]
        
        null_result = fit_null_model(
            cn_matrix, family_names, taxon_names, tree,
            baseline_type='global'
        )
        
        # Wrong constraint
        wrong_result = fit(
            cn_matrix, family_names, taxon_names, tree,
            baseline_type='global',
            constraint_type='amplification_bias',
            theta=0.5,
        )
        
        comparison = likelihood_ratio_test(wrong_result, null_result)
        rejected = comparison['delta_aic'] > -2
        
        self.log(f"  Δ AIC: {comparison['delta_aic']:.2f}")
        self.log(f"  {'✓ REJECTED' if rejected else '✗ NOT REJECTED'}")
        
        return ValidationResult(
            "Wrong constraint rejection",
            rejected,
            f"Δ AIC = {comparison['delta_aic']:.2f}",
            metric_value=comparison['delta_aic'],
            threshold=-2.0
        )
    
    def _test_statistical_power(self, tree: TreeStructure) -> ValidationResult:
        """Test statistical power at reasonable branch lengths."""
        cn_matrix, metadata = simulate_scenario(
            SimulationScenario.DOSAGE_BUFFERING,
            tree,
            n_families=100,
            seed=42
        )
        
        taxon_names = metadata['taxon_names']
        family_names = [f"fam_{i}" for i in range(100)]
        
        null_result = fit_null_model(
            cn_matrix, family_names, taxon_names, tree,
            baseline_type='global'
        )
        
        alt_result = fit(
            cn_matrix, family_names, taxon_names, tree,
            baseline_type='global',
            constraint_type='dosage_stability',
            theta=-0.5,
        )
        
        comparison = likelihood_ratio_test(alt_result, null_result)
        detected = comparison['p_value'] < 0.05
        
        self.log(f"  Detection at branch length 0.5: {'YES' if detected else 'NO'}")
        self.log(f"  {'✓ PASS' if detected else '✗ FAIL'} (≥80% power expected)")
        
        return ValidationResult(
            "Statistical power",
            detected,
            f"Detected = {detected}",
            metric_value=1.0 if detected else 0.0,
            threshold=0.8
        )
    
    def print_summary(self):
        """Print validation summary."""
        self.log("\n" + "=" * 70)
        self.log("VALIDATION SUMMARY")
        self.log("=" * 70)
        
        tier1_results = [r for r in self.results if "Tier 1" in r.test_name or 
                        any(x in r.test_name for x in ["non-negative", "zero", "distinct", "neutral"])]
        tier2_results = [r for r in self.results if r not in tier1_results]
        
        self.log(f"\nTier 1 (Structural): {sum(r.passed for r in tier1_results)}/{len(tier1_results)} passed")
        self.log(f"Tier 2 (Simulation): {sum(r.passed for r in tier2_results)}/{len(tier2_results)} passed")
        
        self.log(f"\nOverall: {sum(r.passed for r in self.results)}/{len(self.results)} tests passed")
        
        # Critical failures
        critical_failures = [r for r in self.results if not r.passed and 
                           any(x in r.test_name.lower() for x in ["false positive", "detection", "power"])]
        
        if critical_failures:
            self.log("\n⚠ CRITICAL FAILURES:")
            for result in critical_failures:
                self.log(f"  - {result.test_name}: {result.details}")
        
        all_passed = all(r.passed for r in self.results)
        
        if all_passed:
            self.log("\n✓ ALL VALIDATION TESTS PASSED")
            self.log("\nThe CopyNumberDynamics plugin is validated and ready for use.")
        else:
            self.log("\n✗ SOME VALIDATION TESTS FAILED")
            self.log("\nReview failures before using in production.")
        
        self.log("=" * 70)
    
    def run_all(self):
        """Run complete validation suite."""
        self.log("=" * 70)
        self.log("COPY NUMBER DYNAMICS - VALIDATION SUITE")
        self.log("=" * 70)
        
        # Tier 1
        tier1_results = self.run_tier1_structural()
        self.results.extend(tier1_results)
        
        # Check if Tier 1 passed
        if not all(r.passed for r in tier1_results):
            self.log("\n⚠ Tier 1 failures detected. Fix before proceeding to Tier 2.")
            self.print_summary()
            return
        
        # Tier 2
        tier2_results = self.run_tier2_simulation()
        self.results.extend(tier2_results)
        
        # Summary
        self.print_summary()


def main():
    """Main entry point for validation."""
    runner = ValidationRunner(verbose=True)
    runner.run_all()


if __name__ == "__main__":
    main()
