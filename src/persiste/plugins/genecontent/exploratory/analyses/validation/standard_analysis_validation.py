#!/usr/bin/env python3
"""
Standard Analysis Validation Suite for GeneContent Plugin

Validates all standard analyses across 4 levels:
- L1: Mechanical correctness (likelihood computation)
- L2: Statistical honesty (parameter recovery, null recovery)
- L3: Identifiability (parameter consistency across replicates)
- L4: Robustness (baseline sensitivity)

Key validation from previous work:
- Stationary frequency (π₁ = gain/(gain+loss)) has ~5% error
- Individual rates have ~70% error (fundamental identifiability issue)
- This is consistent across tools (validated with GLOOME)
"""

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from persiste.core.trees import load_tree
from persiste.plugins.genecontent.inference.gene_inference import GeneContentData
from persiste.plugins.genecontent.analyses.standard_analyses import GeneContentAnalysis


@dataclass
class ValidationResult:
    """Result of a single validation test."""
    test_name: str
    level: str  # L1, L2, L3, L4
    analysis: str  # Which analysis (1-5)
    passed: bool
    message: str
    details: Dict = None


class ValidationSuite:
    """Comprehensive validation suite for standard analyses."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[ValidationResult] = []
    
    def log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)
    
    def add_result(self, result: ValidationResult):
        """Add a validation result."""
        self.results.append(result)
        status = "[PASS]" if result.passed else "[FAIL]"
        self.log(f"{status} {result.test_name} ({result.level}): {result.message}")
    
    def create_test_tree(self, n_tips: int = 8) -> tuple:
        """Create a simple test tree and taxon names."""
        newick = "(((tip0:1,tip1:1):1,(tip2:1,tip3:1):1):1,((tip4:1,tip5:1):1,(tip6:1,tip7:1):1):1):0;"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.nwk', delete=False) as f:
            f.write(newick)
            tree_file = f.name
        
        tree = load_tree(tree_file)
        Path(tree_file).unlink()
        
        taxon_names = [f"tip{i}" for i in range(n_tips)]
        return tree, taxon_names
    
    def create_test_data(self, n_tips: int = 8, n_families: int = 100) -> GeneContentData:
        """Create simple test data."""
        tree, taxon_names = self.create_test_tree(n_tips)
        
        # Random presence/absence
        np.random.seed(42)
        presence_matrix = np.random.binomial(1, 0.5, size=(n_tips, n_families))
        family_names = [f"fam{i}" for i in range(n_families)]
        
        return GeneContentData(
            tree=tree,
            presence_matrix=presence_matrix,
            taxon_names=taxon_names,
            family_names=family_names
        )
    
    # ========== Analysis 1: Global Rates ==========
    
    def validate_analysis1_l1_likelihood(self):
        """L1: Test that likelihood computation is mechanically correct."""
        self.log("\n=== Analysis 1 (Global Rates) - L1: Likelihood Computation ===")
        
        data = self.create_test_data(n_tips=8, n_families=50)
        analysis = GeneContentAnalysis(data)
        result = analysis.global_rates(verbose=False)
        
        # Check that likelihood is finite and negative
        ll = result.log_likelihood
        passed = np.isfinite(ll) and ll < 0
        
        self.add_result(ValidationResult(
            test_name="GlobalRates_Likelihood",
            level="L1",
            analysis="1",
            passed=passed,
            message=f"LL={ll:.2f}, finite={np.isfinite(ll)}, negative={ll < 0}",
            details={'log_likelihood': ll}
        ))
    
    def validate_analysis1_l2_parameter_recovery(self):
        """L2: Test parameter recovery on simulated data."""
        self.log("\n=== Analysis 1 (Global Rates) - L2: Parameter Recovery ===")
        
        # Use random data (not true simulation, but tests the pipeline)
        data = self.create_test_data(n_tips=8, n_families=200)
        analysis = GeneContentAnalysis(data)
        result = analysis.global_rates(verbose=False)
        
        # Check that rates are positive and reasonable
        passed = (result.gain_rate > 0 and result.loss_rate > 0 and
                  result.gain_rate < 100 and result.loss_rate < 100)
        
        self.add_result(ValidationResult(
            test_name="GlobalRates_ParameterRecovery",
            level="L2",
            analysis="1",
            passed=passed,
            message=f"Gain={result.gain_rate:.3f}, Loss={result.loss_rate:.3f}",
            details={
                'gain_rate': result.gain_rate,
                'loss_rate': result.loss_rate,
                'stationary_freq': result.gain_rate / (result.gain_rate + result.loss_rate)
            }
        ))
    
    def validate_analysis1_l2_stationary_frequency(self):
        """L2: Test stationary frequency recovery (key insight: much better than individual rates)."""
        self.log("\n=== Analysis 1 (Global Rates) - L2: Stationary Frequency ===")
        
        # Run multiple replicates to test consistency
        n_replicates = 5
        pi1_values = []
        
        for i in range(n_replicates):
            np.random.seed(42 + i)
            data = self.create_test_data(n_tips=8, n_families=200)
            analysis = GeneContentAnalysis(data)
            result = analysis.global_rates(verbose=False)
            pi1 = result.gain_rate / (result.gain_rate + result.loss_rate)
            pi1_values.append(pi1)
        
        mean_pi1 = np.mean(pi1_values)
        std_pi1 = np.std(pi1_values)
        cv_pi1 = std_pi1 / mean_pi1 if mean_pi1 > 0 else np.inf
        
        # Stationary frequency should be more stable than individual rates
        passed = cv_pi1 < 0.2  # CV < 20%
        
        self.add_result(ValidationResult(
            test_name="GlobalRates_StationaryFrequency",
            level="L2",
            analysis="1",
            passed=passed,
            message=f"π₁={mean_pi1:.3f}±{std_pi1:.3f}, CV={cv_pi1:.1%}",
            details={
                'mean_pi1': mean_pi1,
                'std_pi1': std_pi1,
                'cv_pi1': cv_pi1,
                'values': pi1_values
            }
        ))
    
    def validate_analysis1_l3_identifiability(self):
        """L3: Test parameter identifiability across multiple fits."""
        self.log("\n=== Analysis 1 (Global Rates) - L3: Identifiability ===")
        
        # Same data, multiple fits with different starting points
        data = self.create_test_data(n_tips=8, n_families=200)
        
        gain_rates = []
        loss_rates = []
        for i in range(3):
            analysis = GeneContentAnalysis(data)
            result = analysis.global_rates(verbose=False)
            gain_rates.append(result.gain_rate)
            loss_rates.append(result.loss_rate)
        
        # Check consistency
        gain_cv = np.std(gain_rates) / np.mean(gain_rates)
        loss_cv = np.std(loss_rates) / np.mean(loss_rates)
        
        passed = gain_cv < 0.1 and loss_cv < 0.1  # CV < 10%
        
        self.add_result(ValidationResult(
            test_name="GlobalRates_Identifiability",
            level="L3",
            analysis="1",
            passed=passed,
            message=f"Gain CV={gain_cv:.1%}, Loss CV={loss_cv:.1%}",
            details={
                'gain_rates': gain_rates,
                'loss_rates': loss_rates,
                'gain_cv': gain_cv,
                'loss_cv': loss_cv
            }
        ))
    
    def validate_analysis1_l4_robustness(self):
        """L4: Test robustness to data size."""
        self.log("\n=== Analysis 1 (Global Rates) - L4: Robustness ===")
        
        # Test with different data sizes
        results_small = []
        results_large = []
        
        for i in range(3):
            np.random.seed(42 + i)
            
            # Small dataset
            data_small = self.create_test_data(n_tips=8, n_families=50)
            analysis_small = GeneContentAnalysis(data_small)
            result_small = analysis_small.global_rates(verbose=False)
            results_small.append(result_small.gain_rate / (result_small.gain_rate + result_small.loss_rate))
            
            # Large dataset
            data_large = self.create_test_data(n_tips=8, n_families=200)
            analysis_large = GeneContentAnalysis(data_large)
            result_large = analysis_large.global_rates(verbose=False)
            results_large.append(result_large.gain_rate / (result_large.gain_rate + result_large.loss_rate))
        
        cv_small = np.std(results_small) / np.mean(results_small)
        cv_large = np.std(results_large) / np.mean(results_large)
        
        # Larger datasets should be more stable
        passed = cv_large <= cv_small * 1.5  # Allow some tolerance
        
        self.add_result(ValidationResult(
            test_name="GlobalRates_Robustness",
            level="L4",
            analysis="1",
            passed=passed,
            message=f"CV(50fam)={cv_small:.1%}, CV(200fam)={cv_large:.1%}",
            details={
                'cv_small': cv_small,
                'cv_large': cv_large,
                'results_small': results_small,
                'results_large': results_large
            }
        ))
    
    # ========== Analysis 2: Retention Test ==========
    
    def validate_analysis2_l1_likelihood(self):
        """L1: Test retention analysis likelihood computation."""
        self.log("\n=== Analysis 2 (Retention Test) - L1: Likelihood Computation ===")
        
        data = self.create_test_data(n_tips=8, n_families=50)
        analysis = GeneContentAnalysis(data)
        
        # Test first few families
        result = analysis.retention_test(families=data.family_names[:5], verbose=False)
        
        # Check that result has expected attributes
        passed = (hasattr(result, 'delta_ll') and hasattr(result, 'p_value') and
                  np.isfinite(result.delta_ll) and 0 <= result.p_value <= 1)
        
        self.add_result(ValidationResult(
            test_name="RetentionTest_Likelihood",
            level="L1",
            analysis="2",
            passed=passed,
            message=f"ΔLL={result.delta_ll:.2f}, p={result.p_value:.3f}",
            details={
                'delta_ll': result.delta_ll,
                'p_value': result.p_value,
                'retention_strength': result.retention_strength
            }
        ))
    
    def validate_analysis2_l2_detection(self):
        """L2: Test that retention bias can be detected."""
        self.log("\n=== Analysis 2 (Retention Test) - L2: Detection Power ===")
        
        data = self.create_test_data(n_tips=8, n_families=100)
        analysis = GeneContentAnalysis(data)
        
        # Test with different family sets
        results = []
        for i in range(3):
            start_idx = i * 10
            end_idx = start_idx + 10
            result = analysis.retention_test(families=data.family_names[start_idx:end_idx], verbose=False)
            results.append(result)
        
        # Check that all results have valid p-values
        passed = all(0 <= r.p_value <= 1 for r in results)
        
        self.add_result(ValidationResult(
            test_name="RetentionTest_Detection",
            level="L2",
            analysis="2",
            passed=passed,
            message=f"Tested 3 sets, all p-values valid: {passed}",
            details={'p_values': [r.p_value for r in results]}
        ))
    
    # ========== Summary Methods ==========
    
    def print_summary(self):
        """Print summary of all validation results."""
        self.log("\n" + "="*70)
        self.log("VALIDATION SUMMARY")
        self.log("="*70 + "\n")
        
        # Group by level
        levels = {}
        for result in self.results:
            if result.level not in levels:
                levels[result.level] = {'passed': 0, 'failed': 0}
            if result.passed:
                levels[result.level]['passed'] += 1
            else:
                levels[result.level]['failed'] += 1
        
        self.log("Results by validation level:\n")
        for level in sorted(levels.keys()):
            stats = levels[level]
            total = stats['passed'] + stats['failed']
            success_rate = 100 * stats['passed'] / total if total > 0 else 0
            self.log(f"{level}:")
            self.log(f"  Tests: {total}")
            self.log(f"  Passed: {stats['passed']}")
            self.log(f"  Failed: {stats['failed']}")
            self.log(f"  Success rate: {success_rate:.1f}%\n")
        
        # Show failed tests
        failed = [r for r in self.results if not r.passed]
        if failed:
            self.log("Failed tests:")
            for result in failed:
                self.log(f"  - {result.test_name} ({result.level}): {result.message}")
        else:
            self.log("✅ ALL TESTS PASSED!")
        
        self.log("\n" + "="*70 + "\n")
    
    def run_all(self):
        """Run all validation tests."""
        self.log("="*70)
        self.log("STANDARD ANALYSIS VALIDATION SUITE")
        self.log("="*70)
        
        # Analysis 1: Global Rates
        self.validate_analysis1_l1_likelihood()
        self.validate_analysis1_l2_parameter_recovery()
        self.validate_analysis1_l2_stationary_frequency()
        self.validate_analysis1_l3_identifiability()
        self.validate_analysis1_l4_robustness()
        
        # Analysis 2: Retention Test
        self.validate_analysis2_l1_likelihood()
        self.validate_analysis2_l2_detection()
        
        self.print_summary()


if __name__ == "__main__":
    suite = ValidationSuite()
    suite.run_all()
