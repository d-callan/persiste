"""
Demo: CopyNumberDynamics Recipe Workflow

This script demonstrates the recommended workflow for analyzing copy number
evolution using the CopyNumberDynamics plugin recipes.

Workflow:
1. Recipe 0: Null CN Dynamics (descriptive baseline)
2. Diagnostic: Expected vs Observed CN
3. Recipe 1: Dosage Stability Scan
4. Recipe 2: Amplification Bias Test
5. Recipe 3: Lineage-Conditioned Volatility (optional)

Each recipe is self-contained and produces interpretable results.
"""

import numpy as np
from pathlib import Path

# Import recipes
from persiste.plugins.copynumber.recipes import (
    null_cn_dynamics,
    dosage_stability_scan,
    amplification_bias_test,
    lineage_volatility_test,
)

# Import diagnostics
from persiste.plugins.copynumber.diagnostics import (
    expected_vs_observed_cn,
    interpret_diagnostic,
)


def run_copynumber_workflow(
    cn_matrix_path: str,
    tree_path: str,
    output_dir: str = "results",
):
    """
    Complete CopyNumberDynamics analysis workflow.
    
    Args:
        cn_matrix_path: Path to copy number matrix file
        tree_path: Path to phylogenetic tree file (Newick format)
        output_dir: Directory for output files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("COPY NUMBER DYNAMICS ANALYSIS WORKFLOW")
    print("=" * 80)
    print(f"\nInput data: {cn_matrix_path}")
    print(f"Tree: {tree_path}")
    print(f"Output directory: {output_dir}\n")
    
    # =========================================================================
    # Step 1: Recipe 0 - Null CN Dynamics (Descriptive Baseline)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Recipe 0 - Null CN Dynamics (Descriptive Baseline)")
    print("=" * 80)
    print("\nQuestion: What does copy number evolution look like under no hypothesis?")
    print("This is your descriptive anchor - always run this first.\n")
    
    null_report = null_cn_dynamics(
        cn_matrix=cn_matrix_path,
        tree=tree_path,
        baseline_type='hierarchical',  # Recommended
        verbose=True,
    )
    
    # Save report
    with open(output_path / "recipe_0_null_dynamics.txt", 'w') as f:
        f.write(null_report.interpretation)
    
    print(f"\n✓ Report saved to: {output_path / 'recipe_0_null_dynamics.txt'}")
    
    # =========================================================================
    # Step 2: Diagnostic - Expected vs Observed CN
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Diagnostic - Expected vs Observed CN")
    print("=" * 80)
    print("\nCritical check: Does the null model fit the data?")
    print("If expected and observed CN diverge wildly, something is wrong.\n")
    
    # Generate diagnostic plot
    fig = expected_vs_observed_cn(
        cn_matrix=cn_matrix_path,
        tree=tree_path,
        baseline_type='hierarchical',
        save_path=str(output_path / "diagnostic_expected_vs_observed.png"),
    )
    
    # Get interpretation
    diagnostic_interp = interpret_diagnostic(
        cn_matrix=cn_matrix_path,
        tree=tree_path,
        baseline_type='hierarchical',
    )
    
    print(diagnostic_interp)
    
    # Save interpretation
    with open(output_path / "diagnostic_interpretation.txt", 'w') as f:
        f.write(diagnostic_interp)
    
    print(f"\n✓ Diagnostic plot saved to: {output_path / 'diagnostic_expected_vs_observed.png'}")
    print(f"✓ Interpretation saved to: {output_path / 'diagnostic_interpretation.txt'}")
    
    # =========================================================================
    # Step 3: Recipe 1 - Dosage Stability Scan
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Recipe 1 - Dosage Stability Scan")
    print("=" * 80)
    print("\nQuestion: Are copy number changes globally suppressed or enhanced?")
    print("This is the CN analogue of 'Are genes conserved or labile?'\n")
    
    dosage_report = dosage_stability_scan(
        cn_matrix=cn_matrix_path,
        tree=tree_path,
        baseline_type='hierarchical',
        verbose=True,
    )
    
    # Save report
    with open(output_path / "recipe_1_dosage_stability.txt", 'w') as f:
        f.write(dosage_report.interpretation)
        f.write("\n\n")
        f.write(dosage_report.recommendation)
    
    print(f"\n✓ Report saved to: {output_path / 'recipe_1_dosage_stability.txt'}")
    
    # Save summary
    with open(output_path / "recipe_1_summary.txt", 'w') as f:
        f.write(f"Dosage Stability Analysis Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"θ estimate: {dosage_report.theta:.3f}\n")
        f.write(f"95% CI: ({dosage_report.theta_ci[0]:.3f}, {dosage_report.theta_ci[1]:.3f})\n")
        f.write(f"p-value: {dosage_report.p_value:.4f}\n")
        f.write(f"Significant: {'Yes' if dosage_report.p_value < 0.05 else 'No'}\n")
        f.write(f"\nInterpretation:\n")
        if dosage_report.p_value < 0.05:
            if dosage_report.theta < -0.3:
                f.write("Strong dosage buffering detected (θ < -0.3)\n")
            elif dosage_report.theta > 0.3:
                f.write("Copy number volatility detected (θ > 0.3)\n")
            else:
                f.write("Weak dosage effect detected\n")
        else:
            f.write("No significant dosage effect\n")
    
    # =========================================================================
    # Step 4: Recipe 2 - Amplification Bias Test
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Recipe 2 - Amplification Bias Test")
    print("=" * 80)
    print("\nQuestion: Are increases favored over decreases beyond baseline?")
    print("Tests asymmetry between amplification and contraction.\n")
    
    # Only run if multi-copy genes are present
    if null_report.stationary_distribution[2] + null_report.stationary_distribution[3] > 0.05:
        print("Multi-copy genes detected (>5%) - running amplification bias test...\n")
        
        amp_report = amplification_bias_test(
            cn_matrix=cn_matrix_path,
            tree=tree_path,
            baseline_type='hierarchical',
            verbose=True,
        )
        
        # Save report
        with open(output_path / "recipe_2_amplification_bias.txt", 'w') as f:
            f.write(amp_report.interpretation)
            f.write("\n\n")
            f.write(amp_report.recommendation)
        
        print(f"\n✓ Report saved to: {output_path / 'recipe_2_amplification_bias.txt'}")
        
        # Save summary
        with open(output_path / "recipe_2_summary.txt", 'w') as f:
            f.write(f"Amplification Bias Analysis Summary\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"θ estimate: {amp_report.theta:.3f}\n")
            f.write(f"95% CI: ({amp_report.theta_ci[0]:.3f}, {amp_report.theta_ci[1]:.3f})\n")
            f.write(f"p-value: {amp_report.p_value:.4f}\n")
            f.write(f"Significant: {'Yes' if amp_report.p_value < 0.05 else 'No'}\n")
            f.write(f"\nRate multipliers:\n")
            f.write(f"  Amplification (1→2, 2→3): {np.exp(amp_report.theta):.2f}×\n")
            f.write(f"  Contraction (2→1, 3→2):   {np.exp(-amp_report.theta):.2f}×\n")
            f.write(f"\nInterpretation:\n")
            if amp_report.p_value < 0.05:
                if amp_report.theta > 0.3:
                    f.write("Amplification bias detected (θ > 0.3)\n")
                    f.write("Suggests: drug resistance, stress response, adaptive CNV\n")
                elif amp_report.theta < -0.3:
                    f.write("Contraction bias detected (θ < -0.3)\n")
                    f.write("Suggests: dosage constraint on multi-copy genes\n")
                else:
                    f.write("Weak asymmetry detected\n")
            else:
                f.write("No significant amplification bias\n")
    else:
        print("Multi-copy genes rare (<5%) - skipping amplification bias test.")
        print("(Amplification bias is only meaningful with multi-copy genes)\n")
    
    # =========================================================================
    # Step 5: Recipe 3 - Lineage-Conditioned Volatility (Optional)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Recipe 3 - Lineage-Conditioned Volatility (Optional)")
    print("=" * 80)
    print("\nQuestion: Do some clades experience elevated dosage turnover?")
    print("This is descriptive, not causal.\n")
    print("Note: This requires lineage annotations in your tree.")
    print("Skipping for this demo (requires metadata integration).\n")
    
    # In a real analysis with lineage annotations:
    # lineage_report = lineage_volatility_test(
    #     cn_matrix=cn_matrix_path,
    #     tree=tree_path,
    #     target_lineage="host_associated",
    #     baseline_type='hierarchical',
    #     verbose=True,
    # )
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  • recipe_0_null_dynamics.txt")
    print("  • diagnostic_expected_vs_observed.png")
    print("  • diagnostic_interpretation.txt")
    print("  • recipe_1_dosage_stability.txt")
    print("  • recipe_1_summary.txt")
    if null_report.stationary_distribution[2] + null_report.stationary_distribution[3] > 0.05:
        print("  • recipe_2_amplification_bias.txt")
        print("  • recipe_2_summary.txt")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("\nKey Results:")
    print(f"  • Expected CN at equilibrium: {null_report.expected_cn:.2f}")
    print(f"  • Dosage stability θ: {dosage_report.theta:.3f} (p={dosage_report.p_value:.4f})")
    
    if dosage_report.p_value < 0.05:
        if dosage_report.theta < -0.3:
            print("\n  → Strong dosage buffering detected")
            print("     Genes resist copy number changes")
            print("     Likely: essential genes, housekeeping functions")
        elif dosage_report.theta > 0.3:
            print("\n  → Copy number volatility detected")
            print("     Genes experience frequent CN changes")
            print("     Likely: antigen families, stress response")
        else:
            print("\n  → Weak dosage effect")
    else:
        print("\n  → No significant dosage constraint")
        print("     CN evolves according to baseline dynamics")
    
    print("\n" + "=" * 80)
    print("\nNext Steps:")
    print("  1. Review diagnostic plot - ensure model fits data")
    print("  2. Examine per-family results for heterogeneity")
    print("  3. Consider functional enrichment of high-θ families")
    print("  4. If amplification bias detected, investigate biological context")
    print("  5. For joint presence × dosage analysis, see Recipe 4 (future)")
    print("\n" + "=" * 80)


def minimal_example():
    """
    Minimal example showing the essential workflow.
    """
    print("=" * 80)
    print("MINIMAL COPYNUMBER WORKFLOW")
    print("=" * 80)
    
    # Paths to your data
    cn_matrix = "data/cn_matrix.tsv"
    tree = "data/tree.nwk"
    
    # Step 1: Descriptive baseline
    print("\n1. Null CN Dynamics (descriptive baseline)")
    null_report = null_cn_dynamics(cn_matrix, tree=tree)
    
    # Step 2: Diagnostic check
    print("\n2. Diagnostic: Expected vs Observed CN")
    fig = expected_vs_observed_cn(cn_matrix, tree=tree)
    
    # Step 3: Test for dosage stability
    print("\n3. Dosage Stability Scan")
    dosage_report = dosage_stability_scan(cn_matrix, tree=tree)
    
    # Step 4: Test for amplification bias (if multi-copy genes present)
    if null_report.stationary_distribution[2] + null_report.stationary_distribution[3] > 0.05:
        print("\n4. Amplification Bias Test")
        amp_report = amplification_bias_test(cn_matrix, tree=tree)
    
    print("\n" + "=" * 80)
    print("Done! Check the printed interpretations above.")
    print("=" * 80)


if __name__ == "__main__":
    # For demo purposes, you would run:
    # run_copynumber_workflow(
    #     cn_matrix_path="path/to/your/cn_matrix.tsv",
    #     tree_path="path/to/your/tree.nwk",
    #     output_dir="results/copynumber_analysis"
    # )
    
    print(__doc__)
    print("\nTo run this workflow on your data:")
    print("  python recipe_demo.py")
    print("\nOr import and use in your own scripts:")
    print("  from persiste.plugins.copynumber.examples.recipe_demo import run_copynumber_workflow")
    print("  run_copynumber_workflow('data/cn_matrix.tsv', 'data/tree.nwk')")
