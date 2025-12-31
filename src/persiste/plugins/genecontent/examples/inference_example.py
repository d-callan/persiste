"""
GeneContent Plugin: Full Inference Example

Demonstrates the complete inference workflow:
1. Load data (tree + presence/absence matrix)
2. Fit null model (global rates)
3. Fit alternative model (with constraint)
4. Likelihood ratio test
5. Model comparison

This shows how the genecontent plugin uses core framework utilities
for MLE optimization and hypothesis testing.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from persiste.core.trees import TreeStructure
from persiste.core.tree_inference import likelihood_ratio_test, model_selection

from persiste.plugins.genecontent import (
    GeneContentData,
    GeneContentModel,
    GeneContentInference,
)
from persiste.plugins.genecontent.constraints.gene_constraint import (
    NullConstraint,
    RetentionBiasConstraint,
)


def create_synthetic_data(
    n_taxa: int = 8,
    n_families: int = 50,
    true_gain: float = 0.3,
    true_loss: float = 0.5,
    retained_fraction: float = 0.2,
    retention_strength: float = -1.5,
    seed: int = 42,
) -> tuple:
    """
    Create synthetic gene content data with known parameters.
    
    Some families are "retained" (lower loss rate), others are neutral.
    This allows us to test if inference can recover the signal.
    
    Returns:
        (data, retained_families, true_params)
    """
    np.random.seed(seed)
    
    # Create a random tree
    newick = create_random_tree(n_taxa)
    tree = TreeStructure.from_newick(newick, backend="simple")
    
    # Generate family names
    family_names = [f"OG{i:04d}" for i in range(n_families)]
    taxon_names = tree.tip_names
    
    # Select retained families
    n_retained = int(n_families * retained_fraction)
    retained_families = set(family_names[:n_retained])
    
    # Simulate presence/absence data
    # For simplicity, use equilibrium frequencies
    presence_matrix = np.zeros((n_taxa, n_families), dtype=np.int8)
    
    for fam_idx, fam_name in enumerate(family_names):
        if fam_name in retained_families:
            # Retained: lower loss rate -> higher presence probability
            effective_loss = true_loss * np.exp(retention_strength)
            pi_present = true_gain / (true_gain + effective_loss)
        else:
            # Neutral
            pi_present = true_gain / (true_gain + true_loss)
        
        # Sample presence at tips (simplified: independent samples)
        presence_matrix[:, fam_idx] = np.random.binomial(1, pi_present, n_taxa)
    
    # Create data object
    data = GeneContentData(
        tree=tree,
        presence_matrix=presence_matrix,
        taxon_names=taxon_names,
        family_names=family_names,
    )
    
    true_params = {
        'gain_rate': true_gain,
        'loss_rate': true_loss,
        'retention_strength': retention_strength,
    }
    
    return data, retained_families, true_params


def create_random_tree(n_taxa: int) -> str:
    """Create a random Newick tree string."""
    if n_taxa <= 1:
        return "A:0.1;"
    
    # Generate taxon names
    taxa = [chr(ord('A') + i) if i < 26 else f"T{i}" for i in range(n_taxa)]
    
    # Build tree by random joining
    np.random.seed(42)
    nodes = [(t, 0.0) for t in taxa]  # (newick_str, depth)
    
    while len(nodes) > 1:
        # Pick two random nodes to join
        i, j = np.random.choice(len(nodes), 2, replace=False)
        if i > j:
            i, j = j, i
        
        n1, d1 = nodes.pop(j)
        n2, d2 = nodes.pop(i)
        
        # Random branch lengths
        bl1 = np.random.exponential(0.1)
        bl2 = np.random.exponential(0.1)
        
        # Join
        new_node = f"({n1}:{bl1:.4f},{n2}:{bl2:.4f})"
        new_depth = max(d1, d2) + 1
        nodes.append((new_node, new_depth))
    
    return nodes[0][0] + ":0.0;"


def demo_basic_inference():
    """Demonstrate basic MLE inference."""
    print("=" * 70)
    print("1. BASIC MLE INFERENCE")
    print("=" * 70)
    
    # Create synthetic data
    data, retained_families, true_params = create_synthetic_data(
        n_taxa=8,
        n_families=30,
        true_gain=0.3,
        true_loss=0.5,
    )
    
    print(f"\nData: {data}")
    print(f"True parameters: gain={true_params['gain_rate']:.3f}, loss={true_params['loss_rate']:.3f}")
    
    # Create inference engine
    inference = GeneContentInference(data, use_jax=False)
    
    # Fit null model
    print("\nFitting null model (global rates)...")
    null_result = inference.fit_null()
    
    print(f"\nNull model result:")
    print(f"  {null_result}")
    print(f"  Fitted gain rate: {np.exp(null_result.parameters['log_gain']):.4f}")
    print(f"  Fitted loss rate: {np.exp(null_result.parameters['log_loss']):.4f}")
    
    if null_result.standard_errors:
        print(f"  SE(log_gain): {null_result.standard_errors.get('log_gain', 'N/A'):.4f}")
        print(f"  SE(log_loss): {null_result.standard_errors.get('log_loss', 'N/A'):.4f}")
    
    return data, retained_families, true_params, null_result


def demo_constraint_testing(data, retained_families, true_params, null_result):
    """Demonstrate constraint hypothesis testing."""
    print("\n" + "=" * 70)
    print("2. CONSTRAINT HYPOTHESIS TESTING")
    print("=" * 70)
    
    print(f"\nTrue retained families: {sorted(retained_families)[:5]}... ({len(retained_families)} total)")
    print(f"True retention strength: {true_params['retention_strength']:.3f}")
    
    # Create inference engine
    inference = GeneContentInference(data, use_jax=False)
    
    # Test with correct retained families
    print("\n--- Testing with CORRECT retained families ---")
    correct_constraint = RetentionBiasConstraint(
        retained_families=retained_families,
        retention_strength=-1.0,  # Initial value
    )
    
    alt_result = inference.fit_with_constraint(correct_constraint)
    
    print(f"\nAlternative model result:")
    print(f"  {alt_result}")
    print(f"  Fitted gain rate: {np.exp(alt_result.parameters['log_gain']):.4f}")
    print(f"  Fitted loss rate: {np.exp(alt_result.parameters['log_loss']):.4f}")
    
    # LRT
    lrt = inference.likelihood_ratio_test(null_result, alt_result)
    print(f"\nLikelihood Ratio Test:")
    print(f"  {lrt}")
    print(f"  Conclusion: {'Reject null (constraint significant)' if lrt.significant else 'Fail to reject null'}")
    
    # Test with WRONG retained families
    print("\n--- Testing with WRONG retained families ---")
    wrong_families = set(data.family_names[-10:])  # Last 10 families (not actually retained)
    wrong_constraint = RetentionBiasConstraint(
        retained_families=wrong_families,
        retention_strength=-1.0,
    )
    
    wrong_result = inference.fit_with_constraint(wrong_constraint)
    wrong_lrt = inference.likelihood_ratio_test(null_result, wrong_result)
    
    print(f"\nWrong constraint LRT:")
    print(f"  {wrong_lrt}")
    print(f"  Conclusion: {'Reject null' if wrong_lrt.significant else 'Fail to reject null (as expected)'}")
    
    return alt_result, lrt


def demo_model_comparison():
    """Demonstrate model comparison using AIC/BIC."""
    print("\n" + "=" * 70)
    print("3. MODEL COMPARISON (AIC/BIC)")
    print("=" * 70)
    
    # Create data with moderate signal
    data, retained_families, true_params = create_synthetic_data(
        n_taxa=8,
        n_families=50,
        true_gain=0.3,
        true_loss=0.5,
        retained_fraction=0.3,
        retention_strength=-2.0,  # Strong signal
    )
    
    inference = GeneContentInference(data, use_jax=False)
    
    # Fit multiple models
    print("\nFitting multiple models...")
    
    # Model 1: Null
    null_result = inference.fit_null()
    print(f"  Null: LL={null_result.log_likelihood:.2f}, AIC={null_result.aic:.2f}, BIC={null_result.bic:.2f}")
    
    # Model 2: Correct constraint
    correct_constraint = RetentionBiasConstraint(
        retained_families=retained_families,
        retention_strength=-1.0,
    )
    correct_result = inference.fit_with_constraint(correct_constraint)
    print(f"  Correct: LL={correct_result.log_likelihood:.2f}, AIC={correct_result.aic:.2f}, BIC={correct_result.bic:.2f}")
    
    # Model 3: Wrong constraint
    wrong_families = set(data.family_names[-15:])
    wrong_constraint = RetentionBiasConstraint(
        retained_families=wrong_families,
        retention_strength=-1.0,
    )
    wrong_result = inference.fit_with_constraint(wrong_constraint)
    print(f"  Wrong: LL={wrong_result.log_likelihood:.2f}, AIC={wrong_result.aic:.2f}, BIC={wrong_result.bic:.2f}")
    
    # Model selection
    results = [null_result, correct_result, wrong_result]
    model_names = ["Null", "Correct constraint", "Wrong constraint"]
    
    best_aic_idx, best_aic = model_selection(results, criterion="AIC")
    best_bic_idx, best_bic = model_selection(results, criterion="BIC")
    
    print(f"\nModel selection:")
    print(f"  Best by AIC: {model_names[best_aic_idx]} (AIC={best_aic.aic:.2f})")
    print(f"  Best by BIC: {model_names[best_bic_idx]} (BIC={best_bic.bic:.2f})")


def demo_full_workflow():
    """Demonstrate the complete analysis workflow."""
    print("\n" + "=" * 70)
    print("4. COMPLETE ANALYSIS WORKFLOW")
    print("=" * 70)
    
    # Create realistic-ish data
    data, retained_families, true_params = create_synthetic_data(
        n_taxa=12,
        n_families=100,
        true_gain=0.2,
        true_loss=0.4,
        retained_fraction=0.15,
        retention_strength=-1.8,
        seed=123,
    )
    
    print(f"\nDataset: {data.n_taxa} taxa, {data.n_families} gene families")
    print(f"Retained families: {len(retained_families)} ({100*len(retained_families)/data.n_families:.0f}%)")
    
    # Full inference
    inference = GeneContentInference(data, use_jax=False)
    
    constraint = RetentionBiasConstraint(
        retained_families=retained_families,
        retention_strength=-1.0,
    )
    
    print("\nRunning fit_and_test()...")
    null_result, alt_result, lrt = inference.fit_and_test(constraint)
    
    print("\n" + "-" * 50)
    print("RESULTS SUMMARY")
    print("-" * 50)
    
    print(f"\nNull model (global rates):")
    print(f"  Gain rate: {np.exp(null_result.parameters['log_gain']):.4f}")
    print(f"  Loss rate: {np.exp(null_result.parameters['log_loss']):.4f}")
    print(f"  Log-likelihood: {null_result.log_likelihood:.4f}")
    print(f"  AIC: {null_result.aic:.2f}")
    
    print(f"\nAlternative model (retention constraint):")
    print(f"  Gain rate: {np.exp(alt_result.parameters['log_gain']):.4f}")
    print(f"  Loss rate: {np.exp(alt_result.parameters['log_loss']):.4f}")
    print(f"  Log-likelihood: {alt_result.log_likelihood:.4f}")
    print(f"  AIC: {alt_result.aic:.2f}")
    
    print(f"\nLikelihood Ratio Test:")
    print(f"  Statistic: {lrt.statistic:.4f}")
    print(f"  df: {lrt.df}")
    print(f"  p-value: {lrt.pvalue:.4e}")
    print(f"  Significant at α=0.05: {'Yes' if lrt.significant else 'No'}")
    
    print("\n" + "-" * 50)
    if lrt.significant:
        print("CONCLUSION: Evidence for selective retention in specified families")
    else:
        print("CONCLUSION: No significant evidence for selective retention")
    print("-" * 50)


if __name__ == '__main__':
    print("=" * 70)
    print("GeneContent Plugin: Full Inference Example")
    print("=" * 70)
    print("\nThis demonstrates how the genecontent plugin uses core utilities:")
    print("  - TreeStructure for tree representation")
    print("  - FelsensteinPruning for likelihood computation")
    print("  - TreeMLEOptimizer for parameter estimation")
    print("  - likelihood_ratio_test for hypothesis testing")
    
    # Run demos
    data, retained_families, true_params, null_result = demo_basic_inference()
    demo_constraint_testing(data, retained_families, true_params, null_result)
    demo_model_comparison()
    demo_full_workflow()
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
