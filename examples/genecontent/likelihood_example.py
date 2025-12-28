"""
GeneContent Plugin: Likelihood Computation Example

Demonstrates using the core framework's pruning algorithm
to compute gene family likelihoods on a phylogenetic tree.

This shows how plugins can reuse core utilities:
1. TreeStructure - generic tree representation
2. FelsensteinPruning - generic pruning algorithm
3. SimpleBinaryTransitionProvider - 2-state CTMC rates
4. ArrayTipConditionalProvider - tip observations
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from persiste.core.trees import TreeStructure
from persiste.core.pruning import (
    FelsensteinPruning,
    SimpleBinaryTransitionProvider,
    ArrayTipConditionalProvider,
)

from persiste.plugins.genecontent.baselines.gene_baseline import (
    HierarchicalRates,
    GlobalRates,
)
from persiste.plugins.genecontent.constraints.gene_constraint import (
    NullConstraint,
    RetentionBiasConstraint,
)


def demo_tree_parsing():
    """Demonstrate tree parsing with core utilities."""
    print("=" * 60)
    print("1. TREE PARSING (Core Framework)")
    print("=" * 60)
    
    # Simple Newick tree
    newick = "((A:0.1,B:0.2):0.15,(C:0.3,D:0.1):0.25):0.0;"
    
    # Parse using core TreeStructure (no external dependencies)
    tree = TreeStructure.from_newick(newick, backend="simple")
    
    print(f"\nParsed tree: {tree}")
    print(f"  Tips: {tree.tip_names}")
    print(f"  Root index: {tree.root_index}")
    print(f"  Postorder: {tree.postorder}")
    print(f"  Branch lengths: {tree.branch_lengths}")
    
    return tree


def demo_binary_likelihood(tree: TreeStructure):
    """Demonstrate likelihood computation for binary (presence/absence) model."""
    print("\n" + "=" * 60)
    print("2. BINARY LIKELIHOOD (Gene Presence/Absence)")
    print("=" * 60)
    
    # Observed gene presence/absence at tips
    # Rows: taxa (A, B, C, D), Columns: gene families
    data = np.array([
        [1, 0, 1, 1, 0],  # A
        [1, 1, 1, 0, 0],  # B
        [0, 0, 1, 1, 1],  # C
        [1, 1, 0, 1, 0],  # D
    ], dtype=np.int8)
    
    taxon_names = ['A', 'B', 'C', 'D']
    family_names = ['OG0001', 'OG0002', 'OG0003', 'OG0004', 'OG0005']
    n_families = len(family_names)
    
    print(f"\nObserved data ({len(taxon_names)} taxa, {n_families} families):")
    for i, taxon in enumerate(taxon_names):
        print(f"  {taxon}: {data[i]}")
    
    # Setup pruning algorithm
    pruning = FelsensteinPruning(tree, n_states=2, use_jax=False)
    
    # Compute likelihood for each family with different rates
    print("\nPer-family log-likelihoods:")
    print(f"{'Family':<10} {'Gain':<8} {'Loss':<8} {'Log-Lik':<12}")
    print("-" * 40)
    
    total_log_lik = 0.0
    
    for fam_idx, fam_name in enumerate(family_names):
        # Use global rates for simplicity
        gain_rate = 0.5
        loss_rate = 0.3
        
        # Create transition provider
        transition_provider = SimpleBinaryTransitionProvider(
            gain_rate=gain_rate,
            loss_rate=loss_rate,
        )
        
        # Create tip conditional provider (single family)
        single_family_data = data[:, fam_idx:fam_idx+1]
        tip_provider = ArrayTipConditionalProvider(
            data=single_family_data,
            taxon_names=taxon_names,
            n_states=2,
        )
        
        # Compute likelihood
        result = pruning.compute_likelihood(
            transition_provider=transition_provider,
            tip_provider=tip_provider,
            n_sites=1,
        )
        
        print(f"{fam_name:<10} {gain_rate:<8.2f} {loss_rate:<8.2f} {result.log_likelihood:<12.4f}")
        total_log_lik += result.log_likelihood
    
    print("-" * 40)
    print(f"{'Total':<10} {'':<8} {'':<8} {total_log_lik:<12.4f}")
    
    return total_log_lik


def demo_constraint_comparison(tree: TreeStructure):
    """Demonstrate comparing null vs constrained models."""
    print("\n" + "=" * 60)
    print("3. CONSTRAINT COMPARISON (Null vs Retention Bias)")
    print("=" * 60)
    
    # Observed data
    data = np.array([
        [1, 0, 1, 1, 0],  # A
        [1, 1, 1, 0, 0],  # B
        [0, 0, 1, 1, 1],  # C
        [1, 1, 0, 1, 0],  # D
    ], dtype=np.int8)
    
    taxon_names = ['A', 'B', 'C', 'D']
    family_names = ['OG0001', 'OG0002', 'OG0003', 'OG0004', 'OG0005']
    
    # Setup pruning
    pruning = FelsensteinPruning(tree, n_states=2, use_jax=False)
    
    # Baseline rates
    baseline = GlobalRates(gain_rate=0.5, loss_rate=0.3)
    
    # Null constraint (no effect)
    null_constraint = NullConstraint()
    
    # Retention bias constraint (OG0001, OG0003 are retained)
    retention_constraint = RetentionBiasConstraint(
        retained_families={'OG0001', 'OG0003'},
        retention_strength=-1.0,  # ~2.7x reduction in loss
    )
    
    print("\nComparing models:")
    print("  Null: All families have same gain/loss rates")
    print("  Retention: OG0001, OG0003 have reduced loss rate")
    
    # Compute likelihoods under both models
    null_log_lik = 0.0
    retention_log_lik = 0.0
    
    print(f"\n{'Family':<10} {'Null LL':<12} {'Retention LL':<12} {'Δ LL':<10}")
    print("-" * 50)
    
    for fam_idx, fam_name in enumerate(family_names):
        # Get baseline rates
        rates = baseline.get_rates(fam_name)
        
        # Null model: use baseline rates directly
        null_provider = SimpleBinaryTransitionProvider(
            gain_rate=rates.gain_rate,
            loss_rate=rates.loss_rate,
        )
        
        # Retention model: apply constraint
        effect = retention_constraint.get_effect(fam_name)
        retention_provider = SimpleBinaryTransitionProvider(
            gain_rate=rates.gain_rate * effect.gain_multiplier,
            loss_rate=rates.loss_rate * effect.loss_multiplier,
        )
        
        # Tip data
        single_family_data = data[:, fam_idx:fam_idx+1]
        tip_provider = ArrayTipConditionalProvider(
            data=single_family_data,
            taxon_names=taxon_names,
            n_states=2,
        )
        
        # Compute likelihoods
        null_result = pruning.compute_likelihood(
            transition_provider=null_provider,
            tip_provider=tip_provider,
            n_sites=1,
        )
        
        retention_result = pruning.compute_likelihood(
            transition_provider=retention_provider,
            tip_provider=tip_provider,
            n_sites=1,
        )
        
        delta = retention_result.log_likelihood - null_result.log_likelihood
        
        print(f"{fam_name:<10} {null_result.log_likelihood:<12.4f} {retention_result.log_likelihood:<12.4f} {delta:<10.4f}")
        
        null_log_lik += null_result.log_likelihood
        retention_log_lik += retention_result.log_likelihood
    
    print("-" * 50)
    delta_total = retention_log_lik - null_log_lik
    print(f"{'Total':<10} {null_log_lik:<12.4f} {retention_log_lik:<12.4f} {delta_total:<10.4f}")
    
    # LRT
    print("\n" + "-" * 50)
    print("Likelihood Ratio Test:")
    lrt_statistic = 2 * delta_total
    print(f"  LRT statistic: 2 × Δ LL = {lrt_statistic:.4f}")
    print(f"  df = 1 (retention_strength parameter)")
    
    # p-value from chi-squared
    from scipy import stats
    pvalue = stats.chi2.sf(abs(lrt_statistic), df=1)
    print(f"  p-value: {pvalue:.4e}")
    
    if lrt_statistic > 0:
        print(f"\n  → Retention model {'fits better' if pvalue < 0.05 else 'not significantly better'}")
    else:
        print(f"\n  → Null model fits better (retention constraint not supported)")


def demo_hierarchical_baseline(tree: TreeStructure):
    """Demonstrate hierarchical baseline with per-family rate variation."""
    print("\n" + "=" * 60)
    print("4. HIERARCHICAL BASELINE (Per-Family Rate Variation)")
    print("=" * 60)
    
    # Observed data
    data = np.array([
        [1, 0, 1, 1, 0],  # A
        [1, 1, 1, 0, 0],  # B
        [0, 0, 1, 1, 1],  # C
        [1, 1, 0, 1, 0],  # D
    ], dtype=np.int8)
    
    taxon_names = ['A', 'B', 'C', 'D']
    family_names = ['OG0001', 'OG0002', 'OG0003', 'OG0004', 'OG0005']
    
    # Setup pruning
    pruning = FelsensteinPruning(tree, n_states=2, use_jax=False)
    
    # Hierarchical baseline (rates vary per family)
    np.random.seed(42)  # For reproducibility
    hierarchical = HierarchicalRates(
        mu_gain=-1.5,
        sigma_gain=0.5,
        mu_loss=-1.0,
        sigma_loss=0.5,
    )
    
    # Get rates for all families
    rates = hierarchical.get_all_rates(family_names)
    
    print("\nHierarchical per-family rates:")
    print(f"{'Family':<10} {'Gain':<10} {'Loss':<10} {'Log-Lik':<12}")
    print("-" * 45)
    
    total_log_lik = 0.0
    
    for fam_idx, fam_name in enumerate(family_names):
        r = rates[fam_name]
        
        # Create transition provider with family-specific rates
        transition_provider = SimpleBinaryTransitionProvider(
            gain_rate=r.gain_rate,
            loss_rate=r.loss_rate,
        )
        
        # Tip data
        single_family_data = data[:, fam_idx:fam_idx+1]
        tip_provider = ArrayTipConditionalProvider(
            data=single_family_data,
            taxon_names=taxon_names,
            n_states=2,
        )
        
        # Compute likelihood
        result = pruning.compute_likelihood(
            transition_provider=transition_provider,
            tip_provider=tip_provider,
            n_sites=1,
        )
        
        print(f"{fam_name:<10} {r.gain_rate:<10.4f} {r.loss_rate:<10.4f} {result.log_likelihood:<12.4f}")
        total_log_lik += result.log_likelihood
    
    print("-" * 45)
    print(f"{'Total':<10} {'':<10} {'':<10} {total_log_lik:<12.4f}")
    
    # Add log-prior for hierarchical model
    log_prior = hierarchical.log_prior()
    print(f"\nLog-prior (hierarchical): {log_prior:.4f}")
    print(f"Log-posterior: {total_log_lik + log_prior:.4f}")


if __name__ == '__main__':
    print("=" * 60)
    print("GeneContent Plugin: Likelihood Computation Example")
    print("=" * 60)
    print("\nThis demonstrates how plugins use core framework utilities:")
    print("  - TreeStructure: generic tree representation")
    print("  - FelsensteinPruning: generic pruning algorithm")
    print("  - SimpleBinaryTransitionProvider: 2-state CTMC")
    print("  - ArrayTipConditionalProvider: tip observations")
    
    # Parse tree
    tree = demo_tree_parsing()
    
    # Compute likelihoods
    demo_binary_likelihood(tree)
    
    # Compare null vs constrained
    demo_constraint_comparison(tree)
    
    # Hierarchical baseline
    demo_hierarchical_baseline(tree)
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Core framework provides reusable tree/pruning utilities")
    print("  2. Plugins provide domain-specific rates and constraints")
    print("  3. Same pruning algorithm works for phylo, genecontent, etc.")
    print("  4. LRT for hypothesis testing is straightforward")
