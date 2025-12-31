#!/usr/bin/env python
"""Diagnostic script to understand null recovery bias."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from persiste.core.trees import TreeStructure
from persiste.plugins.genecontent.inference.gene_inference import (
    GeneContentData,
    GeneContentInference,
)
from persiste.plugins.genecontent.constraints.gene_constraint import (
    NullConstraint,
    RetentionBiasConstraint,
)


def diagnose_null_recovery():
    """Diagnose why null recovery is failing."""
    
    print("=" * 80)
    print("DIAGNOSING NULL RECOVERY BIAS")
    print("=" * 80)
    print()
    
    # Create tree
    newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    taxon_names = ["A", "B", "C", "D"]
    
    # Simulate data with NO retention bias
    print("Simulating data with NO retention bias (all families equal)...")
    print()
    
    from scipy.linalg import expm
    rng = np.random.default_rng(42)
    
    gain_rate = 2.0
    loss_rate = 3.0
    n_families = 30
    
    # Equilibrium frequencies
    pi_0 = loss_rate / (gain_rate + loss_rate)
    pi_1 = gain_rate / (gain_rate + loss_rate)
    
    presence_matrix = np.zeros((4, n_families), dtype=int)
    
    # Build rate matrix (same for all families)
    Q = np.array([
        [-gain_rate, gain_rate],
        [loss_rate, -loss_rate]
    ])
    
    for fam_idx in range(n_families):
        # Sample root state
        root_state = rng.choice([0, 1], p=[pi_0, pi_1])
        
        # Simulate down tree using edge-based traversal
        node_states = {tree.root_index: root_state}
        
        for child_idx in range(tree.n_nodes):
            parent_idx = tree.parent_indices[child_idx]
            if parent_idx >= 0:  # Not root
                parent_state = node_states[parent_idx]
                t = tree.branch_lengths[child_idx]
                P = expm(Q * t)
                child_state = rng.choice([0, 1], p=P[parent_state, :])
                node_states[child_idx] = child_state
        
        # Extract tip states
        for tip_idx_pos, tip_idx in enumerate(tree.tip_indices):
            presence_matrix[tip_idx_pos, fam_idx] = node_states[tip_idx]
    
    family_names = ["fam{0}".format(i) for i in range(n_families)]
    
    # Check data properties
    print("Data properties:")
    print("  Total families: {0}".format(n_families))
    print("  Families present in all tips: {0}".format(np.sum(presence_matrix.sum(axis=0) == 4)))
    print("  Families absent in all tips: {0}".format(np.sum(presence_matrix.sum(axis=0) == 0)))
    print("  Average presence per family: {0:.2f}".format(presence_matrix.mean(axis=0).mean()))
    print()
    
    gene_data = GeneContentData(
        tree=tree,
        presence_matrix=presence_matrix,
        taxon_names=taxon_names,
        family_names=family_names,
    )
    
    # Test 1: Fit null model (should recover true rates)
    print("Test 1: Fit null model (no constraints)")
    print("-" * 80)
    
    inference = GeneContentInference(gene_data)
    null_result = inference.fit_null()
    
    print("  True gain rate:      {0:.4f}".format(gain_rate))
    print("  Estimated gain rate: {0:.4f}".format(np.exp(null_result.parameters['log_gain'])))
    print("  True loss rate:      {0:.4f}".format(loss_rate))
    print("  Estimated loss rate: {0:.4f}".format(np.exp(null_result.parameters['log_loss'])))
    print("  Log-likelihood:      {0:.4f}".format(null_result.log_likelihood))
    print()
    
    # Test 2: Fit model with retention constraint on subset of families
    print("Test 2: Fit model with retention constraint on fam0, fam1, fam2")
    print("-" * 80)
    
    retained_families = {"fam0", "fam1", "fam2"}
    constraint = RetentionBiasConstraint(retained_families=retained_families)
    
    alt_result = inference.fit_with_constraint(constraint)
    
    print("  True retention strength: 0.0 (no effect)")
    print("  Estimated retention:     {0:.4f}".format(alt_result.parameters['retention_strength']))
    print("  Estimated gain rate:     {0:.4f}".format(np.exp(alt_result.parameters['log_gain'])))
    print("  Estimated loss rate:     {0:.4f}".format(np.exp(alt_result.parameters['log_loss'])))
    print("  Log-likelihood:          {0:.4f}".format(alt_result.log_likelihood))
    print()
    
    # Test 3: Compare likelihoods
    print("Test 3: Likelihood comparison")
    print("-" * 80)
    
    ll_diff = alt_result.log_likelihood - null_result.log_likelihood
    print("  Null LL:  {0:.4f}".format(null_result.log_likelihood))
    print("  Alt LL:   {0:.4f}".format(alt_result.log_likelihood))
    print("  Diff:     {0:.4f}".format(ll_diff))
    print()
    
    if ll_diff > 0.1:
        print("  WARNING: Alternative model has higher likelihood!")
        print("  This suggests the model is finding a spurious effect.")
    else:
        print("  OK: Null and alternative have similar likelihoods.")
    print()
    
    # Test 4: Check prior contribution
    print("Test 4: Prior contribution")
    print("-" * 80)
    
    prior_value = constraint.log_prior()
    print("  Prior on retention strength: {0:.4f}".format(prior_value))
    print("  Prior mean: {0}".format(constraint.prior_mean))
    print("  Prior std:  {0}".format(constraint.prior_std))
    print()
    
    # Test 5: Manual likelihood calculation
    print("Test 5: Manual likelihood at different retention values")
    print("-" * 80)
    
    from persiste.plugins.genecontent.inference.gene_inference import GeneContentModel
    
    model = GeneContentModel(data=gene_data, constraint=constraint)
    
    # Use ESTIMATED baseline rates from null model, not true simulation rates
    estimated_gain = np.exp(null_result.parameters['log_gain'])
    estimated_loss = np.exp(null_result.parameters['log_loss'])
    
    print("  Using estimated baseline rates:")
    print("    Gain: {0:.4f} (true: {1:.4f})".format(estimated_gain, gain_rate))
    print("    Loss: {0:.4f} (true: {1:.4f})".format(estimated_loss, loss_rate))
    print()
    
    test_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
    print("  Retention | Log-Likelihood | Prior | Total")
    print("  " + "-" * 50)
    
    for theta in test_values:
        params = {
            'log_gain': null_result.parameters['log_gain'],
            'log_loss': null_result.parameters['log_loss'],
            'retention_strength': theta,
        }
        
        # Compute likelihood without prior
        constraint.set_parameters({'retention_strength': theta})
        ll_data = 0.0
        
        # Recompute manually to separate data likelihood from prior
        from persiste.core.pruning import FelsensteinPruning
        from persiste.core.pruning import SimpleBinaryTransitionProvider
        from persiste.core.pruning import ArrayTipConditionalProvider
        
        pruning = FelsensteinPruning(tree, n_states=2, use_jax=False)
        
        for fam_idx, fam_name in enumerate(family_names):
            effect = constraint.get_effect(fam_name)
            effective_gain = estimated_gain * effect.gain_multiplier
            effective_loss = estimated_loss * effect.loss_multiplier
            
            transition_provider = SimpleBinaryTransitionProvider(
                gain_rate=effective_gain,
                loss_rate=effective_loss,
            )
            
            single_family_data = presence_matrix[:, fam_idx:fam_idx+1]
            tip_provider = ArrayTipConditionalProvider(
                data=single_family_data,
                taxon_names=taxon_names,
                n_states=2,
            )
            
            result = pruning.compute_likelihood(
                transition_provider=transition_provider,
                tip_provider=tip_provider,
                n_sites=1,
            )
            
            ll_data += result.log_likelihood
        
        prior = constraint.log_prior()
        total = ll_data + prior
        
        print("  {0:>8.2f}  | {1:>14.4f} | {2:>5.2f} | {3:>14.4f}".format(
            theta, ll_data, prior, total
        ))
    
    print()
    print("=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    diagnose_null_recovery()
