"""
Main interface for copy number dynamics analysis.

This module provides the high-level API for fitting copy number models.
"""

from typing import Optional, Union, List, Dict, Any
import numpy as np
from pathlib import Path

from persiste.core.trees import Tree
from persiste.plugins.copynumber.states.cn_states import (
    CopyNumberState,
    get_sparse_transition_graph,
    validate_transition_matrix,
)
from persiste.plugins.copynumber.baselines.cn_baseline import (
    create_baseline,
    CopyNumberBaseline,
)
from persiste.plugins.copynumber.constraints.cn_constraints import (
    create_constraint,
    apply_constraint,
    CopyNumberConstraint,
)
from persiste.plugins.copynumber.observation.cn_observation import (
    create_observation_model,
    DeterministicBinObservation,
)
from persiste.plugins.copynumber.data.cn_data import (
    CopyNumberData,
    CopyNumberResult,
)


def compute_family_likelihood(
    family_data: np.ndarray,
    tree: Tree,
    rate_matrix: np.ndarray,
    obs_model: DeterministicBinObservation,
) -> float:
    """
    Compute likelihood for a single gene family.
    
    Uses Felsenstein pruning algorithm.
    
    Args:
        family_data: (n_taxa,) array of observed states at tips
        tree: Phylogenetic tree
        rate_matrix: (4, 4) rate matrix Q
        obs_model: Observation model
    
    Returns:
        Log-likelihood for this family
    """
    from scipy.linalg import expm
    
    # Get tip likelihoods from observation model
    tip_likelihoods = obs_model.get_tip_likelihoods_matrix(family_data)
    
    # Run Felsenstein pruning using the simple Tree interface
    # This is the production code path
    n_states = rate_matrix.shape[0]
    n_nodes = len(tree.nodes)
    
    # Initialize conditionals
    conditionals = {}
    
    # Set tip conditionals
    # Map leaf nodes to tip data indices
    leaves = tree.get_leaves()
    for taxon_idx, leaf_id in enumerate(leaves):
        conditionals[leaf_id] = tip_likelihoods[taxon_idx, :]
    
    # Postorder traversal
    def postorder(node_id):
        if tree.is_leaf(node_id):
            return
        
        children = tree.get_children(node_id)
        
        # Process children first
        for child_id in children:
            postorder(child_id)
        
        # Combine children
        if len(children) >= 2:
            child1_id = children[0]
            child2_id = children[1]
            
            # Get transition matrices P(t) = exp(Qt)
            t1 = tree.get_branch_length(child1_id)
            t2 = tree.get_branch_length(child2_id)
            
            P1 = expm(rate_matrix * t1)
            P2 = expm(rate_matrix * t2)
            
            # Combine: L_parent[state] = (Σ P1[state,j] * L_child1[j]) * (Σ P2[state,k] * L_child2[k])
            term1 = P1.T @ conditionals[child1_id]
            term2 = P2.T @ conditionals[child2_id]
            
            conditionals[node_id] = term1 * term2
    
    postorder(tree.root)
    
    # Compute likelihood at root with uniform prior
    root_prior = np.ones(n_states) / n_states
    likelihood = np.sum(root_prior * conditionals[tree.root])
    
    return np.log(likelihood + 1e-300)


def fit(
    cn_matrix: Union[np.ndarray, str, Path],
    family_names: Optional[List[str]] = None,
    taxon_names: Optional[List[str]] = None,
    tree: Optional[Union[Tree, str, Path]] = None,
    ploidy: int = 2,
    baseline_type: str = 'hierarchical',
    baseline_params: Optional[Dict[str, float]] = None,
    constraint_type: Optional[str] = None,
    constraint_params: Optional[Dict[str, Any]] = None,
    theta: Optional[float] = None,
    obs_model_type: str = 'deterministic',
    verbose: bool = False,
) -> CopyNumberResult:
    """
    Fit copy number dynamics model.
    
    This is the main entry point for copy number analysis.
    
    Args:
        cn_matrix: Copy number matrix (n_families, n_taxa) or path to file
            Values should be binned states (0-3) or raw counts (will be binned)
        family_names: List of gene family names
        taxon_names: List of taxon names
        tree: Phylogenetic tree or path to tree file
        ploidy: Organism ploidy (for binning raw counts)
        baseline_type: 'hierarchical' (default) or 'global'
        baseline_params: Optional baseline parameters
        constraint_type: Optional constraint type
            - 'dosage_stability' (recommended)
            - 'amplification_bias'
            - 'host_conditioned'
        constraint_params: Optional constraint parameters
        theta: Constraint parameter (if None, will be estimated)
        obs_model_type: 'deterministic' (default) or 'uncertain'
        verbose: Print progress
    
    Returns:
        CopyNumberResult with fitted model
    
    Example:
        >>> # Basic usage (null model)
        >>> result = fit(
        ...     cn_matrix=cn_data,
        ...     family_names=families,
        ...     taxon_names=taxa,
        ...     tree=tree
        ... )
        
        >>> # With dosage stability constraint
        >>> result = fit(
        ...     cn_matrix=cn_data,
        ...     family_names=families,
        ...     taxon_names=taxa,
        ...     tree=tree,
        ...     constraint_type='dosage_stability',
        ...     theta=-0.5  # buffered
        ... )
    """
    if verbose:
        print("=" * 70)
        print("COPY NUMBER DYNAMICS ANALYSIS")
        print("=" * 70)
    
    # Load/validate data
    if isinstance(cn_matrix, (str, Path)):
        # TODO: Implement file loading
        raise NotImplementedError("File loading not yet implemented")
    
    # Check if values need binning
    if cn_matrix.max() > 3:
        if verbose:
            print("\nBinning raw copy numbers...")
        cn_matrix = CopyNumberState.bin_matrix(cn_matrix, ploidy=ploidy)
    
    # Create data object
    if family_names is None:
        family_names = [f"family_{i}" for i in range(cn_matrix.shape[0])]
    if taxon_names is None:
        taxon_names = [f"taxon_{i}" for i in range(cn_matrix.shape[1])]
    
    data = CopyNumberData(
        cn_matrix=cn_matrix,
        family_names=family_names,
        taxon_names=taxon_names,
        ploidy=ploidy,
    )
    
    if verbose:
        data.print_summary()
    
    # Load/validate tree
    if tree is None:
        raise ValueError("tree is required")
    
    if isinstance(tree, (str, Path)):
        # TODO: Implement tree loading
        raise NotImplementedError("Tree loading not yet implemented")
    
    # Create baseline model
    if baseline_params is None:
        baseline_params = {}
    
    baseline = create_baseline(baseline_type, **baseline_params)
    
    if verbose:
        print(f"\nBaseline model: {baseline_type}")
    
    # Sample family rates if hierarchical
    if baseline_type == 'hierarchical':
        rng = np.random.default_rng(42)
        baseline.sample_family_rates(data.n_families, rng)
        if verbose:
            print(f"  Sampled {data.n_families} family-specific rate sets")
    
    # Create constraint if specified
    constraint = None
    if constraint_type is not None:
        if constraint_params is None:
            constraint_params = {}
        constraint = create_constraint(constraint_type, **constraint_params)
        
        if verbose:
            print(f"\nConstraint: {constraint_type}")
            if theta is not None:
                print(f"  θ = {theta}")
    
    # Create observation model
    obs_model = create_observation_model(obs_model_type)
    
    if verbose:
        print(f"\nObservation model: {obs_model_type}")
        print("\nComputing likelihoods...")
    
    # Compute per-family likelihoods
    per_family_lls = np.zeros(data.n_families)
    
    for fam_idx in range(data.n_families):
        # Get baseline rate matrix for this family
        Q_baseline = baseline.build_rate_matrix(family_idx=fam_idx)
        
        # Apply constraint if specified
        if constraint is not None and theta is not None:
            Q = apply_constraint(Q_baseline, constraint, theta, family_idx=fam_idx)
        else:
            Q = Q_baseline
        
        # Validate rate matrix
        allowed = get_sparse_transition_graph()
        validate_transition_matrix(Q, allowed)
        
        # Get family data
        family_data = data.get_family_data(fam_idx)
        
        # Compute likelihood
        ll = compute_family_likelihood(family_data, tree, Q, obs_model)
        per_family_lls[fam_idx] = ll
        
        if verbose and (fam_idx + 1) % 100 == 0:
            print(f"  Processed {fam_idx + 1}/{data.n_families} families...")
    
    # Total log-likelihood
    total_ll = per_family_lls.sum()
    
    if verbose:
        print(f"\nTotal log-likelihood: {total_ll:.2f}")
    
    # Count parameters
    if baseline_type == 'global':
        n_baseline_params = 4  # gain, loss, amplify, contract
    else:  # hierarchical
        n_baseline_params = 5  # 4 global rates + sigma
    
    n_constraint_params = 1 if constraint is not None else 0
    n_params = n_baseline_params + n_constraint_params
    
    # Get baseline parameters for reporting
    if baseline_type == 'global':
        baseline_params_dict = {
            'gain_rate': baseline.gain_rate,
            'loss_rate': baseline.loss_rate,
            'amplify_rate': baseline.amplify_rate,
            'contract_rate': baseline.contract_rate,
        }
    else:  # hierarchical
        baseline_params_dict = {
            'global_gain_rate': baseline.global_gain_rate,
            'global_loss_rate': baseline.global_loss_rate,
            'global_amplify_rate': baseline.global_amplify_rate,
            'global_contract_rate': baseline.global_contract_rate,
            'sigma': baseline.sigma,
        }
    
    # Create result
    result = CopyNumberResult(
        data=data,
        log_likelihood=total_ll,
        baseline_type=baseline_type,
        constraint_type=constraint_type,
        theta=theta,
        baseline_params=baseline_params_dict,
        n_params=n_params,
        tree=tree,
        per_family_likelihoods=per_family_lls,
    )
    
    if verbose:
        print("\n" + "=" * 70)
        result.print_summary()
    
    return result


def fit_null_model(
    cn_matrix: Union[np.ndarray, str, Path],
    family_names: Optional[List[str]] = None,
    taxon_names: Optional[List[str]] = None,
    tree: Optional[Union[Tree, str, Path]] = None,
    ploidy: int = 2,
    baseline_type: str = 'hierarchical',
    baseline_params: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> CopyNumberResult:
    """
    Fit null model (no constraint).
    
    Convenience function for fitting baseline-only model.
    
    Args:
        Same as fit(), but no constraint parameters
    
    Returns:
        CopyNumberResult for null model
    """
    return fit(
        cn_matrix=cn_matrix,
        family_names=family_names,
        taxon_names=taxon_names,
        tree=tree,
        ploidy=ploidy,
        baseline_type=baseline_type,
        baseline_params=baseline_params,
        constraint_type=None,
        verbose=verbose,
    )


def likelihood_ratio_test(
    alternative: CopyNumberResult,
    null: CopyNumberResult,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Perform likelihood ratio test.
    
    Args:
        alternative: Result from alternative model (with constraint)
        null: Result from null model (no constraint)
        verbose: Print results
    
    Returns:
        Dictionary with test statistics
    """
    comparison = alternative.compare_to(null)
    
    if verbose:
        print("=" * 70)
        print("LIKELIHOOD RATIO TEST")
        print("=" * 70)
        
        print(f"\nNull model:")
        print(f"  Baseline: {null.baseline_type}")
        print(f"  Log-likelihood: {null.log_likelihood:.2f}")
        print(f"  Parameters: {null.n_params}")
        
        print(f"\nAlternative model:")
        print(f"  Baseline: {alternative.baseline_type}")
        print(f"  Constraint: {alternative.constraint_type}")
        if alternative.theta is not None:
            print(f"  θ = {alternative.theta:.4f}")
        print(f"  Log-likelihood: {alternative.log_likelihood:.2f}")
        print(f"  Parameters: {alternative.n_params}")
        
        print(f"\nTest results:")
        print(f"  Δ log-likelihood: {comparison['delta_log_likelihood']:.2f}")
        print(f"  LRT statistic: {comparison['lrt_statistic']:.2f}")
        print(f"  df: {comparison['delta_params']}")
        print(f"  p-value: {comparison['p_value']:.4e}")
        
        if comparison['p_value'] < 0.001:
            print("\n  → HIGHLY SIGNIFICANT constraint detected")
        elif comparison['p_value'] < 0.05:
            print("\n  → Significant constraint detected")
        else:
            print("\n  → No significant constraint detected")
        
        print(f"\nModel selection:")
        print(f"  Δ AIC: {comparison['delta_aic']:.2f}")
        print(f"  Δ BIC: {comparison['delta_bic']:.2f}")
        
        if comparison['delta_aic'] < -2:
            print("  → Alternative model preferred (AIC)")
        elif comparison['delta_aic'] > 2:
            print("  → Null model preferred (AIC)")
        else:
            print("  → Models similar (AIC)")
        
        print("=" * 70)
    
    return comparison
