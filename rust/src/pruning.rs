/// Felsenstein pruning algorithm for binary trait evolution.
///
/// Computes the log-likelihood of observing tip states given a tree
/// and binary evolution model (gain/loss rates).

use crate::tree::Tree;

/// Compute transition probability matrix for binary evolution.
///
/// Uses analytical solution for 2×2 continuous-time Markov chain:
/// Q = [[-λ, λ], [μ, -μ]]
/// P(t) = exp(Qt)
///
/// # Arguments
/// * `gain_rate` - Rate of 0→1 transitions (λ)
/// * `loss_rate` - Rate of 1→0 transitions (μ)
/// * `t` - Branch length (time)
///
/// # Returns
/// 2×2 transition probability matrix [[P00, P01], [P10, P11]]
fn transition_matrix(gain_rate: f64, loss_rate: f64, t: f64) -> [[f64; 2]; 2] {
    let total = gain_rate + loss_rate;
    
    if total < 1e-10 {
        // No evolution - identity matrix
        return [[1.0, 0.0], [0.0, 1.0]];
    }
    
    let exp_term = (-total * t).exp();
    
    let p00 = (loss_rate + gain_rate * exp_term) / total;
    let p01 = (gain_rate - gain_rate * exp_term) / total;
    let p10 = (loss_rate - loss_rate * exp_term) / total;
    let p11 = (gain_rate + loss_rate * exp_term) / total;
    
    [[p00, p01], [p10, p11]]
}

/// Compute equilibrium frequencies for binary evolution.
///
/// At equilibrium: π₀ · λ = π₁ · μ (detailed balance)
/// And: π₀ + π₁ = 1
///
/// # Returns
/// [π₀, π₁] equilibrium frequencies
fn equilibrium_frequencies(gain_rate: f64, loss_rate: f64) -> [f64; 2] {
    let total = gain_rate + loss_rate;
    
    if total < 1e-10 {
        return [0.5, 0.5];
    }
    
    let pi_0 = loss_rate / total;
    let pi_1 = gain_rate / total;
    
    [pi_0, pi_1]
}

/// Felsenstein pruning algorithm for a single gene family.
///
/// Computes log P(tip_data | tree, gain_rate, loss_rate) using
/// dynamic programming on the tree.
///
/// # Algorithm
/// 1. Initialize tip conditionals: L[tip, state] = 1 if state matches data, else 0
/// 2. Post-order traversal: For each internal node, compute:
///    L[node, s] = ∏_{children} ∑_{s'} P[s→s'](t) · L[child, s']
/// 3. At root: likelihood = ∑_s π[s] · L[root, s]
///
/// # Arguments
/// * `tree` - Phylogenetic tree structure
/// * `tip_data` - Binary states at tips (0 or 1)
/// * `gain_rate` - Rate of gaining trait (0→1)
/// * `loss_rate` - Rate of losing trait (1→0)
///
/// # Returns
/// Log-likelihood of observing the tip data
pub fn felsenstein_pruning(
    tree: &Tree,
    tip_data: &[i8],
    gain_rate: f64,
    loss_rate: f64,
) -> f64 {
    // Initialize conditional likelihoods
    // conditionals[node][state] = P(data below node | node in state)
    let mut conditionals = vec![[1.0; 2]; tree.n_nodes];
    
    // Set tip conditionals based on observed data
    for (tip_idx_pos, &tip_idx) in tree.tip_indices.iter().enumerate() {
        let observed_state = tip_data[tip_idx_pos] as usize;
        
        // Deterministic observation: P(obs | state) = 1 if state == obs, else 0
        conditionals[tip_idx] = if observed_state == 1 {
            [0.0, 1.0]  // Observed present
        } else {
            [1.0, 0.0]  // Observed absent
        };
    }
    
    // Post-order traversal (tips to root)
    let post_order = tree.post_order();
    
    for &node_idx in &post_order {
        // Skip tips (already initialized)
        if tree.children[node_idx].is_empty() {
            continue;
        }
        
        // Internal node: compute from children
        for &child_idx in &tree.children[node_idx] {
            let branch_length = tree.branch_lengths[child_idx];
            let P = transition_matrix(gain_rate, loss_rate, branch_length);
            
            // For each parent state, integrate over child states
            let mut parent_contrib = [0.0; 2];
            for parent_state in 0..2 {
                for child_state in 0..2 {
                    parent_contrib[parent_state] += 
                        P[parent_state][child_state] * conditionals[child_idx][child_state];
                }
            }
            
            // Multiply contributions from all children
            for state in 0..2 {
                conditionals[node_idx][state] *= parent_contrib[state];
            }
        }
    }
    
    // Compute likelihood at root
    let eq_freqs = equilibrium_frequencies(gain_rate, loss_rate);
    let root_conditional = conditionals[tree.root_index];
    
    let likelihood = eq_freqs[0] * root_conditional[0] + eq_freqs[1] * root_conditional[1];
    
    // Return log-likelihood (add small constant to avoid log(0))
    (likelihood + 1e-300).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_transition_matrix() {
        let P = transition_matrix(1.0, 1.0, 1.0);
        
        // Rows should sum to 1
        assert!((P[0][0] + P[0][1] - 1.0).abs() < 1e-10);
        assert!((P[1][0] + P[1][1] - 1.0).abs() < 1e-10);
        
        // All probabilities should be in [0, 1]
        for row in &P {
            for &p in row {
                assert!(p >= 0.0 && p <= 1.0);
            }
        }
    }
    
    #[test]
    fn test_equilibrium_frequencies() {
        let eq = equilibrium_frequencies(1.5, 2.0);
        
        // Should sum to 1
        assert!((eq[0] + eq[1] - 1.0).abs() < 1e-10);
        
        // Both should be positive
        assert!(eq[0] > 0.0 && eq[1] > 0.0);
        
        // With higher loss rate, π₀ should be larger
        assert!(eq[0] > eq[1]);
    }
    
    #[test]
    fn test_pruning_simple() {
        use crate::tree::Tree;
        
        let taxon_names = vec!["A".to_string(), "B".to_string()];
        let tree = Tree::from_newick("(A:1,B:1);", &taxon_names).unwrap();
        
        // Both tips present
        let tip_data = vec![1, 1];
        let ll = felsenstein_pruning(&tree, &tip_data, 1.0, 1.0);
        
        // Should be finite
        assert!(ll.is_finite());
        assert!(ll < 0.0);  // Log-likelihood should be negative
    }
}
