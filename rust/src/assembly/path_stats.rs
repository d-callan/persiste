//! Path statistics for importance sampling.
//!
//! Sufficient statistics from Gillespie trajectories that enable
//! importance sampling without re-simulation.

use std::collections::HashMap;

use super::state::AssemblyStateId;

/// Sufficient statistics from a single Gillespie trajectory.
///
/// These statistics enable importance sampling: given paths sampled at θ_ref,
/// we can reweight to approximate the distribution at a different θ.
///
/// Key formula:
///     log w(τ) = Σ_k N_k(τ) · (θ_k - θ_k,ref)
///
/// Where N_k(τ) is the count of feature k observed in trajectory τ.
#[derive(Clone, Debug)]
pub struct PathStats {
    /// Counts of each transition feature observed in trajectory.
    /// Key format: feature_name (e.g., "reuse_count", "depth_change", "transition_join")
    pub feature_counts: HashMap<String, u32>,
    /// Final state ID (for observation model).
    pub final_state_id: AssemblyStateId,
    /// Total path log-probability under reference θ (for diagnostics).
    pub log_prob_ref: f64,
    /// Trajectory duration.
    pub duration: f64,
    /// Number of transitions in trajectory.
    pub n_transitions: u32,
}

impl PathStats {
    /// Create new path stats.
    pub fn new(final_state_id: AssemblyStateId, duration: f64) -> Self {
        Self {
            feature_counts: HashMap::new(),
            final_state_id,
            log_prob_ref: 0.0,
            duration,
            n_transitions: 0,
        }
    }

    /// Record a transition's feature contributions.
    pub fn record_transition(&mut self, features: &HashMap<String, f64>) {
        for (feature_name, &value) in features {
            // For counting, we round to nearest integer (most features are 0/1)
            let count = value.round() as i32;
            if count > 0 {
                *self.feature_counts.entry(feature_name.clone()).or_insert(0) += count as u32;
            }
        }
        self.n_transitions += 1;
    }

    /// Compute importance weight for new θ relative to θ_ref.
    ///
    /// log w(τ) = Σ_k N_k(τ) · (θ_k - θ_k,ref)
    ///
    /// This is the key formula for importance sampling.
    pub fn log_weight(&self, theta: &HashMap<String, f64>, theta_ref: &HashMap<String, f64>) -> f64 {
        let mut log_w = 0.0;
        for (feature, &count) in &self.feature_counts {
            let theta_k = theta.get(feature).unwrap_or(&0.0);
            let theta_k_ref = theta_ref.get(feature).unwrap_or(&0.0);
            log_w += (count as f64) * (theta_k - theta_k_ref);
        }
        log_w
    }

    /// Compute unnormalized weight.
    pub fn weight(&self, theta: &HashMap<String, f64>, theta_ref: &HashMap<String, f64>) -> f64 {
        self.log_weight(theta, theta_ref).exp()
    }

    /// Get total count for a specific feature.
    pub fn get_feature_count(&self, feature: &str) -> u32 {
        *self.feature_counts.get(feature).unwrap_or(&0)
    }
}

/// Collection of path statistics with utilities for importance sampling.
#[derive(Clone, Debug)]
pub struct PathStatsCollection {
    /// Individual path statistics.
    pub paths: Vec<PathStats>,
    /// Reference θ at which paths were simulated.
    pub theta_ref: HashMap<String, f64>,
}

impl PathStatsCollection {
    /// Create new collection.
    pub fn new(paths: Vec<PathStats>, theta_ref: HashMap<String, f64>) -> Self {
        Self { paths, theta_ref }
    }

    /// Number of paths.
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Compute normalized weights for a given θ.
    ///
    /// Returns (weights, effective_sample_size).
    pub fn normalized_weights(&self, theta: &HashMap<String, f64>) -> (Vec<f64>, f64) {
        if self.paths.is_empty() {
            return (vec![], 0.0);
        }

        // Compute log weights
        let log_weights: Vec<f64> = self
            .paths
            .iter()
            .map(|p| p.log_weight(theta, &self.theta_ref))
            .collect();

        // Log-sum-exp for numerical stability
        let max_log_w = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_log_w.is_infinite() {
            return (vec![1.0 / self.paths.len() as f64; self.paths.len()], self.paths.len() as f64);
        }

        let sum_exp: f64 = log_weights.iter().map(|&lw| (lw - max_log_w).exp()).sum();
        let log_sum = max_log_w + sum_exp.ln();

        // Normalized weights
        let weights: Vec<f64> = log_weights
            .iter()
            .map(|&lw| (lw - log_sum).exp())
            .collect();

        // Effective sample size: ESS = 1 / Σ w_i²
        let sum_sq: f64 = weights.iter().map(|&w| w * w).sum();
        let ess = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };

        (weights, ess)
    }

    /// Compute effective sample size for a given θ.
    pub fn ess(&self, theta: &HashMap<String, f64>) -> f64 {
        self.normalized_weights(theta).1
    }

    /// Compute weight variance (for early warning of IS breakdown).
    pub fn weight_variance(&self, theta: &HashMap<String, f64>) -> f64 {
        if self.paths.len() < 2 {
            return 0.0;
        }

        let weights: Vec<f64> = self
            .paths
            .iter()
            .map(|p| p.weight(theta, &self.theta_ref))
            .collect();

        let mean: f64 = weights.iter().sum::<f64>() / weights.len() as f64;
        let variance: f64 = weights.iter().map(|&w| (w - mean).powi(2)).sum::<f64>()
            / (weights.len() - 1) as f64;

        variance
    }

    /// Aggregate to state distribution using importance weights.
    pub fn weighted_state_distribution(
        &self,
        theta: &HashMap<String, f64>,
    ) -> HashMap<AssemblyStateId, f64> {
        let (weights, _ess) = self.normalized_weights(theta);

        let mut state_probs: HashMap<AssemblyStateId, f64> = HashMap::new();
        for (path, &weight) in self.paths.iter().zip(weights.iter()) {
            *state_probs.entry(path.final_state_id).or_insert(0.0) += weight;
        }

        state_probs
    }

    /// Check L∞ distance from reference θ.
    pub fn linf_distance(&self, theta: &HashMap<String, f64>) -> f64 {
        let mut max_dist = 0.0_f64;

        // Check all keys in theta
        for (key, &value) in theta {
            let ref_value = self.theta_ref.get(key).unwrap_or(&0.0);
            max_dist = max_dist.max((value - ref_value).abs());
        }

        // Check all keys in theta_ref
        for (key, &ref_value) in &self.theta_ref {
            let value = theta.get(key).unwrap_or(&0.0);
            max_dist = max_dist.max((value - ref_value).abs());
        }

        max_dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_weight_at_reference() {
        let mut path = PathStats::new(42, 10.0);
        path.feature_counts.insert("reuse_count".to_string(), 3);

        let mut theta = HashMap::new();
        theta.insert("reuse_count".to_string(), 1.0);

        // At reference point, log weight should be 0
        let log_w = path.log_weight(&theta, &theta);
        assert_eq!(log_w, 0.0);
    }

    #[test]
    fn test_log_weight_formula() {
        let mut path = PathStats::new(42, 10.0);
        path.feature_counts.insert("reuse_count".to_string(), 3);
        path.feature_counts.insert("depth_change".to_string(), 2);

        let mut theta = HashMap::new();
        theta.insert("reuse_count".to_string(), 2.0);
        theta.insert("depth_change".to_string(), -0.5);

        let theta_ref = HashMap::new(); // All zeros

        // log w = 3 * (2.0 - 0) + 2 * (-0.5 - 0) = 6 - 1 = 5
        let log_w = path.log_weight(&theta, &theta_ref);
        assert!((log_w - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_ess_at_reference() {
        let paths: Vec<PathStats> = (0..100)
            .map(|i| {
                let mut p = PathStats::new(i as u64, 10.0);
                p.feature_counts.insert("reuse_count".to_string(), 1);
                p
            })
            .collect();

        let theta_ref = HashMap::new();
        let collection = PathStatsCollection::new(paths, theta_ref.clone());

        // At reference, all weights should be equal, ESS = N
        let ess = collection.ess(&theta_ref);
        assert!((ess - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_linf_distance() {
        let paths = vec![];
        let mut theta_ref = HashMap::new();
        theta_ref.insert("a".to_string(), 1.0);
        theta_ref.insert("b".to_string(), 2.0);

        let collection = PathStatsCollection::new(paths, theta_ref);

        let mut theta = HashMap::new();
        theta.insert("a".to_string(), 1.5);
        theta.insert("b".to_string(), 4.0);

        // L∞ distance = max(|1.5-1|, |4-2|) = 2.0
        let dist = collection.linf_distance(&theta);
        assert!((dist - 2.0).abs() < 1e-10);
    }
}
