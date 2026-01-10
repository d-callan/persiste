//! Cache manager for importance-sampling with L∞ trust region.
//!
//! Manages cached path statistics and determines when resimulation is needed.

use std::collections::HashMap;

use super::path_stats::PathStats;
use super::state::AssemblyStateId;

/// Configuration for cache management.
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// Trust region radius (L∞ norm in θ-space). Default: 1.0
    pub trust_radius: f64,
    /// Minimum effective sample size ratio. Default: 0.3
    pub ess_threshold: f64,
    /// Maximum weight variance before early warning. Default: 100.0
    pub max_weight_variance: f64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            trust_radius: 1.0,
            ess_threshold: 0.3,
            max_weight_variance: 100.0,
        }
    }
}

/// Reason why cache was invalidated.
#[derive(Clone, Debug, PartialEq)]
pub enum InvalidationReason {
    /// ESS dropped below threshold.
    EssBelowThreshold { ess: f64, threshold: f64 },
    /// θ proposal exited trust region.
    OutsideTrustRegion { distance: f64, radius: f64 },
    /// Weight variance exploded (early warning).
    WeightVarianceExplosion { variance: f64, max_variance: f64 },
    /// Topology-changing θ feature detected.
    TopologyChange { affected_features: Vec<String> },
}

impl std::fmt::Display for InvalidationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InvalidationReason::EssBelowThreshold { ess, threshold } => {
                write!(f, "ESS {:.2} below threshold {:.2}", ess, threshold)
            }
            InvalidationReason::OutsideTrustRegion { distance, radius } => {
                write!(f, "L∞ distance {:.3} exceeds trust radius {:.3}", distance, radius)
            }
            InvalidationReason::WeightVarianceExplosion { variance, max_variance } => {
                write!(f, "Weight variance {:.2} exceeds max {:.2}", variance, max_variance)
            }
            InvalidationReason::TopologyChange { affected_features } => {
                write!(f, "Topology change in features: {:?}", affected_features)
            }
        }
    }
}

/// Result of cache evaluation.
#[derive(Clone, Debug)]
pub enum CacheStatus {
    /// Cache is valid; contains reweighted state distribution and ESS.
    Valid {
        latent_states: HashMap<AssemblyStateId, f64>,
        ess: f64,
    },
    /// Cache is invalid; must resimulate.
    Invalid { reason: InvalidationReason },
}

/// Manager for importance-sampling cache.
///
/// Implements the three orthogonal safety valves:
/// 1. ESS < α·N (default α = 0.3)
/// 2. θ proposal exits L∞ trust region (default radius = 1.0)
/// 3. Weight variance explosion (cheap early warning)
#[derive(Clone, Debug)]
pub struct CacheManager {
    /// Cached path statistics from reference simulation.
    paths: Vec<PathStats>,
    /// Reference θ at which paths were simulated.
    theta_ref: HashMap<String, f64>,
    /// Configuration.
    config: CacheConfig,
}

impl CacheManager {
    /// Create a new cache manager with cached paths.
    pub fn new(
        paths: Vec<PathStats>,
        theta_ref: HashMap<String, f64>,
        config: CacheConfig,
    ) -> Self {
        Self {
            paths,
            theta_ref,
            config,
        }
    }

    /// Number of cached paths.
    pub fn n_paths(&self) -> usize {
        self.paths.len()
    }

    /// Get reference θ.
    pub fn theta_ref(&self) -> &HashMap<String, f64> {
        &self.theta_ref
    }

    /// Get cached paths.
    pub fn paths(&self) -> &[PathStats] {
        &self.paths
    }

    /// Evaluate cache validity and return reweighted states or invalidation reason.
    ///
    /// This is the main entry point for cache management.
    pub fn evaluate(&self, theta: &HashMap<String, f64>) -> CacheStatus {
        if self.paths.is_empty() {
            return CacheStatus::Invalid {
                reason: InvalidationReason::EssBelowThreshold {
                    ess: 0.0,
                    threshold: 0.0,
                },
            };
        }

        // 1. Check trust region (L∞ distance)
        let distance = self.linf_distance(theta);
        if distance > self.config.trust_radius {
            return CacheStatus::Invalid {
                reason: InvalidationReason::OutsideTrustRegion {
                    distance,
                    radius: self.config.trust_radius,
                },
            };
        }

        // 2. Compute log weights
        let log_weights: Vec<f64> = self
            .paths
            .iter()
            .map(|p| self.log_weight(p, theta))
            .collect();

        // 3. Check weight variance (early warning)
        let weight_var = self.weight_variance(&log_weights);
        if weight_var > self.config.max_weight_variance {
            return CacheStatus::Invalid {
                reason: InvalidationReason::WeightVarianceExplosion {
                    variance: weight_var,
                    max_variance: self.config.max_weight_variance,
                },
            };
        }

        // 4. Normalize weights and compute ESS
        let (weights, ess) = self.normalize_and_ess(&log_weights);
        let ess_threshold = self.config.ess_threshold * (self.paths.len() as f64);
        if ess < ess_threshold {
            return CacheStatus::Invalid {
                reason: InvalidationReason::EssBelowThreshold {
                    ess,
                    threshold: ess_threshold,
                },
            };
        }

        // 5. Aggregate to state distribution
        let latent_states = self.aggregate_to_states(&weights);

        CacheStatus::Valid { latent_states, ess }
    }

    /// Compute L∞ distance from reference θ.
    fn linf_distance(&self, theta: &HashMap<String, f64>) -> f64 {
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

    /// Compute log importance weight for a path.
    ///
    /// log w(τ) = Σ_k N_k(τ) · (θ_k - θ_k,ref)
    fn log_weight(&self, path: &PathStats, theta: &HashMap<String, f64>) -> f64 {
        let mut log_w = 0.0;
        for (feature, &count) in &path.feature_counts {
            let theta_k = theta.get(feature).unwrap_or(&0.0);
            let theta_k_ref = self.theta_ref.get(feature).unwrap_or(&0.0);
            log_w += (count as f64) * (theta_k - theta_k_ref);
        }
        log_w
    }

    /// Compute variance of weights (for early warning).
    fn weight_variance(&self, log_weights: &[f64]) -> f64 {
        if log_weights.len() < 2 {
            return 0.0;
        }

        // Convert to weights
        let max_lw = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_lw.is_infinite() {
            return 0.0;
        }

        let weights: Vec<f64> = log_weights.iter().map(|&lw| (lw - max_lw).exp()).collect();
        let mean: f64 = weights.iter().sum::<f64>() / weights.len() as f64;
        let variance: f64 = weights.iter().map(|&w| (w - mean).powi(2)).sum::<f64>()
            / (weights.len() - 1) as f64;

        variance
    }

    /// Normalize log weights and compute ESS.
    fn normalize_and_ess(&self, log_weights: &[f64]) -> (Vec<f64>, f64) {
        if log_weights.is_empty() {
            return (vec![], 0.0);
        }

        // Log-sum-exp for numerical stability
        let max_log_w = log_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_log_w.is_infinite() {
            let n = log_weights.len();
            return (vec![1.0 / n as f64; n], n as f64);
        }

        let sum_exp: f64 = log_weights.iter().map(|&lw| (lw - max_log_w).exp()).sum();
        let log_sum = max_log_w + sum_exp.ln();

        let weights: Vec<f64> = log_weights
            .iter()
            .map(|&lw| (lw - log_sum).exp())
            .collect();

        // ESS = 1 / Σ w_i²
        let sum_sq: f64 = weights.iter().map(|&w| w * w).sum();
        let ess = if sum_sq > 0.0 { 1.0 / sum_sq } else { 0.0 };

        (weights, ess)
    }

    /// Aggregate weighted paths to state distribution.
    fn aggregate_to_states(&self, weights: &[f64]) -> HashMap<AssemblyStateId, f64> {
        let mut state_probs: HashMap<AssemblyStateId, f64> = HashMap::new();
        for (path, &weight) in self.paths.iter().zip(weights.iter()) {
            *state_probs.entry(path.final_state_id).or_insert(0.0) += weight;
        }
        state_probs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_paths(n: usize, feature_count: u32) -> Vec<PathStats> {
        (0..n)
            .map(|i| {
                let mut p = PathStats::new(i as u64, 10.0);
                p.feature_counts.insert("reuse_count".to_string(), feature_count);
                p
            })
            .collect()
    }

    #[test]
    fn test_valid_at_reference() {
        let paths = make_test_paths(100, 1);
        let theta_ref = HashMap::new();
        let config = CacheConfig::default();
        let cache = CacheManager::new(paths, theta_ref.clone(), config);

        let status = cache.evaluate(&theta_ref);
        match status {
            CacheStatus::Valid { ess, .. } => {
                // At reference, all weights equal, ESS = N
                assert!((ess - 100.0).abs() < 1e-6);
            }
            CacheStatus::Invalid { reason } => {
                panic!("Expected valid, got invalid: {}", reason);
            }
        }
    }

    #[test]
    fn test_invalid_outside_trust_region() {
        let paths = make_test_paths(100, 1);
        let theta_ref = HashMap::new();
        let config = CacheConfig {
            trust_radius: 1.0,
            ..Default::default()
        };
        let cache = CacheManager::new(paths, theta_ref, config);

        // θ with large deviation
        let mut theta = HashMap::new();
        theta.insert("reuse_count".to_string(), 2.0); // distance = 2.0 > 1.0

        let status = cache.evaluate(&theta);
        match status {
            CacheStatus::Invalid { reason } => {
                assert!(matches!(reason, InvalidationReason::OutsideTrustRegion { .. }));
            }
            CacheStatus::Valid { .. } => {
                panic!("Expected invalid due to trust region");
            }
        }
    }

    #[test]
    fn test_ess_decreases_with_theta_shift() {
        let paths = make_test_paths(100, 1);
        let theta_ref = HashMap::new();
        let config = CacheConfig {
            trust_radius: 0.5, // Smaller trust region
            ess_threshold: 0.1, // Allow low ESS for this test
            ..Default::default()
        };
        let cache = CacheManager::new(paths, theta_ref, config);

        // Small shift within trust region
        let mut theta = HashMap::new();
        theta.insert("reuse_count".to_string(), 0.3);

        let status = cache.evaluate(&theta);
        match status {
            CacheStatus::Valid { ess, .. } => {
                // ESS should be less than N due to weight variance
                assert!(ess < 100.0);
                assert!(ess > 0.0);
            }
            CacheStatus::Invalid { reason } => {
                panic!("Expected valid, got invalid: {}", reason);
            }
        }
    }

    #[test]
    fn test_linf_distance() {
        let paths = vec![];
        let mut theta_ref = HashMap::new();
        theta_ref.insert("a".to_string(), 1.0);
        theta_ref.insert("b".to_string(), 2.0);

        let cache = CacheManager::new(paths, theta_ref, CacheConfig::default());

        let mut theta = HashMap::new();
        theta.insert("a".to_string(), 1.5);
        theta.insert("b".to_string(), 4.0);

        // L∞ distance = max(|1.5-1|, |4-2|) = 2.0
        let dist = cache.linf_distance(&theta);
        assert!((dist - 2.0).abs() < 1e-10);
    }
}
