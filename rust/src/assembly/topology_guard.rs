//! Topology guard for detecting θ changes that invalidate cached graph topology.
//!
//! Some constraint features can change graph topology when θ changes significantly
//! (e.g., pruning rules, rate floors). This module detects such changes and
//! triggers hard resimulation.

use std::collections::{HashMap, HashSet};

/// Guard for topology-changing θ features.
///
/// Tracks features that can affect edge pruning/existence and enforces
/// resimulation when they change too much.
#[derive(Clone, Debug)]
pub struct TopologyGuard {
    /// Features that affect edge pruning.
    topology_sensitive_features: HashSet<String>,
    /// Threshold for topology change detection.
    /// If |θ_k - θ_k,ref| > threshold for a sensitive feature, trigger resim.
    topology_change_threshold: f64,
    /// Soft floor rate to prevent edges from vanishing.
    /// If set, rates are clamped to max(rate, soft_floor).
    pub soft_floor_rate: Option<f64>,
}

impl Default for TopologyGuard {
    fn default() -> Self {
        Self {
            topology_sensitive_features: HashSet::new(),
            topology_change_threshold: 2.0, // exp(±2) ≈ 7x rate change
            soft_floor_rate: Some(1e-6),
        }
    }
}

impl TopologyGuard {
    /// Create a new topology guard with specified sensitive features.
    pub fn new(
        sensitive_features: &[&str],
        threshold: f64,
        soft_floor: Option<f64>,
    ) -> Self {
        Self {
            topology_sensitive_features: sensitive_features.iter().map(|s| s.to_string()).collect(),
            topology_change_threshold: threshold,
            soft_floor_rate: soft_floor,
        }
    }

    /// Add a topology-sensitive feature.
    pub fn add_sensitive_feature(&mut self, feature: &str) {
        self.topology_sensitive_features.insert(feature.to_string());
    }

    /// Check if θ change would invalidate cached topology.
    ///
    /// Returns None if topology is preserved, or Some(affected_features) if
    /// any sensitive features changed too much.
    pub fn check(
        &self,
        theta: &HashMap<String, f64>,
        theta_ref: &HashMap<String, f64>,
    ) -> Option<Vec<String>> {
        let mut affected = Vec::new();

        for feature in &self.topology_sensitive_features {
            let theta_k = theta.get(feature).unwrap_or(&0.0);
            let theta_k_ref = theta_ref.get(feature).unwrap_or(&0.0);
            let delta = (theta_k - theta_k_ref).abs();

            if delta > self.topology_change_threshold {
                affected.push(feature.clone());
            }
        }

        if affected.is_empty() {
            None
        } else {
            Some(affected)
        }
    }

    /// Check if a specific feature is topology-sensitive.
    pub fn is_sensitive(&self, feature: &str) -> bool {
        self.topology_sensitive_features.contains(feature)
    }

    /// Get all topology-sensitive features.
    pub fn sensitive_features(&self) -> &HashSet<String> {
        &self.topology_sensitive_features
    }

    /// Apply soft floor to a rate if configured.
    pub fn apply_soft_floor(&self, rate: f64) -> f64 {
        match self.soft_floor_rate {
            Some(floor) => rate.max(floor),
            None => rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_sensitive_features() {
        let guard = TopologyGuard::default();
        let mut theta = HashMap::new();
        theta.insert("reuse_count".to_string(), 5.0);

        let result = guard.check(&theta, &HashMap::new());
        assert!(result.is_none());
    }

    #[test]
    fn test_sensitive_feature_within_threshold() {
        let guard = TopologyGuard::new(&["depth_penalty"], 2.0, None);
        
        let mut theta = HashMap::new();
        theta.insert("depth_penalty".to_string(), 1.5);

        let result = guard.check(&theta, &HashMap::new());
        assert!(result.is_none()); // 1.5 < 2.0
    }

    #[test]
    fn test_sensitive_feature_exceeds_threshold() {
        let guard = TopologyGuard::new(&["depth_penalty"], 2.0, None);
        
        let mut theta = HashMap::new();
        theta.insert("depth_penalty".to_string(), 3.0);

        let result = guard.check(&theta, &HashMap::new());
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec!["depth_penalty".to_string()]);
    }

    #[test]
    fn test_multiple_sensitive_features() {
        let guard = TopologyGuard::new(&["depth_penalty", "size_limit"], 2.0, None);
        
        let mut theta = HashMap::new();
        theta.insert("depth_penalty".to_string(), 3.0);
        theta.insert("size_limit".to_string(), 2.5);
        theta.insert("reuse_count".to_string(), 10.0); // Not sensitive

        let result = guard.check(&theta, &HashMap::new());
        assert!(result.is_some());
        let affected = result.unwrap();
        assert!(affected.contains(&"depth_penalty".to_string()));
        assert!(affected.contains(&"size_limit".to_string()));
        assert_eq!(affected.len(), 2);
    }

    #[test]
    fn test_soft_floor() {
        let guard = TopologyGuard::new(&[], 2.0, Some(1e-6));
        
        assert_eq!(guard.apply_soft_floor(0.5), 0.5);
        assert_eq!(guard.apply_soft_floor(1e-10), 1e-6);
        assert_eq!(guard.apply_soft_floor(0.0), 1e-6);
    }

    #[test]
    fn test_no_soft_floor() {
        let guard = TopologyGuard::new(&[], 2.0, None);
        
        assert_eq!(guard.apply_soft_floor(1e-10), 1e-10);
        assert_eq!(guard.apply_soft_floor(0.0), 0.0);
    }
}
