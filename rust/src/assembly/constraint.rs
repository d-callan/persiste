//! Assembly theory constraint model.
//!
//! Constraint encodes:
//! - Motif stability
//! - Reusability of subassemblies
//! - Environmental compatibility
//! - Recursive reuse (core assembly theory idea)

use std::collections::HashMap;

use super::baseline::TransitionType;
use super::state::AssemblyState;

/// Assembly constraint model.
///
/// Defines constraint contribution C(i → j; θ) that modifies baseline rates:
///     λ_eff(i → j) = λ_baseline(i → j) × exp(C(i → j; θ))
///
/// Where C(i → j; θ) = θ · f(i → j)
///
/// θ = feature weights (this is the theory/hypothesis)
/// f = feature vector (extracted from transition)
#[derive(Clone, Debug, Default)]
pub struct AssemblyConstraint {
    /// Feature weights: feature_name -> weight (log-scale)
    pub feature_weights: HashMap<String, f64>,
}

impl AssemblyConstraint {
    /// Create a new constraint with given feature weights.
    pub fn new(feature_weights: HashMap<String, f64>) -> Self {
        Self { feature_weights }
    }

    /// Create null model (no constraints).
    pub fn null_model() -> Self {
        Self::default()
    }

    /// Compute constraint contribution C(i → j; θ).
    ///
    /// C(i → j; θ) = θ · f(i → j)
    ///
    /// This is added to log-rate: log(λ_eff) = log(λ_baseline) + C
    pub fn constraint_contribution(
        &self,
        source: &AssemblyState,
        target: &AssemblyState,
        transition_type: TransitionType,
    ) -> f64 {
        let features = self.extract_features(source, target, transition_type);

        let mut contribution = 0.0;
        for (feature_name, feature_value) in &features {
            if let Some(&weight) = self.feature_weights.get(feature_name) {
                contribution += weight * feature_value;
            }
        }
        contribution
    }

    /// Compute effective rate multiplier: exp(C).
    pub fn rate_multiplier(
        &self,
        source: &AssemblyState,
        target: &AssemblyState,
        transition_type: TransitionType,
    ) -> f64 {
        self.constraint_contribution(source, target, transition_type).exp()
    }

    /// Extract features from a transition.
    ///
    /// Features are hypothesis-neutral observables.
    fn extract_features(
        &self,
        source: &AssemblyState,
        target: &AssemblyState,
        transition_type: TransitionType,
    ) -> HashMap<String, f64> {
        let mut features = HashMap::new();

        // Reuse count: is source a subassembly of target?
        let reuse_count = if source.is_subassembly_of(target) {
            1.0
        } else {
            0.0
        };
        features.insert("reuse_count".to_string(), reuse_count);

        // Depth change
        let depth_change = target.depth() as f64 - source.depth() as f64;
        features.insert("depth_change".to_string(), depth_change);

        // Size change
        let size_change = target.size() as f64 - source.size() as f64;
        features.insert("size_change".to_string(), size_change);

        // Transition type indicator
        features.insert(
            format!("transition_{}", transition_type.as_feature_key()),
            1.0,
        );

        // Motif indicators
        for motif in target.motifs() {
            if !source.motifs().contains(motif) {
                features.insert(format!("motif_gained_{}", motif), 1.0);
            }
        }
        for motif in source.motifs() {
            if !target.motifs().contains(motif) {
                features.insert(format!("motif_lost_{}", motif), 1.0);
            }
        }

        features
    }

    /// Get feature names that this constraint uses.
    pub fn feature_names(&self) -> Vec<String> {
        self.feature_weights.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_model() {
        let constraint = AssemblyConstraint::null_model();
        let source = AssemblyState::new(&["A"], 1, None);
        let target = AssemblyState::new(&["A", "B"], 2, None);

        let contribution = constraint.constraint_contribution(&source, &target, TransitionType::Join);
        assert_eq!(contribution, 0.0);
        assert_eq!(constraint.rate_multiplier(&source, &target, TransitionType::Join), 1.0);
    }

    #[test]
    fn test_reuse_bonus() {
        let mut weights = HashMap::new();
        weights.insert("reuse_count".to_string(), 2.0);
        let constraint = AssemblyConstraint::new(weights);

        let source = AssemblyState::new(&["A"], 1, None);
        let target = AssemblyState::new(&["A", "B"], 2, None);

        let contribution = constraint.constraint_contribution(&source, &target, TransitionType::Join);
        assert_eq!(contribution, 2.0); // reuse_count=1 * weight=2.0

        let multiplier = constraint.rate_multiplier(&source, &target, TransitionType::Join);
        assert!((multiplier - 2.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn test_depth_penalty() {
        let mut weights = HashMap::new();
        weights.insert("depth_change".to_string(), -0.5);
        let constraint = AssemblyConstraint::new(weights);

        let source = AssemblyState::new(&["A"], 1, None);
        let target = AssemblyState::new(&["A", "B"], 2, None);

        let contribution = constraint.constraint_contribution(&source, &target, TransitionType::Join);
        // depth_change = 2 - 1 = 1, weight = -0.5
        assert_eq!(contribution, -0.5);
    }
}
