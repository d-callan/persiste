//! Physics-agnostic baseline for assembly transitions.
//!
//! Baseline knows nothing about chemistry, catalysis, or "life."
//! Pure combinatorics and size effects.

use super::state::AssemblyState;

/// Primitive assembly transition types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransitionType {
    /// X + Y → X∘Y
    Join,
    /// X∘Y → X + Y
    Split,
    /// X → ∅
    Decay,
    /// X∘Y → X′∘Y′ (optional, phase 2)
    Rearrange,
}

impl TransitionType {
    /// Get string representation for feature counting.
    pub fn as_feature_key(&self) -> &'static str {
        match self {
            TransitionType::Join => "join",
            TransitionType::Split => "split",
            TransitionType::Decay => "decay",
            TransitionType::Rearrange => "rearrange",
        }
    }
}

/// Physics-agnostic baseline for assembly transitions.
///
/// Factorized rate formula:
///     λ_baseline(i → j) = κ × f(size_i) × g(size_j) × h(type)
///
/// Where:
///     κ - Global rate constant
///     f(size_i) - Source size factor
///     g(size_j) - Target size factor
///     h(type) - Transition type factor
///
/// Key principle: Baseline doesn't know about chemistry.
/// No functional groups, no catalysis, no "life."
#[derive(Clone, Debug)]
pub struct AssemblyBaseline {
    /// Global rate constant
    pub kappa: f64,
    /// Size scaling for join transitions (typically negative)
    pub join_exponent: f64,
    /// Size scaling for split transitions (typically positive)
    pub split_exponent: f64,
    /// Base decay rate
    pub decay_rate: f64,
}

impl Default for AssemblyBaseline {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            join_exponent: -0.5,
            split_exponent: 0.3,
            decay_rate: 0.01,
        }
    }
}

impl AssemblyBaseline {
    /// Create a new baseline with custom parameters.
    pub fn new(kappa: f64, join_exponent: f64, split_exponent: f64, decay_rate: f64) -> Self {
        Self {
            kappa,
            join_exponent,
            split_exponent,
            decay_rate,
        }
    }

    /// Compute baseline transition rate.
    ///
    /// No chemistry. No functional groups. No catalysis.
    /// Pure size and type effects.
    pub fn get_rate(
        &self,
        source: &AssemblyState,
        target: &AssemblyState,
        transition_type: TransitionType,
    ) -> f64 {
        match transition_type {
            TransitionType::Join => self.join_rate(source, target),
            TransitionType::Split => self.split_rate(source, target),
            TransitionType::Decay => self.decay_rate(source),
            TransitionType::Rearrange => self.rearrange_rate(source, target),
        }
    }

    /// Join rate: X + Y → X∘Y
    ///
    /// Harder to join larger assemblies (negative exponent).
    fn join_rate(&self, _source: &AssemblyState, target: &AssemblyState) -> f64 {
        let depth = target.depth() as f64;
        let size_factor = if depth > 0.0 {
            depth.powf(self.join_exponent)
        } else {
            1.0
        };
        self.kappa * size_factor
    }

    /// Split rate: X∘Y → X + Y
    ///
    /// Easier to split larger assemblies (positive exponent).
    fn split_rate(&self, source: &AssemblyState, _target: &AssemblyState) -> f64 {
        let depth = source.depth() as f64;
        let size_factor = if depth > 0.0 {
            depth.powf(self.split_exponent)
        } else {
            1.0
        };
        self.kappa * size_factor
    }

    /// Decay rate: X → ∅
    ///
    /// Constant decay rate.
    fn decay_rate(&self, _source: &AssemblyState) -> f64 {
        self.decay_rate
    }

    /// Rearrange rate: X∘Y → X′∘Y′
    ///
    /// Rare under baseline (phase 2 feature).
    fn rearrange_rate(&self, _source: &AssemblyState, _target: &AssemblyState) -> f64 {
        self.kappa * 0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_baseline() {
        let baseline = AssemblyBaseline::default();
        assert_eq!(baseline.kappa, 1.0);
        assert_eq!(baseline.join_exponent, -0.5);
    }

    #[test]
    fn test_join_rate_decreases_with_depth() {
        let baseline = AssemblyBaseline::default();
        let source = AssemblyState::new(&["A"], 1, None);
        let target_shallow = AssemblyState::new(&["A", "B"], 2, None);
        let target_deep = AssemblyState::new(&["A", "B", "C"], 4, None);

        let rate_shallow = baseline.get_rate(&source, &target_shallow, TransitionType::Join);
        let rate_deep = baseline.get_rate(&source, &target_deep, TransitionType::Join);

        // With negative exponent, deeper targets have lower rates
        assert!(rate_deep < rate_shallow);
    }

    #[test]
    fn test_split_rate_increases_with_depth() {
        let baseline = AssemblyBaseline::default();
        let source_shallow = AssemblyState::new(&["A", "B"], 2, None);
        let source_deep = AssemblyState::new(&["A", "B", "C"], 4, None);
        let target = AssemblyState::new(&["A"], 1, None);

        let rate_shallow = baseline.get_rate(&source_shallow, &target, TransitionType::Split);
        let rate_deep = baseline.get_rate(&source_deep, &target, TransitionType::Split);

        // With positive exponent, deeper sources have higher split rates
        assert!(rate_deep > rate_shallow);
    }
}
