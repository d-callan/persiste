//! Gillespie simulator for assembly CTMC with parallel trajectory support.
//!
//! Simulates stochastic dynamics:
//! - State = AssemblyState
//! - Events = allowed transitions from current state
//! - Rates = λ_eff(i→j) from baseline × constraint
//!
//! Key feature: parallel trajectory simulation using Rayon.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::baseline::{AssemblyBaseline, TransitionType};
use super::constraint::AssemblyConstraint;
use super::path_stats::PathStats;
use super::state::{AssemblyState, AssemblyStateId};

/// Neighbor in the assembly graph: (target_state, effective_rate, transition_type).
pub type Neighbor = (AssemblyState, f64, TransitionType);

/// Configuration for Gillespie simulation.
#[derive(Clone, Debug)]
pub struct SimulationConfig {
    /// Maximum simulation time.
    pub t_max: f64,
    /// Burn-in time (discard early states).
    pub burn_in: f64,
    /// Maximum depth for graph exploration.
    pub max_depth: u32,
    /// Minimum rate threshold for pruning.
    pub min_rate_threshold: f64,
    /// Primitives (initial building blocks).
    pub primitives: Vec<String>,
    /// Depth threshold for non-stationarity (Symmetry Break A).
    /// If Some(d_star), apply depth-gated reuse modifier when max_depth >= d_star.
    pub depth_gate_threshold: Option<u32>,
    /// Reuse modifier strength for depth-gated regime (θ_depth in exp(θ_depth)).
    pub depth_gate_theta: f64,
    /// Optional context-class configuration (Symmetry Break B).
    pub context_class_config: Option<ContextClassConfig>,
    /// Optional founder-bias configuration (Symmetry Break C).
    pub founder_bias_config: Option<FounderBiasConfig>,
}

/// Optional configuration for context-dependent reuse (Symmetry Break B).
#[derive(Clone, Debug)]
pub struct ContextClassConfig {
    /// Mapping from primitive name to context class label.
    pub primitive_classes: HashMap<String, String>,
    /// Log-scale modifier applied when reuse occurs within the same class.
    pub same_class_theta: f64,
    /// Log-scale modifier applied when reuse occurs across classes.
    pub cross_class_theta: f64,
}

/// Configuration for founder bias (Symmetry Break C).
#[derive(Clone, Debug)]
pub struct FounderBiasConfig {
    /// States with visit rank <= threshold get founder bonus.
    pub founder_rank_threshold: u32,
    /// Log-scale bonus applied to founders (exp applied internally).
    pub founder_bonus_theta: f64,
    /// Log-scale penalty applied to late/derived states.
    pub late_penalty_theta: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            t_max: 50.0,
            burn_in: 25.0,
            max_depth: 5,
            min_rate_threshold: 1e-6,
            primitives: vec!["A".to_string()],
            depth_gate_threshold: None,
            depth_gate_theta: 0.0,
            context_class_config: None,
            founder_bias_config: None,
        }
    }
}

/// Gillespie simulator for assembly CTMC.
pub struct GillespieSimulator {
    baseline: AssemblyBaseline,
    constraint: AssemblyConstraint,
    config: SimulationConfig,
}

impl GillespieSimulator {
    /// Create a new simulator.
    pub fn new(
        baseline: AssemblyBaseline,
        constraint: AssemblyConstraint,
        config: SimulationConfig,
    ) -> Self {
        Self {
            baseline,
            constraint,
            config,
        }
    }

    /// Get neighbors of a state with their effective rates.
    ///
    /// Generates:
    /// - Join transitions: current state + primitive → larger state
    /// - Split transitions: remove a part → smaller state
    /// - Decay transitions: state → empty
    fn get_neighbors(&self, state: &AssemblyState) -> Vec<Neighbor> {
        let mut neighbors = Vec::new();

        // Skip if at max depth
        if state.depth() >= self.config.max_depth {
            // Only allow split/decay at max depth
            self.add_split_neighbors(state, &mut neighbors);
            self.add_decay_neighbor(state, &mut neighbors);
            return neighbors;
        }

        // Join transitions: add a primitive
        for primitive in &self.config.primitives {
            let target = state.join_with(primitive);
            let base_rate = self.baseline.get_rate(state, &target, TransitionType::Join);
            let multiplier = self.constraint.rate_multiplier(state, &target, TransitionType::Join);
            let class_multiplier =
                self.compute_class_modifier(state, &target, TransitionType::Join);
            let effective_rate = base_rate * multiplier * class_multiplier;

            if effective_rate > self.config.min_rate_threshold {
                neighbors.push((target, effective_rate, TransitionType::Join));
            }
        }

        // Split transitions
        self.add_split_neighbors(state, &mut neighbors);

        // Decay transition
        self.add_decay_neighbor(state, &mut neighbors);

        neighbors
    }

    /// Add split neighbors (remove one part at a time).
    fn add_split_neighbors(&self, state: &AssemblyState, neighbors: &mut Vec<Neighbor>) {
        if state.total_parts() <= 1 {
            return;
        }

        for (part, &count) in state.parts() {
            // Create state with one fewer of this part
            let mut new_parts = state.parts().clone();
            if count > 1 {
                new_parts.insert(part.clone(), count - 1);
            } else {
                new_parts.remove(part);
            }

            if new_parts.is_empty() {
                continue;
            }

            let new_depth = if state.depth() > 0 {
                state.depth() - 1
            } else {
                0
            };
            let target = AssemblyState::from_parts_map(
                new_parts,
                new_depth,
                state.motifs().clone(),
            );

            let base_rate = self.baseline.get_rate(state, &target, TransitionType::Split);
            let multiplier = self.constraint.rate_multiplier(state, &target, TransitionType::Split);
            let class_multiplier =
                self.compute_class_modifier(state, &target, TransitionType::Split);
            let effective_rate = base_rate * multiplier * class_multiplier;

            if effective_rate > self.config.min_rate_threshold {
                neighbors.push((target, effective_rate, TransitionType::Split));
            }
        }
    }

    /// Add decay neighbor (state → empty).
    fn add_decay_neighbor(&self, state: &AssemblyState, neighbors: &mut Vec<Neighbor>) {
        if state.total_parts() == 0 {
            return;
        }

        let target = AssemblyState::empty();
        let base_rate = self.baseline.get_rate(state, &target, TransitionType::Decay);
        let multiplier = self.constraint.rate_multiplier(state, &target, TransitionType::Decay);
        let class_multiplier =
            self.compute_class_modifier(state, &target, TransitionType::Decay);
        let effective_rate = base_rate * multiplier * class_multiplier;

        if effective_rate > self.config.min_rate_threshold {
            neighbors.push((target, effective_rate, TransitionType::Decay));
        }
    }

    /// Run a single Gillespie simulation trajectory.
    ///
    /// Returns PathStats with sufficient statistics for importance sampling.
    pub fn simulate(&self, initial_state: &AssemblyState, rng: &mut StdRng) -> PathStats {
        let mut current_state = initial_state.clone();
        let mut current_time = 0.0;

        let mut path_stats = PathStats::new(current_state.id(), 0.0);
        let mut max_depth_reached = current_state.depth();
        let mut visit_rank: HashMap<AssemblyStateId, u32> = HashMap::new();
        let mut first_visit_time: HashMap<AssemblyStateId, f64> = HashMap::new();
        let mut next_rank: u32 = 1;
        visit_rank.insert(current_state.id(), 0);
        first_visit_time.insert(current_state.id(), 0.0);

        while current_time < self.config.t_max {
            let neighbors = self.get_neighbors(&current_state);

            if neighbors.is_empty() {
                // Absorbing state
                break;
            }

            // Extract rates with depth-gated modifier (Symmetry Break A)
            let depth_modifier = self.compute_depth_gate_modifier(max_depth_reached);
            let rates: Vec<f64> = neighbors
                .iter()
                .map(|(target, rate, transition_type)| {
                    let mut effective = *rate;
                    if self.is_reuse_transition(&current_state, target) {
                        effective *= depth_modifier;
                        if *transition_type == TransitionType::Join {
                            effective *= self.compute_founder_multiplier(
                                &current_state,
                                &visit_rank,
                            );
                        }
                    }
                    effective
                })
                .collect();
            let total_rate: f64 = rates.iter().sum();

            if total_rate <= 0.0 {
                break;
            }

            // Sample waiting time (exponential)
            let dt = -rng.gen::<f64>().ln() / total_rate;
            current_time += dt;

            if current_time > self.config.t_max {
                break;
            }

            // Sample next state (weighted by rates)
            let u: f64 = rng.gen::<f64>() * total_rate;
            let mut cumsum = 0.0;
            let mut chosen_idx = 0;
            for (i, &rate) in rates.iter().enumerate() {
                cumsum += rate;
                if u <= cumsum {
                    chosen_idx = i;
                    break;
                }
            }

            let (next_state, _, transition_type) = &neighbors[chosen_idx];

            // Record transition features (after burn-in)
            if current_time >= self.config.burn_in {
                let mut features =
                    self.extract_transition_features(&current_state, next_state, *transition_type);
                if let Some(config) = &self.config.founder_bias_config {
                    if *transition_type == TransitionType::Join
                        && self.is_reuse_transition(&current_state, next_state)
                    {
                        let source_rank =
                            *visit_rank.get(&current_state.id()).unwrap_or(&u32::MAX);
                        if source_rank <= config.founder_rank_threshold {
                            features.insert("founder_reuse".to_string(), 1.0);
                        } else {
                            features.insert("derived_reuse".to_string(), 1.0);
                        }
                    }
                }
                path_stats.record_transition(&features);
            }

            current_state = next_state.clone();
            if current_state.depth() > max_depth_reached {
                max_depth_reached = current_state.depth();
            }

            if !visit_rank.contains_key(&current_state.id()) {
                visit_rank.insert(current_state.id(), next_rank);
                first_visit_time.insert(current_state.id(), current_time);
                next_rank += 1;
            }
        }

        // Update final state and duration
        path_stats.final_state_id = current_state.id();
        path_stats.duration = current_time.min(self.config.t_max);
        path_stats.max_depth_reached = max_depth_reached as u16;
        path_stats.founder_rank = *visit_rank.get(&current_state.id()).unwrap_or(&u32::MAX);
        path_stats.first_visit_time =
            *first_visit_time.get(&current_state.id()).unwrap_or(&0.0);

        path_stats
    }

    /// Extract features from a transition for path stats.
    fn extract_transition_features(
        &self,
        source: &AssemblyState,
        target: &AssemblyState,
        transition_type: TransitionType,
    ) -> HashMap<String, f64> {
        let mut features = HashMap::new();

        // Reuse count
        let reuse = if source.is_subassembly_of(target) { 1.0 } else { 0.0 };
        features.insert("reuse_count".to_string(), reuse);

        // Depth change
        features.insert(
            "depth_change".to_string(),
            target.depth() as f64 - source.depth() as f64,
        );

        // Size change
        features.insert(
            "size_change".to_string(),
            target.size() as f64 - source.size() as f64,
        );

        // Transition type
        features.insert(
            format!("transition_{}", transition_type.as_feature_key()),
            1.0,
        );

        if let Some(config) = &self.config.context_class_config {
            if transition_type == TransitionType::Join && self.is_reuse_transition(source, target) {
                let source_class = self.state_class(source, &config.primitive_classes);
                let target_class = self.state_class(target, &config.primitive_classes);
                if let (Some(s_class), Some(t_class)) = (source_class, target_class) {
                    if s_class == t_class {
                        features.insert("reuse_same_class".to_string(), 1.0);
                    } else {
                        features.insert("reuse_cross_class".to_string(), 1.0);
                    }
                }
            }
        }

        features
    }

    /// Compute depth-gate modifier for Symmetry Break A.
    ///
    /// Returns exp(θ_depth) if max_depth >= threshold, else 1.0.
    fn compute_depth_gate_modifier(&self, max_depth_reached: u32) -> f64 {
        if let Some(threshold) = self.config.depth_gate_threshold {
            if max_depth_reached >= threshold {
                return self.config.depth_gate_theta.exp();
            }
        }
        1.0
    }

    /// Check if a transition involves reuse (source is subassembly of target).
    fn is_reuse_transition(&self, source: &AssemblyState, target: &AssemblyState) -> bool {
        source.is_subassembly_of(target)
    }

    /// Compute multiplier for founder bias (Symmetry Break C).
    fn compute_founder_multiplier(
        &self,
        state: &AssemblyState,
        visit_rank: &HashMap<AssemblyStateId, u32>,
    ) -> f64 {
        let Some(config) = &self.config.founder_bias_config else {
            return 1.0;
        };

        let rank = visit_rank.get(&state.id()).copied().unwrap_or(u32::MAX);
        if rank <= config.founder_rank_threshold {
            config.founder_bonus_theta.exp()
        } else {
            config.late_penalty_theta.exp()
        }
    }

    /// Compute multiplier for context-class interactions (Symmetry Break B).
    fn compute_class_modifier(
        &self,
        source: &AssemblyState,
        target: &AssemblyState,
        transition_type: TransitionType,
    ) -> f64 {
        let Some(config) = &self.config.context_class_config else {
            return 1.0;
        };

        if transition_type != TransitionType::Join {
            return 1.0;
        }

        if !self.is_reuse_transition(source, target) {
            return 1.0;
        }

        let source_class = self.state_class(source, &config.primitive_classes);
        let target_class = self.state_class(target, &config.primitive_classes);

        match (source_class, target_class) {
            (Some(s), Some(t)) if s == t => config.same_class_theta.exp(),
            (Some(_), Some(_)) => config.cross_class_theta.exp(),
            _ => 1.0,
        }
    }

    fn state_class(
        &self,
        state: &AssemblyState,
        class_map: &HashMap<String, String>,
    ) -> Option<String> {
        let mut counts: HashMap<String, u32> = HashMap::new();

        for (part, count) in state.parts() {
            if let Some(class_label) = class_map.get(part) {
                *counts.entry(class_label.clone()).or_insert(0) += count;
            }
        }

        counts
            .into_iter()
            .max_by_key(|(_class, count)| *count)
            .map(|(class, _)| class)
    }
}

/// Simulate multiple trajectories in parallel.
///
/// This is the main entry point for parallel simulation.
///
/// # Arguments
/// * `baseline` - Baseline rate model
/// * `constraint` - Constraint model with θ weights
/// * `config` - Simulation configuration
/// * `initial_state` - Starting state for all trajectories
/// * `n_samples` - Number of trajectories to simulate
/// * `seed` - Base RNG seed (each trajectory gets seed + trajectory_index)
///
/// # Returns
/// * Vector of PathStats, one per trajectory
pub fn simulate_trajectories_parallel(
    baseline: &AssemblyBaseline,
    constraint: &AssemblyConstraint,
    config: &SimulationConfig,
    initial_state: &AssemblyState,
    n_samples: usize,
    seed: u64,
) -> Vec<PathStats> {
    let simulator = GillespieSimulator::new(
        baseline.clone(),
        constraint.clone(),
        config.clone(),
    );

    // Parallel simulation using Rayon
    (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            simulator.simulate(initial_state, &mut rng)
        })
        .collect()
}

/// Sample final state distribution from parallel trajectories.
///
/// Returns a map from state ID to empirical probability.
pub fn sample_final_states(
    baseline: &AssemblyBaseline,
    constraint: &AssemblyConstraint,
    config: &SimulationConfig,
    initial_state: &AssemblyState,
    n_samples: usize,
    seed: u64,
) -> HashMap<u64, f64> {
    let path_stats = simulate_trajectories_parallel(
        baseline,
        constraint,
        config,
        initial_state,
        n_samples,
        seed,
    );

    let mut state_counts: HashMap<u64, usize> = HashMap::new();
    for ps in &path_stats {
        *state_counts.entry(ps.final_state_id).or_insert(0) += 1;
    }

    let total = path_stats.len() as f64;
    state_counts
        .into_iter()
        .map(|(state_id, count)| (state_id, count as f64 / total))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SimulationConfig {
        SimulationConfig {
            t_max: 10.0,
            burn_in: 2.0,
            max_depth: 3,
            min_rate_threshold: 1e-6,
            primitives: vec!["A".to_string(), "B".to_string()],
            depth_gate_threshold: None,
            depth_gate_theta: 0.0,
        }
    }

    #[test]
    fn test_single_trajectory() {
        let baseline = AssemblyBaseline::default();
        let constraint = AssemblyConstraint::null_model();
        let config = test_config();
        let simulator = GillespieSimulator::new(baseline, constraint, config);

        let initial = AssemblyState::new(&["A"], 0, None);
        let mut rng = StdRng::seed_from_u64(42);

        let path_stats = simulator.simulate(&initial, &mut rng);
        assert!(path_stats.duration > 0.0);
    }

    #[test]
    fn test_parallel_simulation() {
        let baseline = AssemblyBaseline::default();
        let constraint = AssemblyConstraint::null_model();
        let config = test_config();
        let initial = AssemblyState::new(&["A"], 0, None);

        let path_stats = simulate_trajectories_parallel(
            &baseline,
            &constraint,
            &config,
            &initial,
            10,
            42,
        );

        assert_eq!(path_stats.len(), 10);
    }

    #[test]
    fn test_final_state_distribution() {
        let baseline = AssemblyBaseline::default();
        let constraint = AssemblyConstraint::null_model();
        let config = test_config();
        let initial = AssemblyState::new(&["A"], 0, None);

        let distribution = sample_final_states(
            &baseline,
            &constraint,
            &config,
            &initial,
            100,
            42,
        );

        // Probabilities should sum to 1
        let total: f64 = distribution.values().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_constraint_affects_distribution() {
        let baseline = AssemblyBaseline::default();
        let config = test_config();
        let initial = AssemblyState::new(&["A"], 0, None);

        // Null model
        let null_constraint = AssemblyConstraint::null_model();
        let null_dist = sample_final_states(&baseline, &null_constraint, &config, &initial, 100, 42);

        // Strong reuse bonus
        let mut weights = HashMap::new();
        weights.insert("reuse_count".to_string(), 3.0);
        let reuse_constraint = AssemblyConstraint::new(weights);
        let reuse_dist = sample_final_states(&baseline, &reuse_constraint, &config, &initial, 100, 42);

        // Distributions should differ (constraint has effect)
        assert_ne!(null_dist, reuse_dist);
    }
}
