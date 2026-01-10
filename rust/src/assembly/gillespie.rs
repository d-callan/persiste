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
use super::state::AssemblyState;

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
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            t_max: 50.0,
            burn_in: 25.0,
            max_depth: 5,
            min_rate_threshold: 1e-6,
            primitives: vec!["A".to_string()],
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
            let effective_rate = base_rate * multiplier;

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
            let effective_rate = base_rate * multiplier;

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
        let effective_rate = base_rate * multiplier;

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

        while current_time < self.config.t_max {
            let neighbors = self.get_neighbors(&current_state);

            if neighbors.is_empty() {
                // Absorbing state
                break;
            }

            // Extract rates
            let rates: Vec<f64> = neighbors.iter().map(|(_, rate, _)| *rate).collect();
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
                let features = self.extract_transition_features(&current_state, next_state, *transition_type);
                path_stats.record_transition(&features);
            }

            current_state = next_state.clone();
        }

        // Update final state and duration
        path_stats.final_state_id = current_state.id();
        path_stats.duration = current_time.min(self.config.t_max);

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

        features
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
