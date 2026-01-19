//! Assembly theory module for importance-sampling cache and parallel Gillespie simulation.
//!
//! This module provides Rust implementations of:
//! - AssemblyState: Compositional state representation
//! - AssemblyBaseline: Physics-agnostic rate model
//! - PathStats: Sufficient statistics for importance sampling
//! - GillespieSimulator: Parallel trajectory simulation
//! - CacheManager: Importance-sampling cache with L∞ trust region
//! - TopologyGuard: Detection of topology-changing θ

pub mod state;
pub mod baseline;
pub mod constraint;
pub mod path_stats;
pub mod gillespie;
pub mod cache_manager;
pub mod topology_guard;

pub use state::{AssemblyState, AssemblyStateId};
pub use baseline::{AssemblyBaseline, TransitionType};
pub use constraint::{AssemblyConstraint, ContextClassConfig, FounderBiasConfig};
pub use path_stats::PathStats;
pub use gillespie::{
    GillespieSimulator,
    SimulationConfig,
    simulate_trajectories_parallel,
};
pub use cache_manager::{CacheManager, CacheConfig, CacheStatus, InvalidationReason};
pub use topology_guard::TopologyGuard;
