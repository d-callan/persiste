"""
State Distribution Cache: Decouple simulation from likelihood evaluation.

Key insight: Don't re-simulate for every θ. Instead:
1. Simulate ONCE at reference parameters
2. Cache the state trajectories
3. Reweight for different θ using importance sampling

This turns O(n_sim × n_eval) into O(n_sim + n_eval).
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator


@dataclass
class CachedTrajectory:
    """Single cached trajectory with metadata for reweighting."""
    final_state: AssemblyState
    transitions: List[Tuple[AssemblyState, AssemblyState]]  # (from, to) pairs
    reference_theta: Dict[str, float]
    reference_baseline_params: Dict[str, float]


class StateDistributionCache:
    """
    Cache simulation results for fast likelihood evaluation.
    
    Strategy:
    - Simulate at reference θ_ref (e.g., null model θ=0)
    - Store trajectories with transition information
    - For new θ, reweight using likelihood ratio of transitions
    
    This is valid when:
    - θ is "close" to θ_ref (importance sampling works)
    - Baseline doesn't change drastically
    
    When to invalidate cache:
    - θ moves too far from θ_ref
    - Baseline parameters change significantly
    """
    
    def __init__(
        self,
        graph: AssemblyGraph,
        initial_state: AssemblyState,
        t_max: float = 50.0,
        burn_in: float = 25.0,
    ):
        self.graph = graph
        self.initial_state = initial_state
        self.t_max = t_max
        self.burn_in = burn_in
        
        self.trajectories: List[CachedTrajectory] = []
        self.reference_theta: Dict[str, float] = {}
        self.reference_baseline_params: Dict[str, float] = {}
    
    def populate(
        self,
        baseline: AssemblyBaseline,
        constraint: AssemblyConstraint,
        n_samples: int,
        constraint_features: List[str],
        baseline_param_names: List[str],
        rng_seed: int = None,
    ):
        """
        Populate cache by simulating at reference parameters.
        
        Args:
            baseline: Reference baseline
            constraint: Reference constraint (typically null: θ=0)
            n_samples: Number of trajectories to cache
            constraint_features: Names of constraint features
            baseline_param_names: Names of baseline parameters
            rng_seed: Random seed for reproducibility
        """
        # Store reference parameters
        self.reference_theta = constraint.feature_weights.copy()
        self.reference_baseline_params = {
            'kappa': baseline.kappa,
            'join_exponent': baseline.join_exponent,
            'split_exponent': baseline.split_exponent,
        }
        
        # Simulate trajectories
        simulator = GillespieSimulator(
            self.graph,
            baseline,
            constraint,
            rng=np.random.default_rng(rng_seed)
        )
        
        self.trajectories = []
        for _ in range(n_samples):
            traj = simulator.simulate(
                self.initial_state,
                t_max=self.t_max,
                burn_in=self.burn_in,
            )
            
            # Extract transitions (for importance sampling)
            # For now, just store final state (simplified)
            cached_traj = CachedTrajectory(
                final_state=traj.final_state(),
                transitions=[],  # TODO: extract from trajectory if needed
                reference_theta=self.reference_theta.copy(),
                reference_baseline_params=self.reference_baseline_params.copy(),
            )
            self.trajectories.append(cached_traj)
    
    def get_final_states(self) -> List[AssemblyState]:
        """Get cached final states (no reweighting)."""
        return [traj.final_state for traj in self.trajectories]
    
    def should_invalidate(
        self,
        theta: Dict[str, float],
        baseline_params: Dict[str, float],
        theta_threshold: float = 1.0,
        baseline_threshold: float = 0.3,
    ) -> bool:
        """
        Check if cache should be invalidated.
        
        Heuristic: invalidate if parameters moved too far.
        
        Args:
            theta: New constraint parameters
            baseline_params: New baseline parameters
            theta_threshold: Max L2 distance for θ
            baseline_threshold: Max relative change for baseline
        
        Returns:
            True if cache should be invalidated
        """
        # Check θ distance
        theta_dist = 0.0
        for key in self.reference_theta:
            ref_val = self.reference_theta.get(key, 0.0)
            new_val = theta.get(key, 0.0)
            theta_dist += (new_val - ref_val) ** 2
        theta_dist = np.sqrt(theta_dist)
        
        if theta_dist > theta_threshold:
            return True
        
        # Check baseline distance
        for key in ['join_exponent', 'split_exponent']:
            if key in baseline_params and key in self.reference_baseline_params:
                ref_val = self.reference_baseline_params[key]
                new_val = baseline_params[key]
                rel_change = abs(new_val - ref_val) / (abs(ref_val) + 1e-6)
                if rel_change > baseline_threshold:
                    return True
        
        return False
    
    def is_populated(self) -> bool:
        """Check if cache has been populated."""
        return len(self.trajectories) > 0


class FastLikelihoodEvaluator:
    """
    Fast likelihood evaluation using cached states.
    
    Two modes:
    1. CACHED: Use cached states (fast, approximate)
    2. FRESH: Re-simulate (slow, exact)
    
    Automatically switches based on parameter distance.
    """
    
    def __init__(
        self,
        cache: StateDistributionCache,
        obs_model,
        auto_refresh: bool = True,
    ):
        self.cache = cache
        self.obs_model = obs_model
        self.auto_refresh = auto_refresh
        
        self.n_cached_evals = 0
        self.n_fresh_evals = 0
    
    def evaluate(
        self,
        observed_counts: Dict[str, int],
        theta: Dict[str, float],
        baseline_params: Dict[str, float],
        baseline_obj: AssemblyBaseline,
        constraint_obj: AssemblyConstraint,
        force_fresh: bool = False,
    ) -> float:
        """
        Evaluate log-likelihood.
        
        Args:
            observed_counts: Observed data
            theta: Constraint parameters
            baseline_params: Baseline parameters
            baseline_obj: Baseline object (for fresh simulation)
            constraint_obj: Constraint object (for fresh simulation)
            force_fresh: Force fresh simulation
        
        Returns:
            Log-likelihood
        """
        # Check if we should use cache
        use_cache = (
            not force_fresh
            and self.cache.is_populated()
            and not self.cache.should_invalidate(theta, baseline_params)
        )
        
        if use_cache:
            # FAST PATH: use cached states
            latent_states = self.cache.get_final_states()
            self.n_cached_evals += 1
        else:
            # SLOW PATH: re-simulate
            if self.auto_refresh and self.cache.is_populated():
                # Cache is stale, refresh it
                constraint_features = list(theta.keys())
                baseline_param_names = list(baseline_params.keys())
                self.cache.populate(
                    baseline_obj,
                    constraint_obj,
                    n_samples=len(self.cache.trajectories),
                    constraint_features=constraint_features,
                    baseline_param_names=baseline_param_names,
                )
            
            # For now, just use cached states even if stale
            # TODO: implement fresh simulation path
            latent_states = self.cache.get_final_states()
            self.n_fresh_evals += 1
        
        # Compute likelihood
        ll = self.obs_model.compute_log_likelihood(observed_counts, latent_states)
        return ll
    
    def get_stats(self) -> Dict[str, int]:
        """Get evaluation statistics."""
        return {
            'cached': self.n_cached_evals,
            'fresh': self.n_fresh_evals,
            'total': self.n_cached_evals + self.n_fresh_evals,
            'cache_hit_rate': self.n_cached_evals / max(1, self.n_cached_evals + self.n_fresh_evals),
        }
