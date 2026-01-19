import pytest
import numpy as np
from persiste.plugins.assembly.recipes.symmetry_breaks import founder_persistence_diagnostic
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.observation.cached_observation import (
    CacheConfig,
    CachedAssemblyObservationModel,
    SimulationSettings,
)

def simulate_founder_data(theta, founder_rank_threshold=2, primitives=None):
    """Helper to simulate data with founder bias."""
    if primitives is None:
        primitives = ["A", "B", "C"]
    
    baseline = AssemblyBaseline(kappa=1.0)
    initial_state = AssemblyState.from_parts(primitives[:1], depth=0)
    
    constraint = AssemblyConstraint(
        feature_weights=theta,
        founder_rank_threshold=founder_rank_threshold
    )
    
    graph = AssemblyGraph(primitives, max_depth=5)
    cached_model = CachedAssemblyObservationModel(
        primitives=primitives,
        baseline=baseline,
        initial_state=initial_state,
        simulation=SimulationSettings(n_samples=500, t_max=20.0, max_depth=5),
        cache_config=CacheConfig(trust_radius=10.0, ess_threshold=0.0),
        rng_seed=42,
        graph=graph,
    )
    
    # Store constraint and get latent states
    # CachedAssemblyObservationModel uses self._current_constraint
    latent_states = cached_model.get_latent_states(constraint)
    
    return {
        "primitives": primitives,
        "cached_model": cached_model,
        "latent_states": latent_states
    }

def test_recipe_5_founder_persistence_signal():
    """Test that Recipe 5 identifies strong founder persistence."""
    # Strong bonus for founders
    theta = {"founder_reuse": 10.0}
    data = simulate_founder_data(theta, founder_rank_threshold=3)
    
    report = founder_persistence_diagnostic(
        data=data,
        founder_rank_threshold=3
    )
    
    assert len(report.founder_ids) > 0
    # With 10.0 bonus, founders should definitely persist
    assert report.persistence_ratio > 1.0
    assert report.severity in ["ok", "warning"]

def test_recipe_5_no_persistence():
    """Test that Recipe 5 detects when founders are lost (null model)."""
    # No bonus for founders
    theta = {}
    data = simulate_founder_data(theta, founder_rank_threshold=3)
    
    report = founder_persistence_diagnostic(
        data=data,
        founder_rank_threshold=3
    )
    
    assert len(report.founder_ids) > 0
    # Without bonus, persistence ratio should be around 1.0 or less
    # depending on how many states are reachable
    assert report.persistence_ratio < 5.0 # Should not be massive
