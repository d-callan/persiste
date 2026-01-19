import pytest
import numpy as np
from persiste.plugins.assembly.recipes.symmetry_breaks import symmetry_break_discrimination
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline, TransitionType
from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.observation.cached_observation import (
    CacheConfig,
    CachedAssemblyObservationModel,
    SimulationSettings,
)
from persiste.plugins.assembly.states.resolver import StateIDResolver

def simulate_mock_data(theta, primitives=None, break_params=None):
    """Helper to simulate mock data with optional symmetry break params."""
    if primitives is None:
        primitives = ["A", "B", "C"]
    
    baseline = AssemblyBaseline(kappa=1.0)
    initial_state = AssemblyState.from_parts(primitives[:1], depth=0)
    
    # Setup constraint
    constraint_params = {"feature_weights": theta}
    if break_params:
        constraint_params.update(break_params)
    constraint = AssemblyConstraint(**constraint_params)
    
    graph = AssemblyGraph(primitives, max_depth=5)
    
    # To get a signal, we need to ensure the simulation actually uses the bonus
    # and that the bonus creates a skewed distribution.
    # Let's verify that the feature extractor is actually working by manually
    # checking a transition.
    from persiste.plugins.assembly.features.assembly_features import AssemblyFeatureExtractor
    extractor = AssemblyFeatureExtractor(depth_gate_threshold=1)
    s1 = AssemblyState.from_parts(["A"], depth=0)
    s2 = AssemblyState.from_parts(["A", "B"], depth=1) # effective_depth=1
    
    # Use the globally imported TransitionType
    feats = extractor.extract_features(s1, s2, TransitionType.JOIN, target_depth=1)
    print(f"DEBUG: Feature test: reuse={feats.reuse_count}, gate={feats.depth_gate_reuse}")

    # Use higher n_samples and even stronger theta
    cached_model = CachedAssemblyObservationModel(
        primitives=primitives,
        baseline=baseline,
        initial_state=initial_state,
        simulation=SimulationSettings(n_samples=1000, t_max=50.0, max_depth=5),
        cache_config=CacheConfig(trust_radius=10.0, ess_threshold=0.0),
        rng_seed=42,
        graph=graph,
    )
    
    latent_states = cached_model.get_latent_states(constraint)
    resolver = StateIDResolver(primitives)
    
    # Generate some "observed" IDs from the latent distribution
    observed_ids = set()
    observation_records = []
    
    # Sort by probability descending and pick top states
    sorted_states = sorted(latent_states.items(), key=lambda x: x[1], reverse=True)
    for sid, prob in sorted_states[:10]: # Pick top 10 states
        if prob > 0.0:
            observed_ids.add(sid)
            # Use high frequency to boost LL sensitivity
            observation_records.append({"compound_id": sid, "frequency": 1000.0})
            
            # Debug: check features for this state
            state_obj = cached_model.graph.get_state(sid)
            if state_obj:
                print(f"DEBUG: Observed state {sid} prob={prob:.4f} depth={state_obj.assembly_depth}")
                
    # Inspect the simulation result directly to see feature counts
    # This is a bit of a hack but we can access the private cache if it was initialized
    if cached_model._cache:
        avg_depth_gate = sum(p.get("depth_gate_reuse", 0) for p in cached_model._cache.feature_counts) / len(cached_model._cache.feature_counts)
        print(f"DEBUG: Average depth_gate_reuse in cache: {avg_depth_gate:.4f}")
        
    print(f"DEBUG: Latent distribution size: {len(latent_states)}")
    print(f"DEBUG: Top 5 probabilities: {[f'{p:.4f}' for s, p in sorted_states[:5]]}")
    
    return {
        "primitives": primitives,
        "cached_model": cached_model,
        "stable_observed_ids": observed_ids,
        "observation_records": observation_records,
        "compound_to_state": {sid: sid for sid in observed_ids}
    }

def test_recipe_3_discrimination_signal():
    """Test that Recipe 3 correctly identifies a symmetry break when present."""
    # Data generated WITH a strong depth gate
    # Set threshold to 1 so that even depth 1 -> 2 transitions trigger it
    break_params = {"depth_gate_threshold": 1}
    # Strong weights to ensure signal
    theta_with_break = {"depth_gate_reuse": 10.0}
    
    # Use global reuse as well to ensure there's a baseline to distinguish from
    data = simulate_mock_data(
        theta=theta_with_break,
        break_params=break_params
    )
    
    # Compare null (zeros) vs break
    # Base model has NO weights
    report = symmetry_break_discrimination(
        data=data,
        base_theta={},
        break_theta=theta_with_break,
        break_type="A",
        break_params=break_params
    )
    
    print(f"DEBUG: base_ll={report.base_ll}, break_ll={report.break_ll}, delta={report.delta_ll}")
    
    assert report.delta_ll > 0, f"Expected positive delta_ll, got {report.delta_ll}"
    assert report.best_model == "Symmetry Break"
    assert report.severity == "ok"

def test_recipe_3_no_signal():
    """Test that Recipe 3 prefers stationary model when break adds no value."""
    # Data generated WITHOUT a break
    data = simulate_mock_data(theta={"reuse_count": 1.0})
    
    # Try to force a break that wasn't in the data
    report = symmetry_break_discrimination(
        data=data,
        base_theta={"reuse_count": 1.0},
        break_theta={"reuse_count": 1.0, "depth_gate_reuse": 0.0},
        break_type="A",
        break_params={"depth_gate_threshold": 2}
    )
    
    assert report.best_model == "Base (Stationary)"
    assert report.delta_ll <= 2.0
    assert report.severity == "warning"
