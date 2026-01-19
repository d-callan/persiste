import pytest
import numpy as np
from persiste.plugins.assembly.recipes.symmetry_breaks import context_class_sensitivity
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
from persiste.plugins.assembly.features.assembly_features import AssemblyFeatureExtractor

def simulate_context_data(theta, primitive_classes, primitives=None):
    """Helper to simulate data with context classes."""
    if primitives is None:
        primitives = list(primitive_classes.keys())
    
    baseline = AssemblyBaseline(kappa=1.0)
    initial_state = AssemblyState.from_parts(primitives[:1], depth=0)
    
    constraint = AssemblyConstraint(
        feature_weights=theta,
        primitive_classes=primitive_classes
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

    latent_states = cached_model.get_latent_states(constraint)
    
    observed_ids = set()
    observation_records = []
    
    # Sort by probability descending and pick top states
    sorted_states = sorted(latent_states.items(), key=lambda x: x[1], reverse=True)
    
    # Check individual transition features for ALL states in the latent distribution
    # not just the observed ones, to see if the bonus is applied anywhere.
    found_bonus = False
    for sid, prob in latent_states.items():
        state_obj = cached_model.graph.get_state(sid)
        if state_obj:
            for primitive_state in cached_model.graph.get_primitive_states():
                target = cached_model.graph._join_states(state_obj, primitive_state)
                feats = constraint.feature_extractor.extract_features(state_obj, target, TransitionType.JOIN)
                fd = feats.to_dict()
                if fd.get("same_class_reuse", 0) > 0 or fd.get("cross_class_reuse", 0) > 0:
                    if not found_bonus:
                        found_bonus = True
    
    if not found_bonus:
        print("DEBUG: WARNING - No same_class_reuse or cross_class_reuse features found in ANY transitions!")

    for sid, prob in sorted_states[:10]:
        if prob > 0.0:
            observed_ids.add(sid)
            observation_records.append({"compound_id": sid, "frequency": 1000.0})
            
    return {
        "primitives": primitives,
        "cached_model": cached_model,
        "stable_observed_ids": observed_ids,
        "observation_records": observation_records,
        "compound_to_state": {sid: sid for sid in observed_ids}
    }

def test_recipe_4_context_sensitivity_signal():
    """Test that Recipe 4 identifies real context class signal."""
    # Use 6 primitives and 3 classes
    primitives = ["A", "B", "C", "D", "E", "F"]
    primitive_classes = {
        "A": "class1", "B": "class1", 
        "C": "class2", "D": "class2",
        "E": "class3", "F": "class3"
    }
    # Very strong signal
    theta = {"same_class_reuse": 10.0, "cross_class_reuse": -10.0}
    
    data = simulate_context_data(theta, primitive_classes, primitives=primitives)
    
    report = context_class_sensitivity(
        data=data,
        theta=theta,
        primitive_classes=primitive_classes,
        n_shuffles=10 # Reduced shuffles for speed
    )
    
    # Shuffled classes should perform much worse than the true classes
    assert report.severity == "ok"
    assert report.p_value < 0.15

def test_recipe_4_no_signal():
    """Test that Recipe 4 detects when context classes are arbitrary."""
    primitive_classes = {"A": "class1", "B": "class1", "C": "class2"}
    # No class-based weights
    theta = {"same_class_reuse": 0.0, "cross_class_reuse": 0.0}
    
    data = simulate_context_data(theta, primitive_classes)
    
    report = context_class_sensitivity(
        data=data,
        theta={"same_class_reuse": 1.0, "cross_class_reuse": 1.0},
        primitive_classes=primitive_classes,
        n_shuffles=5
    )
    
    # Shuffling should not significantly hurt LL if there was no signal
    assert report.severity in ["warning", "fail"]
