"""Smoke tests for the assembly observation adapter and interface helpers."""

import numpy as np

from persiste.core.data import ObservedTransitions
from persiste.core.inference import ConstraintResult
from persiste.plugins.assembly.assembly_interface import (
    AssemblyBaselineConfig,
    AssemblyGraphConfig,
    PresenceObservationConfig,
    SimulationSettings,
    fit_presence_observations,
)
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.observation.assembly_observation import AssemblyObservationModel
from persiste.plugins.assembly.observation.presence_model import PresenceObservationModel
from persiste.plugins.assembly.states.assembly_state import AssemblyState


def test_assembly_observation_adapter_populates_latent_states() -> None:
    primitives = ["A"]
    graph = AssemblyGraph(primitives, max_depth=2, min_rate_threshold=1e-6)
    baseline = AssemblyBaseline(kappa=1.0)
    obs_backend = PresenceObservationModel(detection_prob=0.8, false_positive_prob=0.02)
    initial_state = AssemblyState.from_parts([primitives[0]], depth=0)

    adapter = AssemblyObservationModel(
        graph=graph,
        baseline=baseline,
        obs_model=obs_backend,
        initial_state=initial_state,
        simulation=SimulationSettings(n_samples=5, t_max=5.0, burn_in=1.0),
        rng_seed=123,
    )

    constraint = AssemblyConstraint(feature_weights={"reuse_count": 0.1})
    data = ObservedTransitions(counts={})
    data.observed_compounds = {"A"}

    ll = adapter.log_likelihood(data, constraint, graph=None)

    assert np.isfinite(ll)
    assert hasattr(data, "latent_states")
    assert data.latent_states  # type: ignore[attr-defined]
    assert adapter.last_latent_states is not None
    assert adapter.last_latent_states == data.latent_states  # type: ignore[attr-defined]


def test_fit_presence_observations_returns_constraint_result() -> None:
    primitives = ["A", "B"]
    observed = {"A", "B"}

    result = fit_presence_observations(
        observed_compounds=observed,
        feature_names=["reuse_count"],
        primitives=primitives,
        baseline_config=AssemblyBaselineConfig(kappa=1.0, join_exponent=-0.4, split_exponent=0.2),
        graph_config=AssemblyGraphConfig(max_depth=3, min_rate_threshold=1e-6),
        observation_config=PresenceObservationConfig(detection_prob=0.85, false_positive_prob=0.01),
        simulation=SimulationSettings(n_samples=8, t_max=8.0, burn_in=2.0),
        initial_state_parts=[primitives[0]],
        rng_seed=42,
        inference_kwargs={"options": {"maxiter": 10}},
    )

    assert isinstance(result, ConstraintResult)
    assert "reuse_count" in result.parameters.get("theta", {})
    assert np.isfinite(result.log_likelihood)
