"""
High-level interface for assembly constraint inference.

Provides user-friendly helpers that wire together the assembly graph,
baseline, observation adapter, and the shared ConstraintInference engine.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from persiste.core.data import ObservedTransitions
from persiste.core.inference import ConstraintInference, ConstraintResult
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.observation import (
    AssemblyObservationModel,
    FrequencyWeightedPresenceModel,
    PresenceObservationModel,
    SimulationSettings,
)
from persiste.plugins.assembly.states.assembly_state import AssemblyState


@dataclass
class AssemblyBaselineConfig:
    """Simple container for baseline hyperparameters."""

    kappa: float = 1.0
    join_exponent: float = -0.5
    split_exponent: float = 0.3
    decay_rate: float = 0.01


@dataclass
class AssemblyGraphConfig:
    """Configuration for lazy assembly graph construction."""

    max_depth: int = 5
    min_rate_threshold: float = 1e-6


@dataclass
class PresenceObservationConfig:
    """Settings for presence-only observation models."""

    detection_prob: float = 0.9
    false_positive_prob: float = 0.01


@dataclass
class FrequencyObservationConfig:
    """Settings for frequency-weighted observation models."""

    detection_prob: float = 0.9
    false_positive_rate: float = 0.1


def fit_presence_observations(
    observed_compounds: Iterable[str],
    feature_names: Sequence[str],
    *,
    primitives: Sequence[str],
    baseline_config: AssemblyBaselineConfig | None = None,
    graph_config: AssemblyGraphConfig | None = None,
    observation_config: PresenceObservationConfig | None = None,
    simulation: SimulationSettings | None = None,
    initial_state_parts: Sequence[str] | None = None,
    inference_kwargs: dict[str, Any] | None = None,
    rng_seed: int | None = None,
) -> ConstraintResult:
    """
    Fit constraint parameters using presence/absence observations.

    Args:
        observed_compounds: Iterable of detected compound identifiers.
        feature_names: Constraint feature names to optimize (order matters).
        primitives: Primitive building blocks that seed the assembly graph.
        baseline_config: Optional baseline hyperparameters.
        graph_config: Optional graph construction parameters.
        observation_config: Presence observation hyperparameters.
        simulation: Gillespie simulation settings.
        initial_state_parts: Optional parts list for the initial assembly state.
        inference_kwargs: Extra arguments forwarded to ConstraintInference.fit().

    Returns:
        ConstraintResult with fitted parameters and diagnostics.
    """

    obs_backend = PresenceObservationModel(
        detection_prob=(observation_config or PresenceObservationConfig()).detection_prob,
        false_positive_prob=(observation_config or PresenceObservationConfig()).false_positive_prob,
    )
    adapter = _build_observation_adapter(
        primitives=primitives,
        baseline_config=baseline_config,
        graph_config=graph_config,
        obs_backend=obs_backend,
        simulation=simulation,
        initial_state_parts=initial_state_parts,
        rng_seed=rng_seed,
    )
    constraint = _initialize_constraint(feature_names)
    engine = ConstraintInference(constraint, adapter)

    data = _wrap_observed_transitions()
    data.observed_compounds = set(observed_compounds)

    kwargs = inference_kwargs.copy() if inference_kwargs else {}
    return engine.fit(data, method="MLE", **kwargs)


def fit_frequency_observations(
    observed_counts: Mapping[str, int],
    feature_names: Sequence[str],
    *,
    primitives: Sequence[str],
    baseline_config: AssemblyBaselineConfig | None = None,
    graph_config: AssemblyGraphConfig | None = None,
    observation_config: FrequencyObservationConfig | None = None,
    simulation: SimulationSettings | None = None,
    initial_state_parts: Sequence[str] | None = None,
    inference_kwargs: dict[str, Any] | None = None,
) -> ConstraintResult:
    """
    Fit constraint parameters using frequency-weighted presence observations.

    Args:
        observed_counts: Mapping from compound to observed frequency.
        feature_names: Constraint feature names to optimize.
        primitives: Primitive building blocks for the assembly graph.
        baseline_config: Optional baseline hyperparameters.
        graph_config: Optional graph construction parameters.
        observation_config: Frequency observation hyperparameters.
        simulation: Gillespie simulation settings.
        initial_state_parts: Optional initial state composition.
        inference_kwargs: Extra arguments forwarded to ConstraintInference.fit().

    Returns:
        ConstraintResult with fitted parameters and diagnostics.
    """

    obs_cfg = observation_config or FrequencyObservationConfig()
    obs_backend = FrequencyWeightedPresenceModel(
        detection_prob=obs_cfg.detection_prob,
        false_positive_rate=obs_cfg.false_positive_rate,
    )
    adapter = _build_observation_adapter(
        primitives=primitives,
        baseline_config=baseline_config,
        graph_config=graph_config,
        obs_backend=obs_backend,
        simulation=simulation,
        initial_state_parts=initial_state_parts,
    )
    constraint = _initialize_constraint(feature_names)
    engine = ConstraintInference(constraint, adapter)

    data = _wrap_observed_transitions()
    data.observed_counts = dict(observed_counts)

    kwargs = inference_kwargs.copy() if inference_kwargs else {}
    return engine.fit(data, method="MLE", **kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _initialize_constraint(feature_names: Sequence[str]) -> AssemblyConstraint:
    weights = {name: 0.0 for name in feature_names}
    return AssemblyConstraint(feature_weights=weights)


def _build_observation_adapter(
    *,
    primitives: Sequence[str],
    baseline_config: AssemblyBaselineConfig | None,
    graph_config: AssemblyGraphConfig | None,
    obs_backend: PresenceObservationModel | FrequencyWeightedPresenceModel,
    simulation: SimulationSettings | None,
    initial_state_parts: Sequence[str] | None,
    rng_seed: int | None,
) -> AssemblyObservationModel:
    if not primitives:
        raise ValueError("primitives list cannot be empty")

    graph_cfg = graph_config or AssemblyGraphConfig()
    baseline_cfg = baseline_config or AssemblyBaselineConfig()
    sim_cfg = simulation or SimulationSettings()

    graph = AssemblyGraph(
        list(primitives),
        max_depth=graph_cfg.max_depth,
        min_rate_threshold=graph_cfg.min_rate_threshold,
    )
    baseline = AssemblyBaseline(
        kappa=baseline_cfg.kappa,
        join_exponent=baseline_cfg.join_exponent,
        split_exponent=baseline_cfg.split_exponent,
        decay_rate=baseline_cfg.decay_rate,
    )

    if initial_state_parts is None:
        initial_state_parts = [primitives[0]]
    initial_state = AssemblyState.from_parts(list(initial_state_parts), depth=0)

    return AssemblyObservationModel(
        graph=graph,
        baseline=baseline,
        obs_model=obs_backend,
        initial_state=initial_state,
        simulation=sim_cfg,
        rng_seed=rng_seed,
    )


def _wrap_observed_transitions() -> ObservedTransitions:
    """
    Create a minimal ObservedTransitions instance and allow attaching metadata.
    """

    data = ObservedTransitions(counts={})
    return data
