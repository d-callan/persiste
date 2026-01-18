"""
Adapters that bridge assembly observation backends to the shared ObservationModel contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from persiste.core.data import ObservedTransitions
from persiste.core.observation_models import ObservationModel
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator, Trajectory
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.observation.timeslice_model import TimeSlicedPresenceModel
from persiste.plugins.assembly.states.assembly_state import AssemblyState


@dataclass
class SimulationSettings:
    """Configuration for Gillespie simulations used by the adapter."""

    n_samples: int = 50
    t_max: float = 50.0
    burn_in: float = 25.0

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.t_max <= 0:
            raise ValueError("t_max must be positive")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative")


class AssemblyObservationModel(ObservationModel):
    """
    Adapter that turns assembly observation backends into full ObservationModels.

    This class performs the latent-state simulation step that existing assembly
    observation models expect, allowing them to satisfy the
    ``log_likelihood(data, baseline, graph)`` contract required by
    :class:`persiste.core.inference.ConstraintInference`.
    """

    def __init__(
        self,
        *,
        graph: AssemblyGraph,
        baseline: AssemblyBaseline,
        obs_model: ObservationModel,
        initial_state: AssemblyState,
        simulation: SimulationSettings | None = None,
        rng_seed: int | None = None,
        require_trajectories: bool | None = None,
    ) -> None:
        """
        Initialize the adapter.

        Args:
            graph: Assembly graph used for Gillespie simulation.
            baseline: Physics-agnostic baseline supplying raw rates.
            obs_model: Downstream observation model (presence, fragments, etc.).
            initial_state: Starting state for simulations.
            simulation: Optional simulation settings (samples, horizon, burn-in).
            rng_seed: Optional seed for reproducible simulations.
            require_trajectories: Force trajectory simulation (default infers based on obs_model).
        """
        self.graph = graph
        self.baseline = baseline
        self.obs_model = obs_model
        self.initial_state = initial_state
        self.simulation = simulation or SimulationSettings()

        if require_trajectories is None:
            require_trajectories = isinstance(obs_model, TimeSlicedPresenceModel)
        self.require_trajectories = require_trajectories

        self._seed_seq = np.random.SeedSequence(rng_seed) if rng_seed is not None else None
        self._last_latent_states: dict[AssemblyState, float] | None = None
        self._last_trajectories: list[Trajectory] | None = None

    def rate(self, i: int, j: int) -> float:  # pragma: no cover
        """Delegates to the wrapped observation model."""
        return self.obs_model.rate(i, j)

    def log_likelihood(
        self,
        data: ObservedTransitions,
        baseline: Any,
        graph: Any,
    ) -> float:
        """
        Compute the likelihood after ensuring required latent data are available.

        Args:
            data: ObservedTransitions augmented with domain-specific attributes.
            baseline: Expected to be an AssemblyConstraint instance (from ConstraintInference).
            graph: TransitionGraph (unused; adapter tracks graph internally).

        DEVIATION RATIONALE:
        Core PERSISTE expects 'baseline' to be a Baseline object. In assembly,
        constraints are applied dynamically during lazy graph traversal, so
        ConstraintInference.get_constrained_baseline() returns an AssemblyConstraint.
        """
        if not isinstance(baseline, AssemblyConstraint):
            raise TypeError(
                "AssemblyObservationModel expects an AssemblyConstraint instance from "
                "ConstraintInference.get_constrained_baseline()"
            )

        obs_data = data
        trajectories = getattr(obs_data, "trajectories", None)
        latent_states = getattr(obs_data, "latent_states", None)

        if self.require_trajectories and trajectories is None:
            trajectories = self._simulate_trajectories(baseline)
            setattr(obs_data, "trajectories", trajectories)
            latent_states = self._latent_states_from_trajectories(trajectories)
            setattr(obs_data, "latent_states", latent_states)
        elif latent_states is None:
            latent_states = self._simulate_latent_states(baseline)
            setattr(obs_data, "latent_states", latent_states)

        return self.obs_model.log_likelihood(obs_data, self.baseline, self.graph)

    @property
    def last_latent_states(self) -> dict[AssemblyState, float] | None:
        """Return latent state distribution from the most recent evaluation."""
        return self._last_latent_states

    @property
    def last_trajectories(self) -> list[Trajectory] | None:
        """Return cached trajectories from the most recent evaluation, if any."""
        return self._last_trajectories

    def _simulate_latent_states(self, constraint: AssemblyConstraint) -> dict[AssemblyState, float]:
        simulator = GillespieSimulator(
            self.graph,
            self.baseline,
            constraint,
            rng=self._make_rng(),
        )
        latent_states = simulator.sample_final_states(
            self.initial_state,
            n_samples=self.simulation.n_samples,
            t_max=self.simulation.t_max,
            burn_in=self.simulation.burn_in,
        )
        self._last_latent_states = latent_states
        self._last_trajectories = None
        return latent_states

    def _simulate_trajectories(self, constraint: AssemblyConstraint) -> list[Trajectory]:
        simulator = GillespieSimulator(
            self.graph,
            self.baseline,
            constraint,
            rng=self._make_rng(),
        )
        trajectories: list[Trajectory] = []
        for _ in range(self.simulation.n_samples):
            traj = simulator.simulate(
                self.initial_state,
                t_max=self.simulation.t_max,
                burn_in=self.simulation.burn_in,
            )
            trajectories.append(traj)

        self._last_trajectories = trajectories
        self._last_latent_states = self._latent_states_from_trajectories(trajectories)
        return trajectories

    def _latent_states_from_trajectories(
        self,
        trajectories: list[Trajectory],
    ) -> dict[AssemblyState, float]:
        if not trajectories:
            return {}

        counts: dict[AssemblyState, int] = {}
        for traj in trajectories:
            final_state = traj.final_state()
            counts[final_state] = counts.get(final_state, 0) + 1

        total = len(trajectories)
        return {state: count / total for state, count in counts.items()}

    def _make_rng(self) -> np.random.Generator:
        if self._seed_seq is None:
            return np.random.default_rng()
        child = self._seed_seq.spawn(1)[0]
        return np.random.default_rng(child)
