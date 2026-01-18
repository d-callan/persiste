"""
Gillespie simulator for assembly CTMC.

Minimal implementation:
- Stochastic simulation (no matrix exponentials)
- Finite horizon or equilibrium stopping
- Trajectory output
- Final state distribution sampling

This makes θ → λ_eff → P(state) concrete.
"""

from dataclasses import dataclass

import numpy as np

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.states.assembly_state import AssemblyState


@dataclass
class Trajectory:
    """
    Trajectory from Gillespie simulation.

    Attributes:
        states: List of visited states
        times: List of transition times
    """
    states: list[AssemblyState]
    times: list[float]

    def final_state(self) -> AssemblyState:
        """Get final state."""
        return self.states[-1]

    def duration(self) -> float:
        """Total simulation time."""
        return self.times[-1] if self.times else 0.0


class GillespieSimulator:
    """
    Gillespie algorithm for assembly CTMC.

    Simulates stochastic dynamics:
    - State = AssemblyState
    - Events = allowed transitions from current state
    - Rates = λ_eff(i→j) from baseline × constraint

    This is the minimal implementation that makes θ influence P(state).
    """

    def __init__(
        self,
        graph: AssemblyGraph,
        baseline: AssemblyBaseline,
        constraint: AssemblyConstraint,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize Gillespie simulator.

        Args:
            graph: Assembly graph (lazy generation)
            baseline: Baseline rate model
            constraint: Constraint model (defines θ)
            rng: Random number generator (default: np.random.default_rng())
        """
        self.graph = graph
        self.baseline = baseline
        self.constraint = constraint
        self.rng = rng if rng is not None else np.random.default_rng()

    def simulate(
        self,
        initial_state: AssemblyState,
        t_max: float = 100.0,
        equilibrium_threshold: float | None = None,
        burn_in: float = 0.0,
    ) -> Trajectory:
        """
        Run Gillespie simulation.

        Args:
            initial_state: Starting state
            t_max: Maximum simulation time
            equilibrium_threshold: Optional stopping criterion (not implemented)
            burn_in: Time to discard before recording (default: 0.0)

        Returns:
            Trajectory object
        """
        states = [initial_state]
        times = [0.0]

        current_state = initial_state
        current_time = 0.0

        while current_time < t_max:
            # Get neighbors with effective rates
            neighbors = self.graph.get_neighbors(current_state, self.baseline, self.constraint)

            if not neighbors:
                # No transitions available (absorbing state)
                break

            # Extract rates
            rates = np.array([rate for _, rate, _ in neighbors])
            total_rate = np.sum(rates)

            if total_rate <= 0:
                # No viable transitions
                break

            # Sample waiting time (exponential)
            dt = self.rng.exponential(1.0 / total_rate)
            current_time += dt

            if current_time > t_max:
                break

            # Sample next state (weighted by rates)
            probs = rates / total_rate
            idx = self.rng.choice(len(neighbors), p=probs)
            next_state, _, _ = neighbors[idx]

            # Record transition (if past burn-in)
            if current_time >= burn_in:
                states.append(next_state)
                times.append(current_time)

            current_state = next_state

        return Trajectory(states=states, times=times)

    def sample_final_states(
        self,
        initial_state: AssemblyState,
        n_samples: int = 100,
        t_max: float = 100.0,
        burn_in: float = 50.0,
    ) -> dict[AssemblyState, float]:
        """
        Sample final state distribution.

        Runs multiple trajectories and counts final states.

        Args:
            initial_state: Starting state
            n_samples: Number of trajectories
            t_max: Simulation time per trajectory
            burn_in: Burn-in time (discard early states)

        Returns:
            Dict mapping states to empirical probabilities
        """
        final_states = []

        for _ in range(n_samples):
            traj = self.simulate(initial_state, t_max=t_max, burn_in=burn_in)
            final_states.append(traj.final_state())

        # Count occurrences
        state_counts = {}
        for state in final_states:
            state_counts[state] = state_counts.get(state, 0) + 1

        # Convert to probabilities
        state_probs = {
            state: count / n_samples
            for state, count in state_counts.items()
        }

        return state_probs

    def estimate_stationary_distribution(
        self,
        initial_states: list[AssemblyState],
        n_samples_per_state: int = 50,
        t_max: float = 100.0,
        burn_in: float = 50.0,
    ) -> dict[AssemblyState, float]:
        """
        Estimate stationary distribution from multiple initial states.

        More robust than single initial state.

        Args:
            initial_states: List of starting states
            n_samples_per_state: Samples per initial state
            t_max: Simulation time
            burn_in: Burn-in time

        Returns:
            Dict mapping states to empirical probabilities
        """
        all_final_states = []

        for init_state in initial_states:
            for _ in range(n_samples_per_state):
                traj = self.simulate(init_state, t_max=t_max, burn_in=burn_in)
                all_final_states.append(traj.final_state())

        # Count and normalize
        state_counts = {}
        for state in all_final_states:
            state_counts[state] = state_counts.get(state, 0) + 1

        total = len(all_final_states)
        state_probs = {
            state: count / total
            for state, count in state_counts.items()
        }

        return state_probs

    def __str__(self) -> str:
        return f"GillespieSimulator(graph={self.graph})"
