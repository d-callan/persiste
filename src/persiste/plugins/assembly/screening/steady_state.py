"""
Steady-state approximation for assembly dynamics.

Default deterministic screening model: algebraic approximation suitable for ranking.
NOT used for parameter estimation - only for hypothesis triage.
"""

import math
from dataclasses import dataclass

from persiste.plugins.assembly.baselines.assembly_baseline import (
    AssemblyBaseline,
    TransitionType,
)
from persiste.plugins.assembly.states.assembly_state import AssemblyState


@dataclass
class SteadyStateConfig:
    """Configuration for steady-state approximation."""

    max_depth: int = 5
    """Maximum assembly depth to consider."""

    max_states: int = 1000
    """Maximum number of states to enumerate."""

    convergence_tol: float = 1e-6
    """Convergence tolerance for fixed-point iteration."""

    max_iterations: int = 100
    """Maximum iterations for fixed-point solver."""


class SteadyStateAssemblyModel:
    """
    Deterministic approximation of assembly dynamics.

    Default: steady-state / algebraic approximation suitable for ranking.

    Scope (be explicit):
    - Approximates expected occupancy of state classes
    - Ignores path correlations
    - NOT used for parameter estimation
    - Screening is monotonic but biased

    Optional (experimental): coarse-grained ODE screening if steady-state
    fails to separate hypotheses cleanly.
    """

    def __init__(
        self,
        primitives: list[str],
        baseline: AssemblyBaseline,
        config: SteadyStateConfig | None = None,
    ):
        """
        Initialize steady-state model.

        Args:
            primitives: List of primitive building blocks
            baseline: Baseline rate model
            config: Configuration (default: SteadyStateConfig())
        """
        self.primitives = primitives
        self.baseline = baseline
        self.config = config or SteadyStateConfig()

        self._state_cache: dict[int, AssemblyState] = {}

    def expected_occupancy(
        self,
        theta: dict[str, float],
        initial_state: AssemblyState,
    ) -> dict[int, float]:
        """
        Compute steady-state / algebraic approximation of occupancy.

        Returns approximate P(state) under the deterministic approximation.

        This uses a simplified mean-field approach:
        - Enumerate reachable states up to max_depth
        - Compute effective rates based on θ
        - Solve for steady-state via fixed-point iteration

        Args:
            theta: Feature weights (constraint parameters)
            initial_state: Starting state

        Returns:
            Dict mapping state_id -> approximate probability
        """
        states = self._enumerate_states(initial_state)
        if not states:
            return {initial_state.id(): 1.0} if hasattr(initial_state, 'id') else {}

        # Build transition matrix approximation
        n_states = len(states)
        state_list = list(states.values())
        state_to_idx = {s.id(): i for i, s in enumerate(state_list)}

        # Initialize uniform distribution
        probs = [1.0 / n_states] * n_states

        # Fixed-point iteration
        for _ in range(self.config.max_iterations):
            new_probs = [0.0] * n_states

            for i, state in enumerate(state_list):
                # Compute outflow rate
                outflow = 0.0
                neighbors = self._get_neighbors(state, theta)

                for target, rate in neighbors:
                    if target.id() in state_to_idx:
                        j = state_to_idx[target.id()]
                        new_probs[j] += probs[i] * rate
                    outflow += rate

                # Self-loop (remaining probability)
                if outflow < 1.0:
                    new_probs[i] += probs[i] * (1.0 - min(outflow, 1.0))

            # Normalize
            total = sum(new_probs)
            if total > 0:
                new_probs = [p / total for p in new_probs]

            # Check convergence
            max_diff = max(abs(new_probs[i] - probs[i]) for i in range(n_states))
            probs = new_probs

            if max_diff < self.config.convergence_tol:
                break

        # Convert to dict
        return {state_list[i].id(): probs[i] for i in range(n_states)}

    def approximate_log_likelihood(
        self,
        theta: dict[str, float],
        observed_compounds: set[str],
        initial_state: AssemblyState,
    ) -> float:
        """
        Cheap approximate log-likelihood for screening.

        Computes approximate probability of observing the given compounds
        under the steady-state distribution.

        Args:
            theta: Feature weights
            observed_compounds: Set of observed compound identifiers
            initial_state: Starting state

        Returns:
            Approximate log-likelihood
        """
        occupancy = self.expected_occupancy(theta, initial_state)

        if not occupancy:
            return -math.inf

        # Compute observation probability
        # Simple model: compound present if any state with that part has nonzero prob
        log_lik = 0.0
        states = self._enumerate_states(initial_state)

        for compound in observed_compounds:
            # Probability that compound is present
            prob_present = 0.0
            for state_id, prob in occupancy.items():
                if state_id in states:
                    state = states[state_id]
                    if state.contains_part(compound):
                        prob_present += prob

            # Clamp to avoid log(0)
            prob_present = max(prob_present, 1e-10)
            log_lik += math.log(prob_present)

        return log_lik

    def _enumerate_states(
        self,
        initial_state: AssemblyState,
    ) -> dict[int, AssemblyState]:
        """Enumerate reachable states up to max_depth."""
        states: dict[int, AssemblyState] = {}
        frontier = [initial_state]
        states[initial_state.id()] = initial_state

        while frontier and len(states) < self.config.max_states:
            state = frontier.pop(0)

            if state.assembly_depth >= self.config.max_depth:
                continue

            # Generate neighbors
            for primitive in self.primitives:
                # Join transition
                new_state = self._join_state(state, primitive)
                if new_state.id() not in states:
                    states[new_state.id()] = new_state
                    frontier.append(new_state)

        return states

    def _join_state(self, state: AssemblyState, primitive: str) -> AssemblyState:
        """Create a new state by joining with a primitive."""
        new_parts = state.get_parts_list() + [primitive]
        return AssemblyState.from_parts(
            new_parts,
            depth=state.assembly_depth + 1,
            motifs=state.motifs if state.motifs else None,
        )

    def _get_neighbors(
        self,
        state: AssemblyState,
        theta: dict[str, float],
    ) -> list[tuple[AssemblyState, float]]:
        """Get neighbors with effective rates."""
        neighbors = []

        # Join transitions
        for primitive in self.primitives:
            target = self._join_state(state, primitive)
            base_rate = self.baseline.get_assembly_rate(
                state, target, TransitionType.JOIN
            )
            # Apply constraint
            multiplier = self._compute_multiplier(state, target, theta)
            effective_rate = base_rate * multiplier
            neighbors.append((target, effective_rate))

        return neighbors

    def _compute_multiplier(
        self,
        source: AssemblyState,
        target: AssemblyState,
        theta: dict[str, float],
    ) -> float:
        """Compute constraint multiplier exp(θ · features)."""
        contribution = 0.0

        # Reuse feature
        if source.is_subassembly_of(target):
            contribution += theta.get("reuse_count", 0.0)

        # Depth change feature
        depth_change = target.assembly_depth - source.assembly_depth
        contribution += theta.get("depth_change", 0.0) * depth_change

        # Size change feature
        size_change = target.size - source.size
        contribution += theta.get("size_change", 0.0) * size_change

        return math.exp(contribution)

    # Add id() method compatibility for AssemblyState
    def _get_state_id(self, state: AssemblyState) -> int:
        """Get a hashable ID for a state."""
        return hash(state)


# Monkey-patch AssemblyState.id() if it doesn't exist
if not hasattr(AssemblyState, 'id'):
    def _assembly_state_id(self) -> int:
        return hash(self)
    AssemblyState.id = _assembly_state_id
