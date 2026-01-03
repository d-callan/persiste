"""Core tip-observation utilities shared across plugins."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class OneHotTipObservation:
    """Deterministic observation that returns one-hot likelihood vectors."""

    n_states: int

    def __post_init__(self) -> None:
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")

    def get_tip_likelihood(self, observed_state: int) -> np.ndarray:
        state = int(observed_state)
        if state < 0 or state >= self.n_states:
            raise ValueError(
                f"observed_state must be within [0, {self.n_states - 1}], got {observed_state}"
            )
        likelihood = np.zeros(self.n_states)
        likelihood[state] = 1.0
        return likelihood

    def get_tip_likelihoods_matrix(self, observed_states: np.ndarray) -> np.ndarray:
        states = np.asarray(observed_states, dtype=int)
        likelihoods = np.zeros((states.shape[0], self.n_states))
        for idx, state in enumerate(states):
            likelihoods[idx] = self.get_tip_likelihood(int(state))
        return likelihoods
