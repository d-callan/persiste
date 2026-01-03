"""Shared utilities for multiplicative constraints across plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

MAX_RATE = 1e6


class MultiplicativeConstraint(ABC):
    """Base class for constraints that scale transition rates."""

    @abstractmethod
    def get_rate_multipliers(
        self,
        theta: float | None = None,
        *,
        family_idx: int | None = None,
        family_id: str | None = None,
        lineage_id: str | None = None,
        context: dict | None = None,
    ) -> dict[tuple[int, int], float]:
        """Return multiplicative factors for affected transitions."""
        raise NotImplementedError


def apply_multiplicative_constraint(
    baseline_q: np.ndarray,
    multipliers: dict[tuple[int, int], float],
    *,
    max_rate: float = MAX_RATE,
    validator: Callable[[np.ndarray], None] | None = None,
) -> np.ndarray:
    """Apply multiplicative modifiers to a CTMC rate matrix."""
    out = baseline_q.copy()

    for (i, j), mult in multipliers.items():
        if not np.isfinite(mult) or mult < 0:
            raise ValueError(
                f"Constraint produced invalid multiplier {mult} for transition {(i, j)}"
            )
        out[i, j] *= mult
        if out[i, j] > max_rate:
            raise ValueError(
                f"Constraint produced excessively large rate {out[i, j]} for transition {(i, j)}"
            )

    if not np.isfinite(out).all():
        raise ValueError("Constraint produced non-finite entries in rate matrix")

    for i in range(out.shape[0]):
        row_sum = out[i, :].sum() - out[i, i]
        out[i, i] = -row_sum
        if out[i, i] >= 0:
            raise ValueError(
                f"Invalid diagonal entry after constraint application at row {i}: {out[i, i]}"
            )

    if validator is not None:
        validator(out)

    return out
