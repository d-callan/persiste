"""Utilities for applying assembly constraints with shared helpers."""

from __future__ import annotations

import numpy as np

from persiste.core.constraint_utils import apply_multiplicative_constraint
from persiste.plugins.assembly.baselines.assembly_baseline import TransitionType
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.states.assembly_state import AssemblyState


def apply_assembly_constraint(
    baseline_rate: float,
    constraint: AssemblyConstraint,
    *,
    source: AssemblyState,
    target: AssemblyState,
    transition_type: TransitionType,
    max_rate: float = 1e6,
) -> float:
    """Apply an assembly constraint to a scalar baseline rate."""
    if baseline_rate < 0:
        raise ValueError(f"baseline_rate must be non-negative, got {baseline_rate}")
    if baseline_rate == 0:
        return 0.0

    epsilon = max(baseline_rate * 1e-9, 1e-12)
    rate_matrix = np.array(
        [
            [-baseline_rate, baseline_rate],
            [epsilon, -epsilon],
        ]
    )

    multipliers = constraint.get_rate_multipliers(
        context={"source": source, "target": target, "transition_type": transition_type},
    )

    constrained = apply_multiplicative_constraint(
        rate_matrix,
        multipliers,
        max_rate=max_rate,
        validator=None,
    )

    return float(constrained[0, 1])
