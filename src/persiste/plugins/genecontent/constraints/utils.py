"""Helper utilities for applying gene content constraints."""

from __future__ import annotations

from persiste.core.constraint_utils import apply_multiplicative_constraint
from persiste.plugins.genecontent.baselines.gene_baseline import RateParameters
from persiste.plugins.genecontent.constraints.gene_constraint import GeneContentConstraint


def apply_gene_constraint_to_rates(
    rate_params: RateParameters,
    constraint: GeneContentConstraint,
    *,
    family_id: str,
    context: dict | None = None,
    max_rate: float | None = None,
) -> RateParameters:
    """Apply a gene content constraint to baseline gain/loss rates."""

    Q = rate_params.rate_matrix()
    multipliers = constraint.get_rate_multipliers(
        family_id=family_id,
        context=context,
    )

    constrained_Q = apply_multiplicative_constraint(
        Q,
        multipliers,
        max_rate=max_rate or 1e6,
        validator=None,
    )

    return RateParameters(
        gain_rate=float(constrained_Q[0, 1]),
        loss_rate=float(constrained_Q[1, 0]),
        family_id=family_id,
    )
