"""Shared observation likelihood helpers for assembly inference paths."""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def compute_observation_ll(
    latent_states: dict[int, float],
    observed_compounds: set[str],
    primitives: list[str],
    *,
    observation_records: list[dict[str, Any]] | None = None,
    max_depth: int | None = None,
    null_latent_states: dict[int, float] | None = None,
    ess_ratio: float = 1.0,
    compound_to_state: dict[str, int] | None = None,
) -> float:
    """Generative Flux Likelihood (First-Order).

    This calculates P(Counts | Flux(θ)), comparing the observed counts of compounds
    to their expected abundance given the transition dynamics constraints θ.

    Principles:
    - Constraints (Reuse, Depth, etc.) modify transition rates.
    - Modified rates produce a global flux, resulting in a stationary distribution P(State | θ).
    - We treat observed data as a sample from this distribution (Multinomial/Poisson).
    - LL = Σ k_i * log(P(State_i | θ))
    - If null_latent_states is provided, we compute the Log-Likelihood Ratio (ΔLL)
      evidence against the null model.
    """
    if not latent_states:
        return -math.inf

    # 1. Normalize latent distributions (The "Flux")
    total_latent_mass = sum(latent_states.values())
    total_latent_mass = total_latent_mass if total_latent_mass > 0 else 1.0

    null_total_mass = 1.0
    if null_latent_states:
        null_total_mass = sum(null_latent_states.values())
        null_total_mass = null_total_mass if null_total_mass > 0 else 1.0

    # 2. Build Count Vector (k_i)
    # If enriched records provided, use frequencies. Otherwise count=1 (Presence/Absence).
    counts_by_compound: dict[str, float] = {}

    if observation_records:
        for record in observation_records:
            cid = record.get("compound_id")
            freq = record.get("frequency", 1.0)
            if cid:
                # Accumulate if multiple records for same compound (rare but possible)
                counts_by_compound[cid] = counts_by_compound.get(cid, 0.0) + freq
            else:
                # Try state_id mapping if compound_id missing
                sid = record.get("state_id")
                if sid is not None:
                    # Synthetic compound ID for lookup
                    scid = f"state_{sid}"
                    counts_by_compound[scid] = counts_by_compound.get(scid, 0.0) + freq

    # Ensure all observed_compounds are in the map
    for compound in observed_compounds:
        if compound not in counts_by_compound:
            counts_by_compound[compound] = 1.0

    # 3. Calculate Log-Likelihood
    total_ll = 0.0

    # We sum over observed species.
    # Note: For multinomial LL comparison between models with same N,
    # the unobserved terms cancel out or are constant, so summing observed is sufficient
    # for optimizing θ.

    for compound, count in counts_by_compound.items():
        state_id = None

        # 1. Check direct mapping if provided
        if compound_to_state and compound in compound_to_state:
            state_id = compound_to_state[compound]

        # 2. Resolve synthetic State ID
        if state_id is None and isinstance(compound, str) and compound.startswith("state_"):
            try:
                state_id = int(compound.split("state_")[1])
            except (IndexError, ValueError):
                pass
        elif state_id is None and isinstance(compound, int):
            # If compound is already an int, it's a stable_id
            state_id = compound

        # If we can't map to a state ID (e.g. unknown compound), we can't evaluate its
        # probability under the model.
        # Fallback: Penalty term.
        if state_id is None:
            # "Unknown" penalty.
            # If it's a primitive, we give it a small prob.
            # If it's complex, we give it a smaller prob.
            prob = 0.9 if compound in primitives else 1e-6
            total_ll += count * math.log(prob)

            if null_latent_states:
                # Null model suffers same penalty -> ΔLL = 0 for unmapped things
                total_ll -= count * math.log(prob)
            continue

        # Get Probability under Theta (The Flux)
        raw_prob = latent_states.get(state_id, 0.0)
        theta_prob = max(raw_prob / total_latent_mass, 1e-9)

        term = count * math.log(theta_prob)

        # Subtract Null Model (Log-Likelihood Ratio)
        if null_latent_states:
            null_raw = null_latent_states.get(state_id, 0.0)
            null_prob = max(null_raw / null_total_mass, 1e-9)
            term -= count * math.log(null_prob)

        total_ll += term

    # 4. Apply Importance Sampling Correction
    # If the simulator's ESS is low, our estimate of P(State|θ) is noisy.
    # We downweight the likelihood to avoid overfitting to noise.
    # IMPORTANT: ESS scaling should only be applied to the ΔLL (Log-Likelihood Ratio),
    # not absolute LL, to ensure that at the reference point (ESS ratio = 1.0),
    # the stochastic LL matches the expected analytic LL.
    if null_latent_states:
        total_ll *= max(0.0, min(1.0, ess_ratio))

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Flux LL | compounds=%d count_mass=%.1f loglik=%.3f ess=%.2f",
            len(counts_by_compound),
            sum(counts_by_compound.values()),
            total_ll,
            ess_ratio
        )

    return total_ll
