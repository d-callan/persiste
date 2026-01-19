"""
Baseline Sanity Check for Assembly Plugin.

Detects whether baseline dynamics poorly explain observations using
low-dimensional, robust summaries that avoid rare states and exact topology.
"""

from dataclasses import dataclass

import numpy as np
import persiste_rust

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.states.assembly_state import AssemblyState

DELTA_LL_MULTIPLIERS = {
    "none": 1.0,
    "mild": 2.0,
    "severe": 3.0,
}


@dataclass
class BaselineSanityResult:
    """
    Result from baseline sanity check.

    Attributes:
        baseline_ok: Whether baseline appears consistent with observations
        warning_level: Severity level ('none', 'mild', 'severe')
        observed_summary: Low-dimensional summaries from observations
        expected_summary: Low-dimensional summaries from baseline simulation
        divergence_score: Simple heuristic divergence measure
        message: Human-readable message
        delta_ll_multiplier: Threshold adjustment multiplier
    """

    baseline_ok: bool
    warning_level: str
    observed_summary: dict[str, float]
    expected_summary: dict[str, float]
    divergence_score: float
    message: str
    delta_ll_multiplier: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "baseline_ok": self.baseline_ok,
            "warning_level": self.warning_level,
            "observed_summary": self.observed_summary,
            "expected_summary": self.expected_summary,
            "divergence_score": self.divergence_score,
            "message": self.message,
            "delta_ll_multiplier": self.delta_ll_multiplier,
        }


def check_baseline_sanity(
    observed_compounds: set[str],
    primitives: list[str],
    baseline: AssemblyBaseline,
    initial_state: AssemblyState,
    n_samples: int = 100,
    max_depth: int = 5,
    seed: int = 42,
) -> BaselineSanityResult:
    """
    Quick baseline sanity check.

    Compares low-dimensional, robust summaries between observed data
    and baseline simulation at θ=0.

    Summaries used (robust to rare states):
    - Total compound count
    - Compound diversity (fraction of primitives observed)
    - Depth statistics (mean, std)

    Args:
        observed_compounds: Set of observed compound identifiers
        primitives: List of primitive building blocks
        baseline: AssemblyBaseline instance
        initial_state: Initial assembly state
        n_samples: Number of simulation samples (default: 100)
        max_depth: Maximum assembly depth (default: 5)
        seed: RNG seed for reproducibility

    Returns:
        BaselineSanityResult with divergence assessment
    """
    # Simulate under null (θ=0)
    simulation_result = persiste_rust.simulate_assembly_trajectories(
        primitives=primitives,
        initial_parts=initial_state.get_parts_list(),
        theta={},
        n_samples=n_samples,
        t_max=50.0,
        burn_in=25.0,
        max_depth=max_depth,
        seed=seed,
        kappa=baseline.kappa,
        join_exponent=baseline.join_exponent,
        split_exponent=baseline.split_exponent,
        decay_rate=baseline.decay_rate,
        initial_state_id=initial_state.stable_id,
    )
    null_trajectories = simulation_result["paths"]

    # Compute expected summaries (low-dimensional, robust)
    # 1. Unique compounds across all trajectories
    all_expected_compounds: set[str] = set()
    expected_depths: list[int] = []

    for traj in null_trajectories:
        # Extract compounds from feature counts or final state
        feature_counts = traj.get("feature_counts", {})
        # Approximate: count unique primitives that appear
        for prim in primitives:
            if feature_counts.get("reuse_count", 0) > 0 or prim in primitives[:2]:
                all_expected_compounds.add(prim)

        # Depth from transitions
        depth = feature_counts.get("depth_change", 0) + 1
        expected_depths.append(max(1, depth))

    expected_compound_count = len(all_expected_compounds)
    expected_depth_mean = float(np.mean(expected_depths)) if expected_depths else 1.0
    expected_depth_std = float(np.std(expected_depths)) if expected_depths else 0.0
    expected_diversity = expected_compound_count / max(len(primitives), 1)

    # Compute observed summaries
    observed_compound_count = len(observed_compounds)
    observed_diversity = observed_compound_count / max(len(primitives), 1)

    # Divergence score (simple heuristic)
    # Avoid rare states, exact topology - use robust aggregates only
    count_diff = abs(observed_compound_count - expected_compound_count) / max(
        expected_compound_count, 1
    )
    diversity_diff = abs(observed_diversity - expected_diversity)
    divergence_score = max(count_diff, diversity_diff)

    # Classify with multiplier-based threshold adjustment
    if divergence_score > 2.0:
        warning_level = "severe"
        baseline_ok = False
        message = (
            "Baseline dynamics poorly explain observations; "
            "constraint inference may be biased"
        )
    elif divergence_score > 1.0:
        warning_level = "mild"
        baseline_ok = False
        message = "Moderate baseline mismatch detected; interpret constraints cautiously"
    else:
        warning_level = "none"
        baseline_ok = True
        message = "Baseline appears consistent with observations"

    return BaselineSanityResult(
        baseline_ok=baseline_ok,
        warning_level=warning_level,
        observed_summary={
            "compound_count": float(observed_compound_count),
            "diversity": observed_diversity,
        },
        expected_summary={
            "compound_count": float(expected_compound_count),
            "depth_mean": expected_depth_mean,
            "depth_std": expected_depth_std,
            "diversity": expected_diversity,
        },
        divergence_score=divergence_score,
        message=message,
        delta_ll_multiplier=DELTA_LL_MULTIPLIERS[warning_level],
    )
