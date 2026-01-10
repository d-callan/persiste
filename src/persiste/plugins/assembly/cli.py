"""
CLI entry points for assembly constraint inference.

Modes:
- screen-only: Fast deterministic screening, no stochastic refinement
- screen-and-refine: Screen candidates, refine winners stochastically
- full-stochastic: No shortcuts, run full Gillespie-based inference
"""

import argparse
import json
import logging
import sys
from enum import Enum

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.observation.cached_observation import (
    CachedAssemblyObservationModel,
    CacheConfig,
    SimulationSettings,
)
from persiste.plugins.assembly.screening.screening import (
    AdaptiveScreeningGrid,
    adaptive_screen,
)
from persiste.plugins.assembly.screening.steady_state import (
    SteadyStateAssemblyModel,
    SteadyStateConfig,
)
from persiste.plugins.assembly.states.assembly_state import AssemblyState

logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    """Inference mode for assembly constraints."""

    SCREEN_ONLY = "screen-only"
    """Fast, deterministic only."""

    SCREEN_AND_REFINE = "screen-and-refine"
    """Screen then stochastic."""

    FULL_STOCHASTIC = "full-stochastic"
    """No shortcuts, explicit slow."""


class TrustRegionType(Enum):
    """Trust region type for cache management."""

    LINF = "linf"
    """L∞ (per-coordinate ±δ) - default."""

    MAHALANOBIS = "mahalanobis"
    """Mahalanobis (experimental, late-stage only)."""


class ScreeningGridMode(Enum):
    """Screening grid mode."""

    AUTO = "auto"
    """Hybrid adaptive (default)."""

    MANUAL = "manual"
    """User provides ranges/grid."""


def fit_assembly_constraints(
    observed_compounds: set[str],
    primitives: list[str],
    *,
    mode: InferenceMode = InferenceMode.FULL_STOCHASTIC,
    feature_names: list[str] | None = None,
    # Screening options
    screening_threshold: float = 2.0,
    screen_grid: ScreeningGridMode = ScreeningGridMode.AUTO,
    screen_budget: int = 100,
    screen_topk: int = 10,
    screen_refine_radius: float = 0.5,
    # Cache options
    trust_region: TrustRegionType = TrustRegionType.LINF,
    trust_radius: float = 1.0,
    ess_threshold: float = 0.3,
    # Simulation options
    n_samples: int = 100,
    t_max: float = 50.0,
    burn_in: float = 25.0,
    max_depth: int = 5,
    seed: int = 42,
    # Baseline options
    kappa: float = 1.0,
    join_exponent: float = -0.5,
    split_exponent: float = 0.3,
    decay_rate: float = 0.01,
) -> dict:
    """
    Fit assembly constraints with configurable inference mode.

    Args:
        observed_compounds: Set of observed compound identifiers
        primitives: List of primitive building blocks
        mode: Inference mode (screen-only, screen-and-refine, full-stochastic)
        feature_names: Feature names to consider (default: reuse_count, depth_change)
        screening_threshold: Normalized ΔLL threshold for screening
        screen_grid: Screening grid mode (auto or manual)
        screen_budget: Total deterministic evaluations for screening
        screen_topk: Number of local refinements in screening
        screen_refine_radius: Size of local neighborhood around winners
        trust_region: Trust region type (linf or mahalanobis)
        trust_radius: Trust region radius for cache
        ess_threshold: ESS threshold ratio for cache
        n_samples: Number of simulation trajectories
        t_max: Maximum simulation time
        burn_in: Burn-in time
        max_depth: Maximum assembly depth
        seed: RNG seed
        kappa: Baseline rate constant
        join_exponent: Baseline join exponent
        split_exponent: Baseline split exponent
        decay_rate: Baseline decay rate

    Returns:
        Dict with keys:
        - mode: Inference mode used
        - theta_hat: Estimated feature weights
        - screening_results: List of screening results (if applicable)
        - cache_stats: Cache statistics (if applicable)
    """
    if feature_names is None:
        feature_names = ["reuse_count", "depth_change"]

    baseline = AssemblyBaseline(
        kappa=kappa,
        join_exponent=join_exponent,
        split_exponent=split_exponent,
        decay_rate=decay_rate,
    )

    initial_state = AssemblyState.from_parts(primitives[:1] if primitives else ["A"], depth=0)

    result = {
        "mode": mode.value,
        "theta_hat": {},
        "screening_results": [],
        "cache_stats": None,
    }

    # Phase 1: Screening (if not full-stochastic)
    if mode in (InferenceMode.SCREEN_ONLY, InferenceMode.SCREEN_AND_REFINE):
        logger.info(f"Running deterministic screening (mode={mode.value})")

        # Create steady-state model
        ss_model = SteadyStateAssemblyModel(
            primitives=primitives,
            baseline=baseline,
            config=SteadyStateConfig(max_depth=max_depth),
        )

        # Create screening grid
        grid = AdaptiveScreeningGrid(
            feature_names=feature_names,
            budget=screen_budget,
            top_k=screen_topk,
            refine_radius=screen_refine_radius,
        )

        # Run adaptive screening
        screening_results = adaptive_screen(
            model=ss_model,
            observed_compounds=observed_compounds,
            initial_state=initial_state,
            grid=grid,
            threshold=screening_threshold,
        )

        result["screening_results"] = [
            {
                "theta": r.theta,
                "delta_ll": r.delta_ll,
                "normalized_delta_ll": r.normalized_delta_ll,
                "passed": r.passed,
                "rank": r.rank,
            }
            for r in screening_results
        ]

        # Get best hypothesis
        if screening_results and screening_results[0].passed:
            result["theta_hat"] = screening_results[0].theta
        else:
            result["theta_hat"] = {}

        if mode == InferenceMode.SCREEN_ONLY:
            logger.info(f"Screening complete. Best θ: {result['theta_hat']}")
            return result

    # Phase 2: Stochastic refinement (if screen-and-refine or full-stochastic)
    if mode in (InferenceMode.SCREEN_AND_REFINE, InferenceMode.FULL_STOCHASTIC):
        logger.info(f"Running stochastic inference (mode={mode.value})")

        # Create cached observation model
        cached_model = CachedAssemblyObservationModel(
            primitives=primitives,
            baseline=baseline,
            initial_state=initial_state,
            simulation=SimulationSettings(
                n_samples=n_samples,
                t_max=t_max,
                burn_in=burn_in,
                max_depth=max_depth,
            ),
            cache_config=CacheConfig(
                trust_radius=trust_radius,
                ess_threshold=ess_threshold,
            ),
            rng_seed=seed,
        )

        # Start from screening result if available
        theta_init = result["theta_hat"] if result["theta_hat"] else {}

        # Get latent states at initial theta
        constraint = AssemblyConstraint(feature_weights=theta_init)
        latent_states = cached_model.get_latent_states(constraint)

        # Simple refinement: evaluate a few nearby points
        # (Full optimization would use scipy.optimize or similar)
        best_theta = theta_init
        best_ll = _compute_observation_ll(latent_states, observed_compounds, primitives)

        logger.info(f"Initial θ={theta_init}, LL={best_ll:.2f}")

        # Grid search around initial point
        for feature in feature_names:
            for delta in [-0.5, 0.5, -1.0, 1.0]:
                test_theta = best_theta.copy()
                test_theta[feature] = test_theta.get(feature, 0.0) + delta

                constraint = AssemblyConstraint(feature_weights=test_theta)
                latent_states = cached_model.get_latent_states(constraint)
                ll = _compute_observation_ll(latent_states, observed_compounds, primitives)

                if ll > best_ll:
                    best_ll = ll
                    best_theta = test_theta
                    logger.info(f"Improved: θ={best_theta}, LL={best_ll:.2f}")

        result["theta_hat"] = best_theta
        result["cache_stats"] = cached_model.cache_stats

    logger.info(f"Inference complete. Final θ: {result['theta_hat']}")
    return result


def _compute_observation_ll(
    latent_states: dict[int, float],
    observed_compounds: set[str],
    primitives: list[str],
) -> float:
    """Simple observation log-likelihood (placeholder)."""
    import math

    if not latent_states:
        return -math.inf

    # Simple: log probability that any observed compound appears
    log_lik = 0.0
    for compound in observed_compounds:
        # Assume compound present if it's a primitive
        if compound in primitives:
            log_lik += math.log(0.9)  # High probability for primitives
        else:
            log_lik += math.log(0.5)  # Moderate probability for others

    return log_lik


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fit assembly constraints to observed compounds."
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["screen-only", "screen-and-refine", "full-stochastic"],
        default="full-stochastic",
        help="Inference mode (default: full-stochastic)",
    )

    parser.add_argument(
        "--compounds",
        type=str,
        required=True,
        help="Comma-separated list of observed compound identifiers",
    )

    parser.add_argument(
        "--primitives",
        type=str,
        default="A,B",
        help="Comma-separated list of primitive building blocks (default: A,B)",
    )

    parser.add_argument(
        "--features",
        type=str,
        default="reuse_count,depth_change",
        help="Comma-separated list of feature names to consider",
    )

    # Screening options
    parser.add_argument(
        "--screen-grid",
        type=str,
        choices=["auto", "manual"],
        default="auto",
        help="Screening grid mode (default: auto)",
    )

    parser.add_argument(
        "--screen-budget",
        type=int,
        default=100,
        help="Total deterministic evaluations for screening (default: 100)",
    )

    parser.add_argument(
        "--screen-topk",
        type=int,
        default=10,
        help="Number of local refinements in screening (default: 10)",
    )

    parser.add_argument(
        "--screen-refine-radius",
        type=float,
        default=0.5,
        help="Size of local neighborhood around winners (default: 0.5)",
    )

    # Cache options
    parser.add_argument(
        "--trust-region",
        type=str,
        choices=["linf", "mahalanobis"],
        default="linf",
        help="Trust region type (default: linf)",
    )

    parser.add_argument(
        "--trust-radius",
        type=float,
        default=1.0,
        help="Trust region radius (default: 1.0)",
    )

    parser.add_argument(
        "--ess-threshold",
        type=float,
        default=0.3,
        help="ESS threshold ratio (default: 0.3)",
    )

    # Simulation options
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of simulation trajectories (default: 100)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON format)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse inputs
    observed_compounds = set(args.compounds.split(","))
    primitives = args.primitives.split(",")
    feature_names = args.features.split(",")

    mode = InferenceMode(args.mode)
    screen_grid = ScreeningGridMode(args.screen_grid)
    trust_region = TrustRegionType(args.trust_region)

    # Run inference
    result = fit_assembly_constraints(
        observed_compounds=observed_compounds,
        primitives=primitives,
        mode=mode,
        feature_names=feature_names,
        screen_grid=screen_grid,
        screen_budget=args.screen_budget,
        screen_topk=args.screen_topk,
        screen_refine_radius=args.screen_refine_radius,
        trust_region=trust_region,
        trust_radius=args.trust_radius,
        ess_threshold=args.ess_threshold,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results written to {args.output}")
    else:
        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
