"""
CLI entry points for assembly constraint inference.

Modes:
- safety-only: Run only safety checks, no inference (pre-flight mode)
- screen-only: Fast deterministic screening, no stochastic refinement
- screen-and-refine: Screen candidates, refine winners stochastically
- full-stochastic: No shortcuts, run full Gillespie-based inference

Safety checks (Tier 1) run automatically unless --skip-safety-checks is set.
"""

import argparse
import json
import logging
import sys
from enum import Enum
from typing import Any

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.diagnostics.artifacts import CachedPathData, InferenceArtifacts
from persiste.plugins.assembly.likelihood import compute_observation_ll
from persiste.plugins.assembly.observation.cached_observation import (
    CacheConfig,
    CachedAssemblyObservationModel,
    SimulationSettings,
)
from persiste.plugins.assembly.safety import run_safety_checks
from persiste.plugins.assembly.screening.screening import (
    AdaptiveScreeningGrid,
    ScreeningResult,
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

    SAFETY_ONLY = "safety-only"
    """Run only safety checks, no inference."""

    SCREEN_ONLY = "screen-only"
    """Fast, deterministic only."""

    SCREEN_AND_REFINE = "screen-and-refine"
    """Screen then stochastic (optional deterministic stage)."""

    FULL_STOCHASTIC = "full-stochastic"
    """No shortcuts, stochastic inference only."""


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
    # Safety options
    skip_safety_checks: bool = False,
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
    # Observation statistics (optional enriched inputs)
    observation_records: list[dict[str, Any]] | None = None,
    # Screening toggle
    use_screening: bool = False,
) -> dict:
    """
    Fit assembly constraints with configurable inference mode.

    Args:
        observed_compounds: Set of observed compound identifiers
        primitives: List of primitive building blocks
        mode: Inference mode (safety-only, screen-only, screen-and-refine, full-stochastic)
        feature_names: Feature names to consider (default: reuse_count, depth_change)
        skip_safety_checks: Skip Tier 1 safety checks (not recommended)
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
        observation_records: Enriched observation records (frequencies, etc.)

    Returns:
        Dict with keys:
        - mode: Inference mode used
        - theta_hat: Estimated feature weights (None if safety-only)
        - screening_results: List of screening results (if applicable)
        - cache_stats: Cache statistics (if applicable)
        - safety: SafetyReport (unless skip_safety_checks=True)
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
        "safety": None,
        "deterministic_delta_ll": None,
        "deterministic_ll": None,
        "deterministic_null_ll": None,
        "deterministic_normalized_delta_ll": None,
        "deterministic_vs_stochastic_delta_gap": None,
    }

    # Safety-only mode: run checks and return early
    if mode == InferenceMode.SAFETY_ONLY:
        logger.info("Running safety-only mode (pre-flight checks)")

        # Run minimal screening for identifiability check
        ss_model = SteadyStateAssemblyModel(
            primitives=primitives,
            baseline=baseline,
            config=SteadyStateConfig(max_depth=max_depth),
        )

        grid = AdaptiveScreeningGrid(
            feature_names=feature_names,
            budget=min(screen_budget, 20),  # Reduced budget for safety-only
            top_k=3,
            refine_radius=screen_refine_radius,
        )

        screening_results = adaptive_screen(
            model=ss_model,
            observed_compounds=observed_compounds,
            initial_state=initial_state,
            grid=grid,
            threshold=screening_threshold,
            observation_records=observation_records,
            max_depth=max_depth,
        )

        # Run safety checks
        safety_report = run_safety_checks(
            observed_compounds=observed_compounds,
            primitives=primitives,
            baseline=baseline,
            initial_state=initial_state,
            theta_hat={},
            screening_results=screening_results,
            cache_stats={"initialized": False, "n_paths": 0},
            n_baseline_samples=n_samples,
            max_depth=max_depth,
            seed=seed,
        )

        result["safety"] = safety_report.to_dict()
        logger.info(f"Safety check complete. Status: {safety_report.overall_status}")
        return result

    # Determine whether deterministic screening should run
    should_screen = use_screening or mode in (
        InferenceMode.SCREEN_ONLY,
        InferenceMode.SCREEN_AND_REFINE,
    )

    # Phase 1: Screening (if enabled)
    if should_screen and mode in (
        InferenceMode.SCREEN_ONLY,
        InferenceMode.SCREEN_AND_REFINE,
    ):
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
            observation_records=observation_records,
            max_depth=max_depth,
        )

        result["screening_results"] = [
            {
                "theta": r.theta,
                "delta_ll": r.delta_ll,
                "normalized_delta_ll": r.normalized_delta_ll,
                "passed": r.passed,
                "rank": r.rank,
                "absolute_ll": r.absolute_ll,
                "null_ll": r.null_ll,
            }
            for r in screening_results
        ]
        if screening_results:
            best = screening_results[0]
            result["deterministic_delta_ll"] = best.delta_ll
            result["deterministic_ll"] = best.absolute_ll
            result["deterministic_null_ll"] = best.null_ll
            result["deterministic_normalized_delta_ll"] = best.normalized_delta_ll

        # Get best hypothesis
        if screening_results and screening_results[0].passed:
            result["theta_hat"] = screening_results[0].theta
        else:
            result["theta_hat"] = {}

        if mode == InferenceMode.SCREEN_ONLY:
            logger.info(f"Screening complete. Best θ: {result['theta_hat']}")
            return result
    else:
        result["screening_results"] = []
        if mode == InferenceMode.SCREEN_ONLY:
            # Already raised earlier, safeguard
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

        # Get latent states at null model for ΔLL baseline
        null_constraint = AssemblyConstraint(feature_weights={})
        null_latent_states = cached_model.get_latent_states(null_constraint)

        null_ll = compute_observation_ll(
            null_latent_states,
            observed_compounds,
            primitives,
            observation_records=observation_records,
            max_depth=max_depth,
            null_latent_states=None,  # Absolute LL for null
            ess_ratio=1.0,
        )

        # Get latent states at initial theta
        constraint = AssemblyConstraint(feature_weights=theta_init)
        latent_states = cached_model.get_latent_states(constraint)
        ess_ratio = cached_model.current_ess_ratio

        # Simple refinement: evaluate a few nearby points
        # (Full optimization would use scipy.optimize or similar)
        best_theta = theta_init
        best_ll = compute_observation_ll(
            latent_states,
            observed_compounds,
            primitives,
            observation_records=observation_records,
            max_depth=max_depth,
            null_latent_states=None,
            ess_ratio=ess_ratio,
        )
        best_delta_ll = best_ll - null_ll

        logger.info(f"Initial θ={theta_init}, ΔLL={best_delta_ll:.2f}, null_LL={null_ll:.2f}")

        # Iterative Coordinate Descent (Hill Climbing)
        # We loop until convergence or max iterations is reached.
        # This allows the optimizer to climb from 0.0 to higher values (e.g. 2.5) step-by-step.
        max_iterations = 10
        iteration = 0
        converged = False

        logger.info(f"Starting stochastic hill climbing (max_iter={max_iterations})...")

        while not converged and iteration < max_iterations:
            iteration += 1
            improved_in_pass = False

            # Grid search around current best point
            # Regularization: skip large steps if current ΔLL is small (avoid overfitting noise)
            regularization_threshold = 0.5

            for feature in feature_names:
                # Check neighbors in discrete steps
                # Added finer steps (+/- 0.1) to catch subtle parameters like depth_change
                for delta in [-0.1, 0.1, -0.5, 0.5, -1.0, 1.0]:
                    test_theta = best_theta.copy()
                    test_theta[feature] = test_theta.get(feature, 0.0) + delta

                    # Skip if this would push θ too far from zero when ΔLL is already small
                    if best_delta_ll < regularization_threshold:
                        theta_norm = sum(abs(v) for v in test_theta.values())
                        if theta_norm > 10.0:
                            continue

                    constraint = AssemblyConstraint(feature_weights=test_theta)
                    latent_states = cached_model.get_latent_states(constraint)
                    ess_ratio = cached_model.current_ess_ratio

                    test_ll = compute_observation_ll(
                        latent_states,
                        observed_compounds,
                        primitives,
                        observation_records=observation_records,
                        max_depth=max_depth,
                        null_latent_states=None,
                        ess_ratio=ess_ratio,
                    )

                    delta_ll = test_ll - null_ll

                    # Require a small minimum improvement to avoid numerical jitter loops
                    if delta_ll > best_delta_ll + 0.01:
                        best_delta_ll = delta_ll
                        best_theta = test_theta
                        best_ll = test_ll
                        improved_in_pass = True
                        msg = f"Improved (iter {iteration}): θ={best_theta}"
                        logger.info(msg)
                    elif logger.isEnabledFor(logging.DEBUG):
                        msg = (
                            f"Rejected (iter {iteration}): θ={test_theta}, "
                            f"ΔLL={delta_ll:.2f} <= best={best_delta_ll:.2f}"
                        )
                        logger.debug(msg)

            if not improved_in_pass:
                converged = True
                logger.info(f"Converged after {iteration} iterations.")
            elif iteration >= max_iterations:
                logger.info(f"Reached max iterations ({max_iterations}). Stopping.")

        result["theta_hat"] = best_theta
        result["stochastic_ll"] = best_ll
        result["stochastic_null_ll"] = null_ll
        result["stochastic_delta_ll"] = best_delta_ll
        result["cache_stats"] = cached_model.cache_stats

        # Populate deep diagnostic artifacts for Tier 2 recipes
        cache_id = f"cache_{seed}_{iteration}"
        result["artifacts"] = InferenceArtifacts(
            theta_hat=best_theta,
            log_likelihood=best_ll,
            cache_id=cache_id,
            baseline_config={
                "kappa": kappa,
                "join_exponent": join_exponent,
                "split_exponent": split_exponent,
                "decay_rate": decay_rate
            },
            graph_config={
                "primitives": primitives,
                "max_depth": max_depth
            }
        )
        if cached_model._cache:
            result["cache"] = CachedPathData(
                feature_counts=cached_model._cache.feature_counts,
                final_state_ids=cached_model._cache.final_state_ids,
                theta_ref=cached_model._cache.theta_ref
            )

        if result.get("deterministic_delta_ll") is not None:
            result["deterministic_vs_stochastic_delta_gap"] = (
                best_delta_ll - result["deterministic_delta_ll"]
            )
        msg = f"Stochastic refinement: best_θ={best_theta}, ΔLL={best_delta_ll:.2f}"
        logger.info(msg)

    # Run Tier 1 safety checks (unless skipped)
    if not skip_safety_checks:
        logger.info("Running Tier 1 safety checks")

        # Convert screening results to ScreeningResult objects if needed
        screening_result_objs = []
        for r in result.get("screening_results", []):
            if isinstance(r, dict):
                screening_result_objs.append(
                    ScreeningResult(
                        theta=r["theta"],
                        delta_ll=r["delta_ll"],
                        normalized_delta_ll=r["normalized_delta_ll"],
                        passed=r["passed"],
                        rank=r["rank"],
                        absolute_ll=r.get("absolute_ll"),
                        null_ll=r.get("null_ll"),
                    )
                )
            else:
                screening_result_objs.append(r)

        safety_report = run_safety_checks(
            observed_compounds=observed_compounds,
            primitives=primitives,
            baseline=baseline,
            initial_state=initial_state,
            theta_hat=result["theta_hat"],
            screening_results=screening_result_objs,
            cache_stats=result.get("cache_stats") or {"n_paths": n_samples},
            n_baseline_samples=n_samples,
            max_depth=max_depth,
            seed=seed,
        )

        result["safety"] = safety_report.to_dict()

        if not safety_report.overall_safe:
            msg = (
                f"Safety issues: {safety_report.overall_status}"
                f"ΔLL threshold adjusted to {safety_report.adjusted_delta_ll_threshold:.1f}"
            )
            logger.warning(msg)

    logger.info(f"Inference complete. Final θ: {result['theta_hat']}")
    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fit assembly constraints to observed compounds."
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["safety-only", "screen-only", "screen-and-refine", "full-stochastic"],
        default="full-stochastic",
        help="Inference mode (default: full-stochastic)",
    )

    parser.add_argument(
        "--skip-safety-checks",
        action="store_true",
        help="Skip Tier 1 safety checks (not recommended)",
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
        skip_safety_checks=args.skip_safety_checks,
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
