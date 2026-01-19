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

import numpy as np
from scipy.optimize import minimize

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
from persiste.plugins.assembly.states.resolver import StateIDResolver

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
        # Pre-build graph if not already available for mapping
        from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
        graph = AssemblyGraph(primitives, max_depth=max_depth)

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
            graph=graph,
        )

        # 1. Resolve state mapping.
        # We use a StateIDResolver to bridge human-readable compound labels to Rust IDs.
        resolver = StateIDResolver(primitives)
        compound_to_state = {}
        observed_ids = set()
        for compound in observed_compounds:
            try:
                sid = resolver.resolve_string(compound)
                compound_to_state[compound] = sid
                observed_ids.add(sid)
            except ValueError:
                logger.warning(f"Could not resolve compound: {compound}")
                continue

        # 2. Get latent states at null model for ΔLL baseline
        null_constraint = AssemblyConstraint(feature_weights={})
        null_latent_states = cached_model.get_latent_states(null_constraint)
        null_ess_ratio = cached_model.current_ess_ratio

        # IMPORTANT: Use direct compute_observation_ll for absolute LLs
        null_ll = compute_observation_ll(
            null_latent_states,
            observed_ids,
            primitives,
            observation_records=observation_records,
            max_depth=max_depth,
            null_latent_states=None,
            ess_ratio=null_ess_ratio,
            compound_to_state=compound_to_state,
        )

        # Start from screening result if available, otherwise start at null
        theta_init = result["theta_hat"] if result["theta_hat"] else {}
        
        # 3. Get latent states at initial theta
        constraint = AssemblyConstraint(feature_weights=theta_init)
        latent_states = cached_model.get_latent_states(constraint)
        init_ess_ratio = cached_model.current_ess_ratio

        # 4. Continuous Optimization using scipy.optimize
        def objective(x):
            # Map array back to dict
            current_theta = {name: val for name, val in zip(feature_names, x)}

            # Get latent states (handles caching/importance sampling internally)
            constraint = AssemblyConstraint(feature_weights=current_theta)
            current_latent_states = cached_model.get_latent_states(constraint)
            current_ess_ratio = cached_model.current_ess_ratio

            # Compute ΔLL using ratio mode
            delta_ll = compute_observation_ll(
                current_latent_states,
                observed_ids,
                primitives,
                observation_records=observation_records,
                max_depth=max_depth,
                null_latent_states=null_latent_states,
                ess_ratio=current_ess_ratio,
                compound_to_state=compound_to_state,
            )

            # Return negative ΔLL for minimization
            return -delta_ll

        # Calculate initial ΔLL for reporting
        initial_delta_ll = compute_observation_ll(
            latent_states,
            observed_ids,
            primitives,
            observation_records=observation_records,
            max_depth=max_depth,
            null_latent_states=null_latent_states,
            ess_ratio=init_ess_ratio,
            compound_to_state=compound_to_state,
        )

        # Prepare initial guess array
        x0 = np.array([theta_init.get(name, 0.0) for name in feature_names])

        logger.info(f"Initial θ={theta_init}, ΔLL={initial_delta_ll:.2f}, null_LL={null_ll:.2f}")
        logger.info("Starting continuous optimization (L-BFGS-B)...")

        # Run optimization
        opt_result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 20, "ftol": 1e-3, "disp": False}
        )

        # Update best results from optimization
        best_x = opt_result.x
        best_theta_dict = {name: val for name, val in zip(feature_names, best_x)}
        best_delta_ll = -opt_result.fun
        best_ll = null_ll + best_delta_ll

        logger.info(f"Optimization converged: {opt_result.success}, iterations: {opt_result.nit}")
        logger.info(f"Final θ={best_theta_dict}, ΔLL={best_delta_ll:.2f}")

        result["theta_hat"] = best_theta_dict
        result["stochastic_ll"] = best_ll
        result["stochastic_null_ll"] = null_ll
        result["stochastic_delta_ll"] = best_delta_ll
        result["cache_stats"] = cached_model.cache_stats
        result["optimization_status"] = {
            "success": bool(opt_result.success),
            "message": str(opt_result.message),
            "nfev": int(opt_result.nfev),
            "nit": int(opt_result.nit)
        }

        # Populate deep diagnostic artifacts for Tier 2 recipes
        cache_id = f"cache_{seed}_{opt_result.nit}"
        result["artifacts"] = InferenceArtifacts(
            theta_hat=best_theta_dict,
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
        msg = f"Stochastic refinement: best_θ={best_theta_dict}, ΔLL={best_delta_ll:.2f}"
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
