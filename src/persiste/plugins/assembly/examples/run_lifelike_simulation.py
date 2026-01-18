import logging

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.cli import InferenceMode, fit_assembly_constraints
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.observation.cached_observation import (
    CachedAssemblyObservationModel,
    CacheConfig,
    SimulationSettings,
)
from persiste.plugins.assembly.states.assembly_state import AssemblyState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_lifelike_simulation():
    """
    Simulate "lifelike" assembly data with deep recursion and selection pressure,
    then attempt to recover the parameters using the zero-order likelihood model.
    """
    logger.info("Setting up Life-Like Simulation...")

    # 1. Setup Parameters
    # "Lifelike" means:
    # - Deep assembly (depth 15-20)
    # - Strong selection for reuse (autocatalysis/stability proxy)
    # - Asymmetric probability mass (concentrated on specific high-reuse paths)

    seed = 42
    max_depth = 15
    primitives = ["A", "B", "C", "D"]  # More primitives to make the space vast

    # True parameters: Strong reuse bonus
    theta_true = {
        "reuse_count": 2.5,  # Strong selection
        "depth_change": -0.1  # Slight cost to getting deeper (resource constraint)
    }

    logger.info(f"Parameters: Depth={max_depth}, Primitives={primitives}")
    logger.info(f"True Theta: {theta_true}")

    # 2. Simulate Data (The Generative Process)
    # We use the cached model directly to generate the "Truth"

    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5)
    initial_state = AssemblyState.from_parts(primitives[:2], depth=0)  # Start with A, B

    # Simulation settings for data generation
    # We need enough samples to get a representative set of "survivors"
    sim_settings = SimulationSettings(
        n_samples=1000,
        t_max=100.0,
        burn_in=20.0,
        max_depth=max_depth,
    )

    cache_config = CacheConfig(
        trust_radius=10.0,  # Large radius to avoid constant resimulation during inference check
        ess_threshold=0.0
    )

    model = CachedAssemblyObservationModel(
        primitives=primitives,
        baseline=baseline,
        initial_state=initial_state,
        simulation=sim_settings,
        cache_config=cache_config,
        rng_seed=seed
    )

    # Generate the latent states under theta_true
    logger.info("Simulating ground truth trajectories...")
    constraint = AssemblyConstraint(feature_weights=theta_true)
    latent_states = model.get_latent_states(constraint)

    # 3. Harvest "Observed" Data
    # In a life-like system, we only see the high-abundance stable states.
    # We'll take the top K most probable states from the simulation.

    sorted_states = sorted(latent_states.items(), key=lambda x: x[1], reverse=True)

    # Filter for non-trivial states (depth > 5) to ensure we are looking at "assembled" things
    # Note: State IDs are opaque here, but the model handles mapping internally.
    # For simulation purposes, we rely on the fact that high probability states
    # under this theta MUST be high-reuse/deep due to the constraints.

    top_states = sorted_states[:20]
    total_mass = sum(p for _, p in top_states)

    logger.info(f"Top 20 states capture {total_mass:.2%} of probability mass.")

    # Construct observed set
    observed_compounds = {f"state_{sid}" for sid, _ in top_states}
    logger.info(f"Observed {len(observed_compounds)} compounds (Simulated 'Life').")

    # 4. Run Inference
    # Can we recover the reuse_count parameter?

    logger.info("Running Inference to recover parameters...")

    # We use a smaller max_depth for inference to save time,
    # or match the simulation depth to be fair. Let's match it.

    result = fit_assembly_constraints(
        observed_compounds=observed_compounds,
        primitives=primitives,
        mode=InferenceMode.FULL_STOCHASTIC,
        feature_names=["reuse_count", "depth_change"],
        n_samples=1000,     # Increased samples for better inference
        t_max=100.0,
        burn_in=20.0,
        max_depth=max_depth,
        seed=seed + 1      # Different seed for inference!
    )

    # 5. Report Results
    theta_hat = result.get("theta_hat", {})
    delta_ll = result.get("stochastic_delta_ll", 0.0)

    logger.info("=" * 40)
    logger.info("INFERENCE RESULTS")
    logger.info("=" * 40)
    logger.info(f"True Theta: {theta_true}")
    logger.info(f"Inferred Theta: {theta_hat}")
    logger.info(f"Delta LL (Evidence for Selection): {delta_ll:.4f}")

    # Validation checks
    reuse_error = abs(theta_hat.get("reuse_count", 0.0) - theta_true["reuse_count"])
    if reuse_error < 1.0:
        logger.info("✅ SUCCESS: Recovered strong reuse signal!")
    else:
        logger.warning(
            f"⚠️  WARNING: Inferred reuse {theta_hat.get('reuse_count')} "
            f"differs from true {theta_true['reuse_count']}"
        )

    if delta_ll > 10.0:
        logger.info("✅ SUCCESS: Strong statistical evidence for selection.")
    else:
        logger.warning(f"⚠️  WARNING: Weak evidence (Delta LL = {delta_ll})")


if __name__ == "__main__":
    run_lifelike_simulation()
