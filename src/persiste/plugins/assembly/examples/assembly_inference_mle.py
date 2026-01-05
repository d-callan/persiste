"""
Demo: MLE Inference for Assembly Theory

Phase 1.7: Recover θ from observations with the shared assembly interface.
"""

import sys

sys.path.insert(0, "src")

import numpy as np

from persiste.plugins.assembly.assembly_interface import (
    AssemblyBaselineConfig,
    AssemblyGraphConfig,
    PresenceObservationConfig,
    SimulationSettings,
    fit_presence_observations,
)
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.observation.presence_model import PresenceObservationModel
from persiste.plugins.assembly.states.assembly_state import AssemblyState


def main():
    print("=" * 80)
    print("Interface-Driven Assembly MLE Demo")
    print("=" * 80)

    primitives = ["A", "B", "C"]
    graph = AssemblyGraph(primitives, max_depth=4, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    obs_model = PresenceObservationModel(detection_prob=0.9, false_positive_prob=0.01)
    initial_state = AssemblyState.from_parts([primitives[0]], depth=0)

    print("\nSynthetic setup:")
    print(f"  Graph: {graph}")
    print(f"  Baseline: {baseline}")
    print(f"  Observation model: {obs_model}")

    theta_true = {"reuse_count": 1.0, "depth_change": -0.3}
    print("\nTrue parameters:")
    for k, v in theta_true.items():
        print(f"  {k:20s} = {v:.2f}")

    constraint_true = AssemblyConstraint(feature_weights=theta_true)
    simulator = GillespieSimulator(graph, baseline, constraint_true, rng=np.random.default_rng(42))
    latent_states = simulator.sample_final_states(
        initial_state,
        n_samples=120,
        t_max=60.0,
        burn_in=30.0,
    )

    observed_compounds: set[str] = set()
    rng = np.random.default_rng(99)
    for state, prob in latent_states.items():
        for part in state.get_parts_list():
            if rng.random() < obs_model.detection_prob * prob:
                observed_compounds.add(part)

    print(f"\nObserved compounds: {sorted(observed_compounds)}")

    baseline_cfg = AssemblyBaselineConfig(
        kappa=baseline.kappa,
        join_exponent=baseline.join_exponent,
        split_exponent=baseline.split_exponent,
        decay_rate=baseline.decay_rate,
    )
    graph_cfg = AssemblyGraphConfig(
        max_depth=graph.max_depth,
        min_rate_threshold=graph.min_rate,
    )
    obs_cfg = PresenceObservationConfig(
        detection_prob=obs_model.detection_prob,
        false_positive_prob=obs_model.false_positive_prob,
    )
    sim_cfg = SimulationSettings(n_samples=35, t_max=55.0, burn_in=25.0)

    print("\nRunning fit_presence_observations via ConstraintInference...")
    result = fit_presence_observations(
        observed_compounds=observed_compounds,
        feature_names=list(theta_true.keys()),
        primitives=primitives,
        baseline_config=baseline_cfg,
        graph_config=graph_cfg,
        observation_config=obs_cfg,
        simulation=sim_cfg,
        initial_state_parts=[primitives[0]],
        rng_seed=123,
        inference_kwargs={"options": {"maxiter": 60}},
    )

    print(f"\n{'Parameter':<20s}{'True':>10s}{'Estimate':>12s}{'Abs Error':>12s}")
    print("-" * 54)
    for name in theta_true:
        truth = theta_true[name]
        estimate = result.parameters.get(name, 0.0)
        error = abs(estimate - truth)
        print(f"{name:<20s}{truth:>10.3f}{estimate:>12.3f}{error:>12.3f}")

    print("\nDiagnostics")
    print("-" * 20)
    print(f"Log-likelihood: {result.log_likelihood:.4f}")
    print(f"AIC: {result.aic:.3f}")
    print(f"Method: {result.method}")
    print(f"Metadata: {result.metadata}")


if __name__ == "__main__":
    main()
