"""
Assembly interface quickstart demo.

Demonstrates generating synthetic observations and fitting constraints using
the high-level `fit_assembly_counts` helper instead of bespoke plumbing.
"""

import sys

sys.path.insert(0, "src")

import numpy as np

from persiste.plugins.assembly.assembly_interface import (
    AssemblyBaselineConfig,
    AssemblyCountsConfig,
    AssemblyGraphConfig,
    SimulationSettings,
    fit_assembly_counts,
)
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.observation.counts_model import AssemblyCountsModel
from persiste.plugins.assembly.states.assembly_state import AssemblyState


def main():
    print("=" * 80)
    print("Assembly Interface Demo")
    print("=" * 80)

    primitives = ["A", "B", "C"]
    graph = AssemblyGraph(primitives, max_depth=5, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)
    obs_model = AssemblyCountsModel(detection_efficiency=0.9, background_noise=0.01)
    initial_state = AssemblyState.from_parts([primitives[0]], depth=0)

    theta_true = {"reuse_count": 1.1, "depth_change": -0.25}
    constraint = AssemblyConstraint(feature_weights=theta_true)

    simulator = GillespieSimulator(graph, baseline, constraint, rng=np.random.default_rng(101))
    latent_states = simulator.sample_final_states(
        initial_state,
        n_samples=150,
        t_max=70.0,
    )

    from collections import defaultdict
    observed_counts: dict[str, float] = defaultdict(float)
    rng = np.random.default_rng(5)
    for state, prob in latent_states.items():
        for part in state.get_parts_list():
            if rng.random() < obs_model.detection_efficiency * prob:
                observed_counts[part] += 1

    for spur in ["D", "E"]:
        if spur not in observed_counts and rng.random() < obs_model.background_noise:
            observed_counts[spur] = 1

    print("\nObserved counts:", dict(observed_counts))

    result = fit_assembly_counts(
        observed_counts=observed_counts,
        feature_names=list(theta_true.keys()),
        primitives=primitives,
        baseline_config=AssemblyBaselineConfig(
            kappa=baseline.kappa,
            join_exponent=baseline.join_exponent,
            split_exponent=baseline.split_exponent,
        ),
        graph_config=AssemblyGraphConfig(
            max_depth=graph.max_depth,
            min_rate_threshold=graph.min_rate,
        ),
        observation_config=AssemblyCountsConfig(
            detection_efficiency=obs_model.detection_efficiency,
            background_noise=obs_model.background_noise,
        ),
        simulation=SimulationSettings(n_samples=25, t_max=45.0, burn_in=20.0),
        rng_seed=42,
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
    print(f"Log-likelihood: {result.log_likelihood:.3f}")
    print(f"AIC: {result.aic:.3f}")
    print(f"Method: {result.method}")
    print(f"Metadata: {result.metadata}")


if __name__ == "__main__":
    main()
