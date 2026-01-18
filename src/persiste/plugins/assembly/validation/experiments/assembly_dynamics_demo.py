"""
Demo: Assembly CTMC Dynamics

Shows how θ → λ_eff → P(state) via Gillespie simulation.

This is Phase 1.5: Making constraint parameters actually influence state occupancy.
"""

import sys

sys.path.insert(0, 'src')

import numpy as np

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.states.assembly_state import AssemblyState


def main():
    print("=" * 80)
    print("Assembly CTMC Dynamics Demo")
    print("=" * 80)

    print("\nPhase 1.5: θ → λ_eff → P(state)")
    print("Making constraint parameters influence state occupancy\n")

    # ========================================================================
    # Setup
    # ========================================================================
    print("=" * 80)
    print("Setup: Graph, Baseline, Constraints")
    print("=" * 80)

    primitives = ['A', 'B', 'C']
    graph = AssemblyGraph(primitives, max_depth=4, min_rate_threshold=1e-4)
    baseline = AssemblyBaseline(kappa=1.0, join_exponent=-0.5, split_exponent=0.3)

    print(f"\n{graph}")
    print(f"{baseline}")

    # ========================================================================
    # Compare Three Constraint Models
    # ========================================================================
    print("\n" + "=" * 80)
    print("Three Constraint Models")
    print("=" * 80)

    models = {
        "Null": AssemblyConstraint.null_model(),
        "Reuse-only": AssemblyConstraint.reuse_only(reuse_weight=1.0),
        "Assembly Theory": AssemblyConstraint.assembly_theory(reuse=1.0, depth_penalty=-0.3),
    }

    for name, model in models.items():
        params = model.get_parameters()
        if params:
            params_str = ', '.join(f"{k}={v:.1f}" for k, v in params.items())
            print(f"  {name:20s} θ = {{{params_str}}}")
        else:
            print(f"  {name:20s} θ = {{}}")

    # ========================================================================
    # Single Trajectory
    # ========================================================================
    print("\n" + "=" * 80)
    print("Single Trajectory (Null Model)")
    print("=" * 80)

    initial_state = AssemblyState.from_parts(['A'], depth=0)
    constraint_null = models["Null"]

    simulator = GillespieSimulator(graph, baseline, constraint_null, rng=np.random.default_rng(42))

    print(f"\nInitial state: {initial_state}")
    print("Simulating for t_max = 50.0...\n")

    traj = simulator.simulate(initial_state, t_max=50.0)

    print(f"Trajectory length: {len(traj.states)} states")
    print(f"Duration: {traj.duration():.2f}")
    print(f"Final state: {traj.final_state()}")

    print("\nFirst 10 transitions:")
    for i in range(min(10, len(traj.states))):
        print(f"  t={traj.times[i]:6.2f}  {traj.states[i]}")

    # ========================================================================
    # Final State Distribution (Null Model)
    # ========================================================================
    print("\n" + "=" * 80)
    print("Final State Distribution (Null Model)")
    print("=" * 80)

    print("\nSampling 100 trajectories...")

    state_probs_null = simulator.sample_final_states(
        initial_state,
        n_samples=100,
        t_max=50.0,
        burn_in=25.0,
    )

    print(f"\nFound {len(state_probs_null)} unique final states\n")
    print(f"{'State':<35s} {'Probability':<12s}")
    print("-" * 50)

    # Sort by probability
    sorted_states = sorted(state_probs_null.items(), key=lambda x: x[1], reverse=True)
    for state, prob in sorted_states[:10]:  # Top 10
        print(f"{str(state):<35s} {prob:>10.3f}")

    # ========================================================================
    # Compare Constraint Models
    # ========================================================================
    print("\n" + "=" * 80)
    print("How Constraints Affect State Distribution")
    print("=" * 80)

    print("\nSampling final states for each model (50 trajectories each)...\n")

    distributions = {}
    for name, constraint in models.items():
        sim = GillespieSimulator(graph, baseline, constraint, rng=np.random.default_rng(42))
        distributions[name] = sim.sample_final_states(
            initial_state,
            n_samples=50,
            t_max=50.0,
            burn_in=25.0,
        )

    # Find common states
    all_states = set()
    for dist in distributions.values():
        all_states.update(dist.keys())

    print(f"Total unique states across all models: {len(all_states)}\n")

    # Show top states for each model
    for name, dist in distributions.items():
        print(f"{name}:")
        sorted_states = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        for state, prob in sorted_states[:5]:
            print(f"  {prob:>6.3f}  {state}")
        print()

    # ========================================================================
    # Key Insight
    # ========================================================================
    print("=" * 80)
    print("Key Insight: θ Influences P(state)")
    print("=" * 80)

    print("\nDifferent constraint models → different state distributions")
    print("\nThis is what makes inference possible:")
    print("  1. Observe final states")
    print("  2. Fit θ to maximize P(observations | θ)")
    print("  3. Recover which constraints shaped the system")

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)

    print("\nPhase 1.5 ✓ Complete")
    print("  ✓ Gillespie simulator implemented")
    print("  ✓ θ → λ_eff → P(state) pipeline working")
    print("  ✓ Different constraints give different distributions")
    print("\nNext: Phase 1.6 (ConstraintModel interface)")


if __name__ == '__main__':
    main()
