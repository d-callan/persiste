"""
Demo: Lazy Assembly Graph

Shows how the graph generates neighbors on demand and prunes low-probability paths.
"""

import sys

sys.path.insert(0, 'src')

from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.states.assembly_state import AssemblyState


def main():
    print("=" * 70)
    print("Lazy Assembly Graph Demo")
    print("=" * 70)

    # 1. Create graph
    print("\n1. Creating Lazy Graph")
    print("-" * 70)

    primitives = ['A', 'B', 'C']
    graph = AssemblyGraph(
        primitives=primitives,
        max_depth=4,
        min_rate_threshold=1e-3,
    )

    print(f"  {graph}")
    print(f"  Primitives: {[str(s) for s in graph.get_primitive_states()]}")

    # 2. Create baseline and constraint
    print("\n2. Creating Models")
    print("-" * 70)

    baseline = AssemblyBaseline(
        kappa=1.0,
        join_exponent=-0.5,
        split_exponent=0.3,
    )

    constraint = AssemblyConstraint(
        motif_bonuses={'stable': 1.5},
        reuse_bonus=0.8,
        depth_penalty=-0.2,
    )

    print(f"  {baseline}")
    print(f"  {constraint}")

    # 3. Explore neighbors from primitive
    print("\n3. Exploring Neighbors (Lazy Generation)")
    print("-" * 70)

    state_A = AssemblyState.from_parts(['A'], depth=0)
    print(f"\n  Starting from: {state_A}")

    neighbors = graph.get_neighbors(state_A, baseline, constraint)
    print(f"  Found {len(neighbors)} neighbors:")

    for neighbor, rate, trans_type in neighbors[:5]:  # Show first 5
        print(f"    {trans_type.value:10s} → {neighbor} (λ={rate:.4f})")

    # 4. Explore from assembled state
    print("\n4. Exploring from Assembled State")
    print("-" * 70)

    state_AB = AssemblyState.from_parts(['A', 'B'], depth=1)
    print(f"\n  Starting from: {state_AB}")

    neighbors_AB = graph.get_neighbors(state_AB, baseline, constraint)
    print(f"  Found {len(neighbors_AB)} neighbors:")

    for neighbor, rate, trans_type in neighbors_AB:
        print(f"    {trans_type.value:10s} → {neighbor} (λ={rate:.4f})")

    # 5. Count reachable states
    print("\n5. Counting Reachable States (BFS)")
    print("-" * 70)

    for start in [state_A, state_AB]:
        count = graph.count_reachable_states(start, baseline, constraint, max_states=100)
        print(f"  From {start}: {count} reachable states")

    # 6. Demonstrate caching
    print("\n6. Demonstrating Caching")
    print("-" * 70)

    print(f"  Before: {graph}")

    # Query same state again
    neighbors_again = graph.get_neighbors(state_A, baseline, constraint)
    print(f"  After re-query: {graph}")
    print(f"  Same result: {neighbors == neighbors_again}")

    # 7. Show pruning effect
    print("\n7. Pruning Effect")
    print("-" * 70)

    # Create graph with different thresholds
    graph_strict = AssemblyGraph(primitives, max_depth=4, min_rate_threshold=0.1)
    graph_loose = AssemblyGraph(primitives, max_depth=4, min_rate_threshold=1e-6)

    neighbors_strict = graph_strict.get_neighbors(state_AB, baseline, constraint)
    neighbors_loose = graph_loose.get_neighbors(state_AB, baseline, constraint)

    print(f"  Strict threshold (0.1):   {len(neighbors_strict)} neighbors")
    print(f"  Loose threshold (1e-6):   {len(neighbors_loose)} neighbors")
    print(f"  Pruned: {len(neighbors_loose) - len(neighbors_strict)} low-rate transitions")

    print("\n" + "=" * 70)
    print("Graph Demo Complete!")
    print("=" * 70)

    print("\nKey Takeaways:")
    print("  ✓ Graph generates neighbors on demand (lazy)")
    print("  ✓ Pruning removes low-probability transitions")
    print("  ✓ Caching avoids recomputation")
    print("  ✓ Scales sublinearly in state space")
    print("  ✓ Max depth prevents explosion")
    print("\nNext: Implement observation models and inference!")


if __name__ == '__main__':
    main()
