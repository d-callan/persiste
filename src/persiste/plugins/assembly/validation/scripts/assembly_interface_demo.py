"""
Demo: ConstraintModel Interface

Shows Phase 1.6: pack/unpack/get_constrained_baseline methods.

This is pure plumbing - makes assembly plugin look like PERSISTE constraint model.
"""

import sys

sys.path.insert(0, 'src')

from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint


def main():
    print("=" * 80)
    print("ConstraintModel Interface Demo (Phase 1.6)")
    print("=" * 80)

    print("\nPure plumbing: θ ↔ vector for scipy.optimize")

    # ========================================================================
    # Create Constraint Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("Create Constraint Model")
    print("=" * 80)

    constraint = AssemblyConstraint({
        'reuse_count': 1.2,
        'depth_change': -0.4,
        'symmetry_score': 0.1,
    })

    print(f"\n{constraint}")
    print(f"\nParameters: {constraint.get_parameters()}")

    # ========================================================================
    # Pack to Vector
    # ========================================================================
    print("\n" + "=" * 80)
    print("Pack: θ → vector")
    print("=" * 80)

    theta_vec = constraint.pack()

    print("\nFeature weights (dict):")
    for k, v in constraint.get_parameters().items():
        print(f"  {k:20s} = {v:.2f}")

    print("\nPacked vector:")
    print(f"  {theta_vec}")
    print(f"  Shape: {theta_vec.shape}")

    # ========================================================================
    # Unpack from Vector
    # ========================================================================
    print("\n" + "=" * 80)
    print("Unpack: vector → θ")
    print("=" * 80)

    # Modify vector
    theta_vec_modified = theta_vec * 1.5

    print("\nModified vector:")
    print(f"  {theta_vec_modified}")

    theta_dict = constraint.unpack(theta_vec_modified)

    print("\nUnpacked parameters:")
    for k, v in theta_dict.items():
        print(f"  {k:20s} = {v:.2f}")

    # ========================================================================
    # Count Parameters
    # ========================================================================
    print("\n" + "=" * 80)
    print("Count Parameters (for AIC/BIC)")
    print("=" * 80)

    n_params = constraint.num_free_parameters()
    print(f"\nNumber of free parameters: {n_params}")

    # ========================================================================
    # Initial Parameters
    # ========================================================================
    print("\n" + "=" * 80)
    print("Initial Parameters (for optimization)")
    print("=" * 80)

    theta0 = constraint.initial_parameters()

    print("\nInitial vector (neutral):")
    print(f"  {theta0}")
    print("  All zeros (neutral starting point)")

    # ========================================================================
    # Round-Trip Test
    # ========================================================================
    print("\n" + "=" * 80)
    print("Round-Trip Test")
    print("=" * 80)

    # Original
    original = constraint.get_parameters()

    # Pack
    vec = constraint.pack()

    # Unpack
    recovered = constraint.unpack(vec)

    # Compare
    print(f"\nOriginal:  {original}")
    print(f"Recovered: {recovered}")

    match = all(abs(original[k] - recovered[k]) < 1e-10 for k in original.keys())
    print(f"\nRound-trip successful: {match}")

    # ========================================================================
    # Null Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("Null Model (Edge Case)")
    print("=" * 80)

    null = AssemblyConstraint.null_model()

    print(f"\n{null}")
    print(f"Parameters: {null.get_parameters()}")
    print(f"Packed: {null.pack()}")
    print(f"Num params: {null.num_free_parameters()}")

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)

    print("\nPhase 1.6 ✓ Complete")
    print("  ✓ pack() implemented")
    print("  ✓ unpack() implemented")
    print("  ✓ num_free_parameters() implemented")
    print("  ✓ initial_parameters() implemented")
    print("  ✓ get_constrained_baseline() implemented")
    print("\nNext: Phase 1.7 (MLE inference)")


if __name__ == '__main__':
    main()
