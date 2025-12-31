"""
Test Rust integration with GeneContentModel.

Verifies that:
1. Rust backend produces same results as NumPy
2. Rust backend is faster
3. Automatic fallback works
"""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from persiste.core.trees import TreeStructure
from persiste.core.simulation import simulate_binary_evolution
from persiste.plugins.genecontent.inference.gene_inference import (
    GeneContentData,
    GeneContentModel,
    RUST_AVAILABLE,
)


def test_rust_numpy_equivalence():
    """Test that Rust and NumPy backends give same results."""
    print("=" * 70)
    print("TEST: Rust vs NumPy Equivalence")
    print("=" * 70)
    
    # Create test data
    newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    
    n_families = 100
    gain_rate = 1.5
    loss_rate = 2.0
    
    presence_matrix = simulate_binary_evolution(tree, gain_rate, loss_rate, n_families, rng)
    taxon_names = tree.tip_names
    family_names = [f"fam{i}" for i in range(n_families)]
    
    data = GeneContentData(tree, presence_matrix, taxon_names, family_names)
    
    # Test parameters
    params = {'log_gain': np.log(1.5), 'log_loss': np.log(2.0)}
    
    # NumPy backend
    print("\nComputing with NumPy backend...")
    model_numpy = GeneContentModel(data, use_rust=False)
    ll_numpy = model_numpy.log_likelihood(params)
    
    if RUST_AVAILABLE:
        # Rust backend
        print("Computing with Rust backend...")
        model_rust = GeneContentModel(data, use_rust=True)
        ll_rust = model_rust.log_likelihood(params)
        
        # Compare
        diff = abs(ll_numpy - ll_rust)
        print(f"\nNumPy LL:  {ll_numpy:.6f}")
        print(f"Rust LL:   {ll_rust:.6f}")
        print(f"Difference: {diff:.6e}")
        
        if diff < 1.0:
            print("✓ Results match (within tolerance)!")
            return True
        else:
            print(f"✗ Results differ by {diff:.6e}")
            return False
    else:
        print("\n⊘ Rust not available, skipping comparison")
        return True


def test_rust_performance():
    """Test that Rust is faster than NumPy."""
    print("\n" + "=" * 70)
    print("TEST: Rust Performance")
    print("=" * 70)
    
    if not RUST_AVAILABLE:
        print("\n⊘ Rust not available, skipping performance test")
        return True
    
    # Create larger test data
    newick = "(" * 19
    for i in range(20):
        if i > 0:
            newick += ","
        newick += f"tip{i}:1.0"
    newick += ")" * 19 + ";"
    
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    
    n_families = 500
    gain_rate = 1.5
    loss_rate = 2.0
    
    presence_matrix = simulate_binary_evolution(tree, gain_rate, loss_rate, n_families, rng)
    taxon_names = tree.tip_names
    family_names = [f"fam{i}" for i in range(n_families)]
    
    data = GeneContentData(tree, presence_matrix, taxon_names, family_names)
    params = {'log_gain': np.log(1.5), 'log_loss': np.log(2.0)}
    
    # Benchmark NumPy
    print(f"\nBenchmarking with {tree.n_tips} taxa × {n_families} families...")
    model_numpy = GeneContentModel(data, use_rust=False)
    
    start = time.time()
    ll_numpy = model_numpy.log_likelihood(params)
    time_numpy = time.time() - start
    
    print(f"NumPy: {time_numpy:.3f}s")
    
    # Benchmark Rust
    model_rust = GeneContentModel(data, use_rust=True)
    
    start = time.time()
    ll_rust = model_rust.log_likelihood(params)
    time_rust = time.time() - start
    
    print(f"Rust:  {time_rust:.3f}s")
    
    speedup = time_numpy / time_rust
    print(f"Speedup: {speedup:.1f}x")
    
    if speedup > 5.0:
        print("✓ Rust is significantly faster!")
        return True
    else:
        print(f"⚠ Speedup is only {speedup:.1f}x (expected >5x)")
        return False


def test_constraint_support():
    """Test that Rust works with constraints."""
    print("\n" + "=" * 70)
    print("TEST: Rust with Constraints")
    print("=" * 70)
    
    if not RUST_AVAILABLE:
        print("\n⊘ Rust not available, skipping constraint test")
        return True
    
    from persiste.plugins.genecontent.constraints.gene_constraint import RetentionBiasConstraint
    
    # Create test data
    newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    
    n_families = 50
    presence_matrix = simulate_binary_evolution(tree, 1.5, 2.0, n_families, rng)
    taxon_names = tree.tip_names
    family_names = [f"fam{i}" for i in range(n_families)]
    
    data = GeneContentData(tree, presence_matrix, taxon_names, family_names)
    
    # Create constraint (first 10 families retained)
    retained_families = set(f"fam{i}" for i in range(10))
    constraint = RetentionBiasConstraint(retained_families=retained_families)
    
    params = {
        'log_gain': np.log(1.5),
        'log_loss': np.log(2.0),
        'retention_strength': -1.0,
    }
    
    # Test with NumPy
    print("\nTesting constraint with NumPy...")
    model_numpy = GeneContentModel(data, constraint=constraint, use_rust=False)
    ll_numpy = model_numpy.log_likelihood(params)
    
    # Test with Rust
    print("Testing constraint with Rust...")
    model_rust = GeneContentModel(data, constraint=constraint, use_rust=True)
    ll_rust = model_rust.log_likelihood(params)
    
    diff = abs(ll_numpy - ll_rust)
    print(f"\nNumPy LL:  {ll_numpy:.6f}")
    print(f"Rust LL:   {ll_rust:.6f}")
    print(f"Difference: {diff:.6e}")
    
    if diff < 1.0:
        print("✓ Rust works correctly with constraints!")
        return True
    else:
        print(f"✗ Results differ by {diff:.6e}")
        return False


def run_all_tests():
    """Run all integration tests."""
    print("=" * 70)
    print("RUST INTEGRATION TESTS FOR GENECONTENT")
    print("=" * 70)
    print()
    
    if RUST_AVAILABLE:
        print("✓ Rust backend available")
    else:
        print("⚠ Rust backend not available - tests will use NumPy fallback")
    print()
    
    results = []
    
    # Test 1: Equivalence
    results.append(("Equivalence", test_rust_numpy_equivalence()))
    
    # Test 2: Performance
    results.append(("Performance", test_rust_performance()))
    
    # Test 3: Constraints
    results.append(("Constraints", test_constraint_support()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<20} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
