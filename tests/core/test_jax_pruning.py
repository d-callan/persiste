"""
Test JAX-accelerated pruning for correctness and performance.
"""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from persiste.core.trees import TreeStructure
from persiste.core.pruning import FelsensteinPruning, SimpleBinaryTransitionProvider, ArrayTipConditionalProvider
from persiste.core.pruning_jax import (
    JAXFelsensteinPruning,
    check_jax_available,
    get_jax_device_info,
    compute_transition_matrix_2x2
)
from persiste.core.simulation import simulate_binary_evolution


def test_jax_available():
    """Test if JAX is available."""
    print("Checking JAX availability...")
    available = check_jax_available()
    
    if available:
        info = get_jax_device_info()
        print(f"  ✓ JAX is available")
        print(f"    Backend: {info['default_backend']}")
        print(f"    Devices: {info['devices']}")
    else:
        print("  ⚠ JAX is not available - install with: pip install jax jaxlib")
        print("    Skipping JAX tests")
    
    return available


def test_transition_matrix():
    """Test analytical 2x2 transition matrix computation."""
    print("\nTesting analytical transition matrix...")
    
    if not check_jax_available():
        print("  ⊘ Skipped (JAX not available)")
        return
    
    import jax.numpy as jnp
    from scipy.linalg import expm
    
    # Test parameters
    gain_rate = 1.5
    loss_rate = 2.0
    t = 1.0
    
    # Analytical solution
    P_analytical = compute_transition_matrix_2x2(gain_rate, loss_rate, t)
    
    # Matrix exponential (ground truth)
    Q = np.array([[-gain_rate, gain_rate], [loss_rate, -loss_rate]])
    P_expm = expm(Q * t)
    
    # Check they match
    diff = np.max(np.abs(np.array(P_analytical) - P_expm))
    assert diff < 1e-8, f"Analytical solution differs from expm by {diff}"
    
    print(f"  ✓ Analytical solution matches expm (max diff: {diff:.2e})")
    
    # Test edge case: no evolution
    P_zero = compute_transition_matrix_2x2(0.0, 0.0, 1.0)
    assert np.allclose(P_zero, np.eye(2)), "Zero rates should give identity matrix"
    
    print("  ✓ Edge cases handled correctly")


def test_jax_vs_numpy_correctness():
    """Test that JAX pruning gives same results as NumPy."""
    print("\nTesting JAX vs NumPy correctness...")
    
    if not check_jax_available():
        print("  ⊘ Skipped (JAX not available)")
        return
    
    # Create test data
    newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    
    n_families = 100
    gain_rate = 1.5
    loss_rate = 2.0
    
    # Simulate data
    presence_matrix = simulate_binary_evolution(tree, gain_rate, loss_rate, n_families, rng)
    
    # NumPy pruning (sequential)
    numpy_pruning = FelsensteinPruning(tree, n_states=2, use_jax=False)
    
    numpy_lls = []
    taxon_names = [f"tip{i}" for i in range(tree.n_tips)]
    
    for fam_idx in range(n_families):
        # Create transition provider
        transition_provider = SimpleBinaryTransitionProvider(gain_rate, loss_rate)
        
        # Create tip provider - needs (n_taxa, n_sites) array
        tip_data = presence_matrix[:, fam_idx:fam_idx+1]  # Keep 2D shape
        tip_provider = ArrayTipConditionalProvider(tip_data, taxon_names, n_states=2)
        
        # Compute likelihood
        result = numpy_pruning.compute_likelihood(
            transition_provider=transition_provider,
            tip_provider=tip_provider,
            n_sites=1
        )
        numpy_lls.append(result.log_likelihood)
    
    numpy_total = sum(numpy_lls)
    
    # JAX pruning (vectorized)
    jax_pruning = JAXFelsensteinPruning(tree, n_states=2)
    
    gain_rates = np.full(n_families, gain_rate)
    loss_rates = np.full(n_families, loss_rate)
    
    jax_result = jax_pruning.compute_likelihood_batch(
        gain_rates=gain_rates,
        loss_rates=loss_rates,
        tip_data=presence_matrix
    )
    
    # Compare results
    max_diff = np.max(np.abs(np.array(numpy_lls) - jax_result.log_likelihoods))
    total_diff = abs(numpy_total - jax_result.total_log_likelihood)
    
    print(f"  NumPy total LL: {numpy_total:.6f}")
    print(f"  JAX total LL:   {jax_result.total_log_likelihood:.6f}")
    print(f"  Max per-family diff: {max_diff:.2e}")
    print(f"  Total LL diff: {total_diff:.2e}")
    
    # Should match within numerical precision
    assert max_diff < 1e-6, f"Per-family likelihoods differ by {max_diff}"
    assert total_diff < 1e-5, f"Total likelihoods differ by {total_diff}"
    
    print("  ✓ JAX and NumPy results match!")


def test_jax_performance():
    """Benchmark JAX vs NumPy performance."""
    print("\nBenchmarking JAX vs NumPy performance...")
    
    if not check_jax_available():
        print("  ⊘ Skipped (JAX not available)")
        return
    
    # Create larger test case
    newick = "(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    
    n_families = 500
    gain_rate = 1.5
    loss_rate = 2.0
    
    # Simulate data
    presence_matrix = simulate_binary_evolution(tree, gain_rate, loss_rate, n_families, rng)
    
    # Benchmark NumPy (sequential)
    print(f"\n  Testing with {tree.n_tips} taxa, {n_families} families...")
    print("  NumPy (sequential):")
    
    numpy_pruning = FelsensteinPruning(tree, n_states=2, use_jax=False)
    taxon_names = [f"tip{i}" for i in range(tree.n_tips)]
    
    start = time.time()
    numpy_lls = []
    for fam_idx in range(n_families):
        transition_provider = SimpleBinaryTransitionProvider(gain_rate, loss_rate)
        tip_data = presence_matrix[:, fam_idx:fam_idx+1]
        tip_provider = ArrayTipConditionalProvider(tip_data, taxon_names, n_states=2)
        result = numpy_pruning.compute_likelihood(transition_provider, tip_provider, n_sites=1)
        numpy_lls.append(result.log_likelihood)
    numpy_time = time.time() - start
    numpy_total = sum(numpy_lls)
    
    print(f"    Time: {numpy_time:.3f}s")
    print(f"    Total LL: {numpy_total:.2f}")
    
    # Benchmark JAX (vectorized)
    print("  JAX (vectorized):")
    
    jax_pruning = JAXFelsensteinPruning(tree, n_states=2)
    gain_rates = np.full(n_families, gain_rate)
    loss_rates = np.full(n_families, loss_rate)
    
    # Warm-up run (JIT compilation)
    _ = jax_pruning.compute_likelihood_batch(gain_rates, loss_rates, presence_matrix)
    
    # Timed run
    start = time.time()
    jax_result = jax_pruning.compute_likelihood_batch(gain_rates, loss_rates, presence_matrix)
    jax_time = time.time() - start
    
    print(f"    Time: {jax_time:.3f}s")
    print(f"    Total LL: {jax_result.total_log_likelihood:.2f}")
    
    # Compute speedup
    speedup = numpy_time / jax_time
    print(f"\n  ⚡ Speedup: {speedup:.1f}x")
    
    # Verify correctness
    max_diff = np.max(np.abs(np.array(numpy_lls) - jax_result.log_likelihoods))
    assert max_diff < 1e-6, f"Results differ by {max_diff}"
    print(f"  ✓ Results match (max diff: {max_diff:.2e})")
    
    return speedup


def run_all_tests():
    """Run all JAX pruning tests."""
    print("=" * 60)
    print("JAX Pruning Tests")
    print("=" * 60)
    
    try:
        jax_available = test_jax_available()
        
        if jax_available:
            test_transition_matrix()
            test_jax_vs_numpy_correctness()
            speedup = test_jax_performance()
            
            print("\n" + "=" * 60)
            print("✓ ALL JAX TESTS PASSED")
            print(f"  Final speedup: {speedup:.1f}x")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("⚠ JAX NOT AVAILABLE")
            print("  Install with: pip install jax jaxlib")
            print("=" * 60)
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
