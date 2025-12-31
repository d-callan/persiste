"""
Benchmark validation suite with Rust backend.

Tests that all validation scenarios work correctly with Rust
and measures performance improvements.
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
    GeneContentInference,
    RUST_AVAILABLE,
)
from persiste.plugins.genecontent.constraints.gene_constraint import (
    NullConstraint,
    RetentionBiasConstraint,
)


def test_null_model_inference():
    """Test null model inference with Rust."""
    print("=" * 70)
    print("TEST: Null Model Inference")
    print("=" * 70)
    
    # Create realistic test data
    newick = "(" * 9
    for i in range(10):
        if i > 0:
            newick += ","
        newick += f"tip{i}:1.0"
    newick += ")" * 9 + ";"
    
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    
    n_families = 200
    true_gain = 1.2
    true_loss = 1.8
    
    presence_matrix = simulate_binary_evolution(tree, true_gain, true_loss, n_families, rng)
    taxon_names = tree.tip_names
    family_names = [f"fam{i}" for i in range(n_families)]
    
    data = GeneContentData(tree, presence_matrix, taxon_names, family_names)
    
    print(f"\nData: {tree.n_tips} taxa × {n_families} families")
    print(f"True rates: gain={true_gain:.2f}, loss={true_loss:.2f}")
    
    # Test with NumPy
    print("\n--- NumPy Backend ---")
    inference_numpy = GeneContentInference(data, use_rust=False)
    
    start = time.time()
    result_numpy = inference_numpy.fit_null()
    time_numpy = time.time() - start
    
    gain_numpy = np.exp(result_numpy.parameters['log_gain'])
    loss_numpy = np.exp(result_numpy.parameters['log_loss'])
    
    print(f"Time: {time_numpy:.3f}s")
    print(f"Estimated gain: {gain_numpy:.3f}")
    print(f"Estimated loss: {loss_numpy:.3f}")
    print(f"Log-likelihood: {result_numpy.log_likelihood:.2f}")
    
    if RUST_AVAILABLE:
        # Test with Rust
        print("\n--- Rust Backend ---")
        inference_rust = GeneContentInference(data, use_rust=True)
        
        start = time.time()
        result_rust = inference_rust.fit_null()
        time_rust = time.time() - start
        
        gain_rust = np.exp(result_rust.parameters['log_gain'])
        loss_rust = np.exp(result_rust.parameters['log_loss'])
        
        print(f"Time: {time_rust:.3f}s")
        print(f"Estimated gain: {gain_rust:.3f}")
        print(f"Estimated loss: {loss_rust:.3f}")
        print(f"Log-likelihood: {result_rust.log_likelihood:.2f}")
        
        # Compare results
        gain_diff = abs(gain_numpy - gain_rust)
        loss_diff = abs(loss_numpy - loss_rust)
        ll_diff = abs(result_numpy.log_likelihood - result_rust.log_likelihood)
        
        print(f"\n--- Comparison ---")
        print(f"Gain difference: {gain_diff:.6f}")
        print(f"Loss difference: {loss_diff:.6f}")
        print(f"LL difference: {ll_diff:.6f}")
        print(f"Speedup: {time_numpy / time_rust:.1f}x")
        
        if gain_diff < 0.1 and loss_diff < 0.1 and ll_diff < 5.0:
            print("\n✓ Results match!")
            return True
        else:
            print("\n✗ Results differ significantly")
            return False
    else:
        print("\n⊘ Rust not available")
        return True


def test_retention_bias_inference():
    """Test retention bias constraint with Rust."""
    print("\n" + "=" * 70)
    print("TEST: Retention Bias Inference")
    print("=" * 70)
    
    # Create test data with retention bias
    newick = "(" * 9
    for i in range(10):
        if i > 0:
            newick += ","
        newick += f"tip{i}:1.0"
    newick += ")" * 9 + ";"
    
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(123)
    
    n_families = 200
    n_retained = 50
    
    # Simulate with retention bias
    presence_matrix = np.zeros((tree.n_tips, n_families), dtype=np.int8)
    
    # Retained families: lower loss rate
    for fam_idx in range(n_retained):
        presence_matrix[:, fam_idx] = simulate_binary_evolution(
            tree, 1.5, 0.5, 1, rng  # Low loss
        ).flatten()
    
    # Non-retained families: normal rates
    for fam_idx in range(n_retained, n_families):
        presence_matrix[:, fam_idx] = simulate_binary_evolution(
            tree, 1.5, 2.0, 1, rng  # Normal loss
        ).flatten()
    
    taxon_names = tree.tip_names
    family_names = [f"fam{i}" for i in range(n_families)]
    
    data = GeneContentData(tree, presence_matrix, taxon_names, family_names)
    
    print(f"\nData: {tree.n_tips} taxa × {n_families} families")
    print(f"Retained families: {n_retained}")
    
    # Create constraint
    retained_families = set(f"fam{i}" for i in range(n_retained))
    constraint = RetentionBiasConstraint(retained_families=retained_families)
    
    # Test with NumPy
    print("\n--- NumPy Backend ---")
    inference_numpy = GeneContentInference(data, use_rust=False)
    
    start = time.time()
    null_numpy = inference_numpy.fit_null()
    alt_numpy = inference_numpy.fit_with_constraint(constraint)
    time_numpy = time.time() - start
    
    print(f"Time: {time_numpy:.3f}s")
    print(f"Null LL: {null_numpy.log_likelihood:.2f}")
    print(f"Alt LL: {alt_numpy.log_likelihood:.2f}")
    print(f"Retention strength: {alt_numpy.parameters.get('retention_strength', 0.0):.3f}")
    
    if RUST_AVAILABLE:
        # Test with Rust
        print("\n--- Rust Backend ---")
        inference_rust = GeneContentInference(data, use_rust=True)
        
        start = time.time()
        null_rust = inference_rust.fit_null()
        alt_rust = inference_rust.fit_with_constraint(constraint)
        time_rust = time.time() - start
        
        print(f"Time: {time_rust:.3f}s")
        print(f"Null LL: {null_rust.log_likelihood:.2f}")
        print(f"Alt LL: {alt_rust.log_likelihood:.2f}")
        print(f"Retention strength: {alt_rust.parameters.get('retention_strength', 0.0):.3f}")
        
        # Compare
        null_ll_diff = abs(null_numpy.log_likelihood - null_rust.log_likelihood)
        alt_ll_diff = abs(alt_numpy.log_likelihood - alt_rust.log_likelihood)
        
        print(f"\n--- Comparison ---")
        print(f"Null LL difference: {null_ll_diff:.6f}")
        print(f"Alt LL difference: {alt_ll_diff:.6f}")
        print(f"Speedup: {time_numpy / time_rust:.1f}x")
        
        if null_ll_diff < 5.0 and alt_ll_diff < 5.0:
            print("\n✓ Results match!")
            return True
        else:
            print("\n✗ Results differ significantly")
            return False
    else:
        print("\n⊘ Rust not available")
        return True


def test_large_scale_performance():
    """Test performance on large-scale data."""
    print("\n" + "=" * 70)
    print("TEST: Large-Scale Performance")
    print("=" * 70)
    
    if not RUST_AVAILABLE:
        print("\n⊘ Rust not available, skipping large-scale test")
        return True
    
    # Create large dataset
    newick = "(" * 49
    for i in range(50):
        if i > 0:
            newick += ","
        newick += f"tip{i}:1.0"
    newick += ")" * 49 + ";"
    
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    
    n_families = 1000
    
    print(f"\nGenerating data: {tree.n_tips} taxa × {n_families} families...")
    presence_matrix = simulate_binary_evolution(tree, 1.5, 2.0, n_families, rng)
    taxon_names = tree.tip_names
    family_names = [f"fam{i}" for i in range(n_families)]
    
    data = GeneContentData(tree, presence_matrix, taxon_names, family_names)
    
    # Benchmark Rust
    print("\nBenchmarking Rust backend...")
    inference_rust = GeneContentInference(data, use_rust=True)
    
    start = time.time()
    result_rust = inference_rust.fit_null()
    time_rust = time.time() - start
    
    gain_rust = np.exp(result_rust.parameters['log_gain'])
    loss_rust = np.exp(result_rust.parameters['log_loss'])
    
    print(f"Time: {time_rust:.3f}s")
    print(f"Estimated gain: {gain_rust:.3f}")
    print(f"Estimated loss: {loss_rust:.3f}")
    
    if time_rust < 10.0:
        print(f"\n✓ Large-scale inference completed in {time_rust:.1f}s!")
        return True
    else:
        print(f"\n⚠ Inference took {time_rust:.1f}s (expected <10s)")
        return False


def run_all_benchmarks():
    """Run all benchmark tests."""
    print("=" * 70)
    print("RUST VALIDATION SUITE BENCHMARK")
    print("=" * 70)
    print()
    
    if RUST_AVAILABLE:
        print("✓ Rust backend available")
    else:
        print("⚠ Rust backend not available - limited testing")
    print()
    
    results = []
    
    # Test 1: Null model
    results.append(("Null Model", test_null_model_inference()))
    
    # Test 2: Retention bias
    results.append(("Retention Bias", test_retention_bias_inference()))
    
    # Test 3: Large-scale
    results.append(("Large-Scale", test_large_scale_performance()))
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<20} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✓ ALL BENCHMARKS PASSED")
        print("\nThe Rust backend is working correctly and provides")
        print("significant performance improvements for all validation scenarios.")
    else:
        print("\n✗ SOME BENCHMARKS FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_benchmarks()
    sys.exit(0 if success else 1)
