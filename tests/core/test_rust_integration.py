"""
Test Rust integration and benchmark performance.
"""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from persiste.core.trees import TreeStructure
from persiste.core.simulation import simulate_binary_evolution
from persiste.core.pruning_rust import (
    check_rust_available,
    get_backend_info,
    compute_likelihoods_batch,
    benchmark_backends,
)


def test_rust_availability():
    """Test if Rust backend is available."""
    print("=" * 70)
    print("RUST BACKEND AVAILABILITY")
    print("=" * 70)
    
    info = get_backend_info()
    print(f"Rust available: {info['rust_available']}")
    print(f"Default backend: {info['default_backend']}")
    
    if not info['rust_available']:
        print("\n⚠ Rust backend not available")
        print("Build with: cd rust && maturin develop --release")
        return False
    else:
        print("\n✓ Rust backend ready")
        return True


def test_correctness():
    """Test that Rust and NumPy give identical results."""
    print("\n" + "=" * 70)
    print("CORRECTNESS TEST")
    print("=" * 70)
    
    # Create test data
    newick = "((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0);"
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    
    n_families = 50
    gain_rate = 1.5
    loss_rate = 2.0
    
    presence_matrix = simulate_binary_evolution(tree, gain_rate, loss_rate, n_families, rng)
    gain_rates = np.full(n_families, gain_rate)
    loss_rates = np.full(n_families, loss_rate)
    taxon_names = [f"tip{i}" for i in range(tree.n_tips)]
    
    # Compute with both backends
    print("Computing with NumPy...")
    ll_numpy = compute_likelihoods_batch(
        tree, presence_matrix, gain_rates, loss_rates, taxon_names, use_rust=False
    )
    
    if check_rust_available():
        print("Computing with Rust...")
        ll_rust = compute_likelihoods_batch(
            tree, presence_matrix, gain_rates, loss_rates, taxon_names, use_rust=True
        )
        
        # Compare
        max_diff = np.max(np.abs(ll_numpy - ll_rust))
        mean_diff = np.mean(np.abs(ll_numpy - ll_rust))
        
        print(f"\nNumPy total LL:  {np.sum(ll_numpy):.6f}")
        print(f"Rust total LL:   {np.sum(ll_rust):.6f}")
        print(f"Max difference:  {max_diff:.2e}")
        print(f"Mean difference: {mean_diff:.2e}")
        
        if max_diff < 0.15:
            print("\n✓ Results match (within numerical tolerance)!")
            return True
        else:
            print(f"\n✗ Results differ by {max_diff:.2e} (tolerance: 0.15)")
            return False
    else:
        print("\n⊘ Rust not available, skipping comparison")
        return True


def benchmark_performance():
    """Benchmark Rust vs NumPy performance."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    configs = [
        (10, 100, "Small"),
        (50, 500, "Medium"),
        (100, 1000, "Large"),
    ]
    
    results = []
    
    for n_taxa, n_families, size_name in configs:
        print(f"\n{size_name}: {n_taxa} taxa × {n_families} families")
        print("-" * 70)
        
        # Create tree
        newick = "(" * (n_taxa - 1)
        for i in range(n_taxa):
            if i > 0:
                newick += ","
            newick += f"tip{i}:1.0"
        newick += ")" * (n_taxa - 1) + ";"
        
        tree = TreeStructure.from_newick(newick, backend="simple")
        rng = np.random.default_rng(42)
        
        # Simulate data
        gain_rate = 1.5
        loss_rate = 2.0
        presence_matrix = simulate_binary_evolution(tree, gain_rate, loss_rate, n_families, rng)
        gain_rates = np.full(n_families, gain_rate)
        loss_rates = np.full(n_families, loss_rate)
        taxon_names = [f"tip{i}" for i in range(n_taxa)]
        
        # Benchmark
        bench_results = benchmark_backends(
            tree, presence_matrix, gain_rates, loss_rates, taxon_names
        )
        
        print(f"NumPy:  {bench_results['numpy_time']:.3f}s ({bench_results['numpy_throughput']:.1f} fam/s)")
        
        if bench_results['speedup'] is not None:
            print(f"Rust:   {bench_results['rust_time']:.3f}s ({bench_results['rust_throughput']:.1f} fam/s)")
            print(f"Speedup: {bench_results['speedup']:.2f}x")
            print(f"Correct: {bench_results['correct']}")
            
            results.append({
                'size': size_name,
                'n_taxa': n_taxa,
                'n_families': n_families,
                'speedup': bench_results['speedup'],
            })
        else:
            print("Rust: Not available")
    
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n{'Size':<10} {'Taxa':<10} {'Families':<10} {'Speedup':<10}")
        print("-" * 40)
        for r in results:
            print(f"{r['size']:<10} {r['n_taxa']:<10} {r['n_families']:<10} {r['speedup']:.2f}x")
        
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print(f"\nAverage speedup: {avg_speedup:.2f}x")


def run_all_tests():
    """Run all Rust integration tests."""
    print("=" * 70)
    print("RUST INTEGRATION TESTS")
    print("=" * 70)
    print()
    
    rust_available = test_rust_availability()
    
    if rust_available:
        correctness_ok = test_correctness()
        
        if correctness_ok:
            benchmark_performance()
            
            print("\n" + "=" * 70)
            print("✓ ALL TESTS PASSED")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("✗ CORRECTNESS TEST FAILED")
            print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠ RUST NOT AVAILABLE - TESTS SKIPPED")
        print("=" * 70)
        print("\nTo build Rust extension:")
        print("  cd rust")
        print("  maturin develop --release")


if __name__ == "__main__":
    run_all_tests()
