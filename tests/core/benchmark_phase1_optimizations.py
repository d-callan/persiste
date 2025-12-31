"""
Benchmark Phase 1 optimizations: LRU cache for transition matrices.
"""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from persiste.core.trees import TreeStructure
from persiste.core.pruning import FelsensteinPruning, SimpleBinaryTransitionProvider, ArrayTipConditionalProvider
from persiste.core.simulation import simulate_binary_evolution


def benchmark_caching(n_taxa=50, n_families=500):
    """Benchmark transition matrix caching."""
    print("=" * 70)
    print("PHASE 1 OPTIMIZATION BENCHMARK: Transition Matrix Caching")
    print("=" * 70)
    print()
    
    # Create tree
    print(f"Setup: {n_taxa} taxa, {n_families} families")
    newick = "(" * (n_taxa - 1)
    for i in range(n_taxa):
        if i > 0:
            newick += ","
        newick += f"tip{i}:1.0"
    newick += ")" * (n_taxa - 1) + ";"
    
    tree = TreeStructure.from_newick(newick, backend="simple")
    rng = np.random.default_rng(42)
    
    # Simulate data
    print("Simulating data...")
    gain_rate = 1.5
    loss_rate = 2.0
    presence_matrix = simulate_binary_evolution(tree, gain_rate, loss_rate, n_families, rng)
    
    # Setup
    pruning = FelsensteinPruning(tree, n_states=2, use_jax=False)
    taxon_names = [f"tip{i}" for i in range(n_taxa)]
    
    print()
    print("-" * 70)
    print("TEST 1: WITHOUT CACHING")
    print("-" * 70)
    
    start = time.time()
    lls_no_cache = []
    for fam_idx in range(n_families):
        transition_provider = SimpleBinaryTransitionProvider(
            gain_rate, loss_rate, use_cache=False
        )
        tip_data = presence_matrix[:, fam_idx:fam_idx+1]
        tip_provider = ArrayTipConditionalProvider(tip_data, taxon_names, n_states=2)
        result = pruning.compute_likelihood(transition_provider, tip_provider, n_sites=1)
        lls_no_cache.append(result.log_likelihood)
    time_no_cache = time.time() - start
    total_ll_no_cache = sum(lls_no_cache)
    
    print(f"Time: {time_no_cache:.3f}s")
    print(f"Total LL: {total_ll_no_cache:.2f}")
    print(f"Throughput: {n_families/time_no_cache:.1f} families/sec")
    
    print()
    print("-" * 70)
    print("TEST 2: WITH CACHING")
    print("-" * 70)
    
    start = time.time()
    lls_with_cache = []
    for fam_idx in range(n_families):
        transition_provider = SimpleBinaryTransitionProvider(
            gain_rate, loss_rate, use_cache=True
        )
        tip_data = presence_matrix[:, fam_idx:fam_idx+1]
        tip_provider = ArrayTipConditionalProvider(tip_data, taxon_names, n_states=2)
        result = pruning.compute_likelihood(transition_provider, tip_provider, n_sites=1)
        lls_with_cache.append(result.log_likelihood)
    time_with_cache = time.time() - start
    total_ll_with_cache = sum(lls_with_cache)
    
    print(f"Time: {time_with_cache:.3f}s")
    print(f"Total LL: {total_ll_with_cache:.2f}")
    print(f"Throughput: {n_families/time_with_cache:.1f} families/sec")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    speedup = time_no_cache / time_with_cache
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time saved: {time_no_cache - time_with_cache:.3f}s ({100*(1-time_with_cache/time_no_cache):.1f}%)")
    
    # Verify correctness
    max_diff = max(abs(a - b) for a, b in zip(lls_no_cache, lls_with_cache))
    print(f"Max difference: {max_diff:.2e} (should be ~0)")
    
    if max_diff < 1e-10:
        print("✓ Results match perfectly!")
    else:
        print("⚠ Results differ - caching may have issues")
    
    print()
    return speedup


def benchmark_different_sizes():
    """Benchmark across different dataset sizes."""
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS")
    print("=" * 70)
    print()
    
    configs = [
        (10, 100),
        (20, 200),
        (50, 500),
    ]
    
    results = []
    for n_taxa, n_families in configs:
        print(f"\nTesting {n_taxa} taxa × {n_families} families...")
        speedup = benchmark_caching(n_taxa, n_families)
        results.append((n_taxa, n_families, speedup))
        print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Taxa':<10} {'Families':<10} {'Speedup':<10}")
    print("-" * 30)
    for n_taxa, n_families, speedup in results:
        print(f"{n_taxa:<10} {n_families:<10} {speedup:.2f}x")
    
    avg_speedup = sum(s for _, _, s in results) / len(results)
    print()
    print(f"Average speedup: {avg_speedup:.2f}x")
    print()


if __name__ == "__main__":
    # Quick test
    speedup = benchmark_caching(n_taxa=50, n_families=500)
    
    # Uncomment for full scaling analysis
    # benchmark_different_sizes()
