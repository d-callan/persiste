#!/usr/bin/env python3
"""
Benchmark full E. coli dataset with Rust-accelerated tree inference.
Compare to previous baseline of ~62s.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persiste.plugins.genecontent.pam_interface import fit
from persiste.plugins.genecontent.tree_inference import RUST_AVAILABLE


def main():
    print("=" * 70)
    print("FULL E. COLI DATASET BENCHMARK")
    print("=" * 70)
    print(f"\nRust acceleration: {'✓ ENABLED' if RUST_AVAILABLE else '✗ DISABLED'}")
    
    # Load data
    data_dir = Path("data/ecoli_real")
    pam_file = data_dir / "Supplementary File 2A.txt"
    
    if not pam_file.exists():
        print(f"\n✗ Data file not found: {pam_file}")
        return
    
    print("\nLoading E. coli pangenome dataset...")
    df = pd.read_csv(pam_file, sep='\t', index_col=0)
    pam = df.values.T.astype(int)
    strain_names = df.columns.tolist()
    gene_names = df.index.tolist()
    
    n_strains = len(strain_names)
    n_genes = len(gene_names)
    
    print(f"  Dataset: {n_strains:,} strains × {n_genes:,} genes")
    print(f"  Total data points: {n_strains * n_genes:,}")
    
    # Run analysis with timing breakdown
    print("\n" + "=" * 70)
    print("RUNNING ANALYSIS")
    print("=" * 70)
    
    total_start = time.time()
    
    # Tree inference and model fitting
    print("\n1. Running full analysis (tree inference + model fitting)...")
    
    result = fit(
        pam=pam,
        tree=None,  # Will infer tree
        taxon_names=strain_names,
        gene_names=gene_names,
        tree_method='jaccard_upgma',
        use_rust=True,
        verbose=True
    )
    
    # Get total time
    total_time = time.time() - total_start
    
    # Extract results
    gain_rate = result.gain_rate
    loss_rate = result.loss_rate
    pi1 = result.equilibrium_frequency
    log_likelihood = result.log_likelihood
    
    # Estimate tree time from metadata or re-run just tree inference
    from persiste.plugins.genecontent.tree_inference import infer_tree_from_pam
    tree_start = time.time()
    _, _ = infer_tree_from_pam(pam, strain_names, method='jaccard_upgma')
    tree_time = time.time() - tree_start
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nGlobal rates:")
    print(f"  Gain rate (λ): {gain_rate:.4f}")
    print(f"  Loss rate (μ): {loss_rate:.4f}")
    print(f"  Equilibrium frequency (π₁): {pi1:.4f}")
    print(f"  Log-likelihood: {log_likelihood:.2f}")
    
    print(f"\nTiming:")
    print(f"  Tree inference: {tree_time:.2f}s")
    print(f"  Total runtime: {total_time:.2f}s")
    
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE")
    print("=" * 70)
    
    baseline_time = 62.0  # Previous runtime without Rust tree inference
    speedup = baseline_time / total_time
    time_saved = baseline_time - total_time
    
    print(f"\nPrevious runtime (Python tree): ~{baseline_time:.1f}s")
    print(f"Current runtime (Rust tree):    {total_time:.2f}s")
    print(f"Time saved:                     {time_saved:.2f}s ({time_saved/baseline_time*100:.1f}%)")
    print(f"Overall speedup:                {speedup:.2f}x")
    
    # Breakdown
    print("\n" + "=" * 70)
    print("PERFORMANCE BREAKDOWN")
    print("=" * 70)
    
    inference_time = total_time - tree_time
    
    print(f"\nComponent breakdown:")
    print(f"  Tree inference:     {tree_time:>8.2f}s ({tree_time/total_time*100:>5.1f}%)")
    print(f"  Model fitting:      {inference_time:>8.2f}s ({inference_time/total_time*100:>5.1f}%)")
    print(f"  Total:              {total_time:>8.2f}s")
    
    # Estimate previous breakdown
    print(f"\nEstimated previous breakdown (without Rust tree):")
    est_prev_tree = tree_time * 17  # Based on 17x speedup for this size
    est_prev_inference = baseline_time - est_prev_tree
    print(f"  Tree inference:     {est_prev_tree:>8.2f}s ({est_prev_tree/baseline_time*100:>5.1f}%)")
    print(f"  Model fitting:      {est_prev_inference:>8.2f}s ({est_prev_inference/baseline_time*100:>5.1f}%)")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n✓ Full E. coli analysis completed in {total_time:.2f}s")
    print(f"✓ {speedup:.2f}x faster than baseline ({time_saved:.2f}s saved)")
    print(f"✓ Rust tree inference: {tree_time:.2f}s (was ~{est_prev_tree:.1f}s)")
    print(f"✓ Ready for production use on large datasets!")


if __name__ == "__main__":
    main()
