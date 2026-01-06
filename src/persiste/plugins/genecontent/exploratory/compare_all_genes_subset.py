#!/usr/bin/env python3
"""
Compare GeneContent vs GLOOME on ALL genes but subset of strains.

This tests if GLOOME can handle many genes (25,420) with few strains (25).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
import subprocess
import shutil


repo_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(repo_root / "src"))

from persiste.core.tree_utils import prepare_tree_from_binary_matrix
from persiste.core.trees import TreeStructure
from persiste.plugins.genecontent import pam_interface


def run_genecontent(pam, strain_names, gene_names, output_dir: Path):
    """Run GeneContent analysis."""
    print("\n" + "=" * 70)
    print("GENECONTENT")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nDataset: {len(strain_names)} strains × {len(gene_names):,} genes")
    
    # Infer tree
    print("Inferring tree...")
    tree_start = time.time()
    tree, metadata = prepare_tree_from_binary_matrix(
        binary_matrix=pam,
        taxon_names=strain_names,
        tree=None,
        tree_method="jaccard_upgma",
        verbose=False,
    )
    tree_time = time.time() - tree_start
    print(f"  ✓ Tree inferred in {tree_time:.2f}s")
    
    # Save tree
    tree_file = output_dir / "tree.nwk"
    with open(tree_file, 'w') as f:
        f.write(tree.to_newick())
        f.write('\n')
    print(f"  Saved: {tree_file}")
    
    # Fit model
    print("\nFitting model...")
    fit_start = time.time()
    result = pam_interface.fit(
        pam=pam,
        tree=tree,
        taxon_names=strain_names,
        gene_names=gene_names,
        use_rust=True,
        verbose=False,
    )
    fit_time = time.time() - fit_start
    print(f"  ✓ Model fitted in {fit_time:.2f}s")
    
    total_time = tree_time + fit_time
    
    print(f"\nResults:")
    print(f"  Gain rate (λ): {result.gain_rate:.6f}")
    print(f"  Loss rate (μ): {result.loss_rate:.6f}")
    print(f"  π₁: {result.equilibrium_frequency:.6f}")
    print(f"  Log-likelihood: {result.log_likelihood:.2f}")
    print(f"  Total time: {total_time:.2f}s")
    
    # Save results
    results_file = output_dir / "results.txt"
    with open(results_file, 'w') as f:
        f.write(f"GeneContent Results\n")
        f.write(f"=" * 70 + "\n\n")
        f.write(f"Dataset: {len(strain_names)} strains × {len(gene_names):,} genes\n\n")
        f.write(f"Gain rate (λ): {result.gain_rate:.6f}\n")
        f.write(f"Loss rate (μ): {result.loss_rate:.6f}\n")
        f.write(f"π₁: {result.equilibrium_frequency:.6f}\n")
        f.write(f"Log-likelihood: {result.log_likelihood:.2f}\n")
        f.write(f"Tree time: {tree_time:.2f}s\n")
        f.write(f"Fit time: {fit_time:.2f}s\n")
        f.write(f"Total time: {total_time:.2f}s\n")
    
    return {
        'gain': result.gain_rate,
        'loss': result.loss_rate,
        'pi1': result.equilibrium_frequency,
        'll': result.log_likelihood,
        'time': total_time,
        'tree_file': tree_file,
    }


def run_gloome_with_tree(pam, strain_names, gene_names, tree_file: Path, output_dir: Path, label: str):
    """Run GLOOME with provided tree."""
    print("\n" + "=" * 70)
    print(f"GLOOME ({label})")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not shutil.which('gainLoss'):
        print("  ✗ GLOOME not available")
        return None
    
    print(f"\nDataset: {len(strain_names)} strains × {len(gene_names):,} genes")
    
    def quote_taxon(name: str) -> str:
        return "'" + name.replace("'", "\\'") + "'"

    # Create sequence file
    seq_file = output_dir / 'sequences.fa'
    print(f"Writing sequences...")
    with open(seq_file, 'w') as f:
        for i, taxon in enumerate(strain_names):
            f.write(f">{quote_taxon(taxon)}\n")
            sequence = ''.join([str(int(pam[i, j])) for j in range(len(gene_names))])
            f.write(sequence + '\n')
    
    # Copy tree
    gloome_tree = output_dir / 'tree.nwk'
    with open(tree_file, "r") as f:
        newick = f.read().strip()
    with open(gloome_tree, "w") as f:
        f.write(newick)
        f.write("\n")
    
    # Create parameter file
    param_file = output_dir / 'params.txt'
    with open(param_file, 'w') as f:
        f.write(f'_seqFile {seq_file.absolute()}\n')
        f.write(f'_treeFile {gloome_tree.absolute()}\n')
        f.write(f'_outDir {output_dir.absolute()}\n')
        f.write('_logFile gainLoss.log\n')
        f.write('_gainLossDist 1\n')
        f.write('_numberOfGainCategories 1\n')
        f.write('_numberOfLossCategories 1\n')
        f.write('_maxNumOfIterationsModel 10\n')
        f.write('_performOptimizations 1\n')
    
    print(f"Running GLOOME...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['gainLoss', str(param_file.absolute())],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
            cwd=str(output_dir)
        )
        runtime = time.time() - start_time
        
        if result.returncode != 0:
            print(f"  ✗ GLOOME failed (rc={result.returncode})")
            print(f"  STDERR: {result.stderr[:500]}")
            
            log_file = output_dir / 'gainLoss.log'
            if log_file.exists():
                with open(log_file) as f:
                    log_content = f.read()
                    print(f"\n  Last 20 lines of log:")
                    print('  ' + '\n  '.join(log_content.split('\n')[-20:]))
            return None
        
        print(f"  ✓ GLOOME completed in {runtime:.2f}s")
        
        # Parse results
        params_file = output_dir / 'EstimatedParameters.txt'
        if not params_file.exists():
            print("  ✗ Output file not found")
            return None
        
        gain_rate = None
        loss_rate = None
        log_likelihood = None
        
        with open(params_file) as f:
            for line in f:
                if 'Gain Expectation' in line:
                    gain_rate = float(line.split('=')[1].strip())
                elif 'Loss Expectation' in line:
                    loss_rate = float(line.split('=')[1].strip())
                elif 'Log-likelihood' in line:
                    log_likelihood = float(line.split('=')[1].strip())
        
        if gain_rate is None or loss_rate is None:
            print("  ✗ Could not parse results")
            return None
        
        print(f"\nResults:")
        print(f"  Gain rate: {gain_rate:.6f}")
        print(f"  Loss rate: {loss_rate:.6f}")
        print(f"  π₁: {gain_rate/(gain_rate + loss_rate):.6f}")
        if log_likelihood:
            print(f"  Log-likelihood: {log_likelihood:.2f}")
        print(f"  Runtime: {runtime:.2f}s")
        
        # Save results
        results_file = output_dir / "results.txt"
        with open(results_file, 'w') as f:
            f.write(f"GLOOME Results ({label})\n")
            f.write(f"=" * 70 + "\n\n")
            f.write(f"Dataset: {len(strain_names)} strains × {len(gene_names):,} genes\n\n")
            f.write(f"Gain rate: {gain_rate:.6f}\n")
            f.write(f"Loss rate: {loss_rate:.6f}\n")
            f.write(f"π₁: {gain_rate/(gain_rate + loss_rate):.6f}\n")
            if log_likelihood:
                f.write(f"Log-likelihood: {log_likelihood:.2f}\n")
            f.write(f"Runtime: {runtime:.2f}s\n")
        
        return {
            'gain': gain_rate,
            'loss': loss_rate,
            'pi1': gain_rate / (gain_rate + loss_rate),
            'll': log_likelihood,
            'time': runtime,
        }
        
    except subprocess.TimeoutExpired:
        print("  ✗ GLOOME timed out (>30 min)")
        return None
    except Exception as e:
        print(f"  ✗ GLOOME error: {e}")
        return None


def run_gloome_own_tree(pam, strain_names, gene_names, output_dir: Path):
    """Run GLOOME with its own tree inference."""
    print("\n" + "=" * 70)
    print("GLOOME (own tree)")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not shutil.which('gainLoss'):
        print("  ✗ GLOOME not available")
        return None
    
    print(f"\nDataset: {len(strain_names)} strains × {len(gene_names):,} genes")
    
    # Create sequence file
    seq_file = output_dir / 'sequences.fa'
    print(f"Writing sequences...")
    with open(seq_file, 'w') as f:
        for i, taxon in enumerate(strain_names):
            # Clean taxon name for GLOOME's tree inference
            clean_name = taxon.replace('_', '').replace('-', '')
            f.write(f'>{clean_name}\n')
            sequence = ''.join([str(int(pam[i, j])) for j in range(len(gene_names))])
            f.write(sequence + '\n')
    
    # Create parameter file WITHOUT tree
    param_file = output_dir / 'params.txt'
    with open(param_file, 'w') as f:
        f.write(f'_seqFile {seq_file.absolute()}\n')
        f.write(f'_outDir {output_dir.absolute()}\n')
        f.write('_logFile gainLoss.log\n')
        f.write('_gainLossDist 1\n')
        f.write('_numberOfGainCategories 1\n')
        f.write('_numberOfLossCategories 1\n')
        f.write('_maxNumOfIterationsModel 10\n')
        f.write('_performOptimizations 1\n')
    
    print(f"Running GLOOME (will infer tree)...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['gainLoss', str(param_file.absolute())],
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=str(output_dir)
        )
        runtime = time.time() - start_time
        
        if result.returncode != 0:
            print(f"  ✗ GLOOME failed (rc={result.returncode})")
            print(f"  STDERR: {result.stderr[:500]}")
            return None
        
        print(f"  ✓ GLOOME completed in {runtime:.2f}s")
        
        # Parse results
        params_file = output_dir / 'EstimatedParameters.txt'
        if not params_file.exists():
            print("  ✗ Output file not found")
            return None
        
        gain_rate = None
        loss_rate = None
        log_likelihood = None
        
        with open(params_file) as f:
            for line in f:
                if 'Gain Expectation' in line:
                    gain_rate = float(line.split('=')[1].strip())
                elif 'Loss Expectation' in line:
                    loss_rate = float(line.split('=')[1].strip())
                elif 'Log-likelihood' in line:
                    log_likelihood = float(line.split('=')[1].strip())
        
        if gain_rate is None or loss_rate is None:
            print("  ✗ Could not parse results")
            return None
        
        print(f"\nResults:")
        print(f"  Gain rate: {gain_rate:.6f}")
        print(f"  Loss rate: {loss_rate:.6f}")
        print(f"  π₁: {gain_rate/(gain_rate + loss_rate):.6f}")
        if log_likelihood:
            print(f"  Log-likelihood: {log_likelihood:.2f}")
        print(f"  Runtime: {runtime:.2f}s")
        
        return {
            'gain': gain_rate,
            'loss': loss_rate,
            'pi1': gain_rate / (gain_rate + loss_rate),
            'll': log_likelihood,
            'time': runtime,
        }
        
    except subprocess.TimeoutExpired:
        print("  ✗ GLOOME timed out")
        return None
    except Exception as e:
        print(f"  ✗ GLOOME error: {e}")
        return None


def compare_results(gc, gloome_gc_tree, gloome_own_tree):
    """Compare all three results."""
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    tools = []
    if gc:
        tools.append(('GeneContent', gc))
    if gloome_gc_tree:
        tools.append(('GLOOME (GC tree)', gloome_gc_tree))
    if gloome_own_tree:
        tools.append(('GLOOME (own tree)', gloome_own_tree))
    
    if len(tools) < 2:
        print("\n⚠ Need at least 2 tools to compare")
        return
    
    print(f"\n{'Tool':<20} {'Gain (λ)':<12} {'Loss (μ)':<12} {'π₁':<12} {'Log-L':<15} {'Time':<10}")
    print("-" * 85)
    for name, res in tools:
        ll_str = f"{res['ll']:.2f}" if res.get('ll') else "N/A"
        print(f"{name:<20} {res['gain']:<12.6f} {res['loss']:<12.6f} {res['pi1']:<12.6f} {ll_str:<15} {res['time']:<10.2f}s")
    
    # Detailed comparison
    print("\n" + "-" * 85)
    print("Agreement Analysis")
    print("-" * 85)
    
    # π₁ agreement
    pi1_vals = [res['pi1'] for _, res in tools]
    pi1_mean = np.mean(pi1_vals)
    pi1_range = max(pi1_vals) - min(pi1_vals)
    print(f"\nEquilibrium frequency (π₁):")
    print(f"  Mean: {pi1_mean:.6f}")
    print(f"  Range: {pi1_range:.6f} ({pi1_range/pi1_mean*100:.1f}% of mean)")
    
    if gc and gloome_gc_tree:
        pi1_diff = abs(gc['pi1'] - gloome_gc_tree['pi1'])
        print(f"  GeneContent vs GLOOME (same tree): {pi1_diff:.6f} ({pi1_diff/pi1_mean*100:.1f}%)")
    
    if gloome_gc_tree and gloome_own_tree:
        pi1_diff = abs(gloome_gc_tree['pi1'] - gloome_own_tree['pi1'])
        print(f"  GLOOME (GC tree) vs GLOOME (own tree): {pi1_diff:.6f} ({pi1_diff/pi1_mean*100:.1f}%)")
    
    # Loss/Gain ratio
    print(f"\nLoss/Gain ratio (μ/λ):")
    for name, res in tools:
        ratio = res['loss'] / res['gain']
        print(f"  {name:<20} {ratio:.2f}x")
    
    # Performance
    print(f"\nPerformance:")
    if gc:
        print(f"  GeneContent: {gc['time']:.2f}s")
        if gloome_gc_tree:
            speedup = gloome_gc_tree['time'] / gc['time']
            print(f"  vs GLOOME (GC tree): {speedup:.1f}x faster")
        if gloome_own_tree:
            speedup = gloome_own_tree['time'] / gc['time']
            print(f"  vs GLOOME (own tree): {speedup:.1f}x faster")
    
    if gloome_gc_tree and gloome_own_tree:
        speedup = gloome_own_tree['time'] / gloome_gc_tree['time']
        print(f"  GLOOME tree inference overhead: {speedup:.1f}x")


def main():
    print("=" * 70)
    print("ALL GENES SUBSET COMPARISON")
    print("=" * 70)
    print("\nTest: 25 strains × ALL 25,420 genes")
    print("Compare: GeneContent vs GLOOME (GC tree) vs GLOOME (own tree)")
    
    # Load full data
    data_dir = Path("data/ecoli_real")
    pam_file = data_dir / "Supplementary File 2A.txt"
    
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    print("\nLoading full E. coli dataset...")
    df = pd.read_csv(pam_file, sep='\t', index_col=0)
    pam = df.values.T.astype(int)
    strain_names = df.columns.tolist()
    gene_names = df.index.tolist()
    
    print(f"  Full dataset: {len(strain_names):,} strains × {len(gene_names):,} genes")
    
    # Create subset: 25 strains × ALL genes
    print("\nCreating subset...")
    np.random.seed(42)
    n_strains = 25
    strain_idx = np.random.choice(len(strain_names), n_strains, replace=False)
    
    pam_subset = pam[strain_idx, :]  # Keep ALL genes
    strain_subset = [strain_names[i] for i in strain_idx]
    gene_subset = gene_names  # All genes
    
    print(f"  Subset: {len(strain_subset)} strains × {len(gene_subset):,} genes")
    
    # Summary stats
    genes_per_strain = pam_subset.sum(axis=1)
    gene_freq = pam_subset.sum(axis=0) / pam_subset.shape[0]
    
    print(f"\n  Genes per strain: {genes_per_strain.mean():.0f} ± {genes_per_strain.std():.0f}")
    print(f"  Core genes (100%): {(gene_freq == 1.0).sum():,}")
    print(f"  Rare genes (<20%): {(gene_freq < 0.2).sum():,}")
    
    # Run analyses
    output_base = Path("results/all_genes_25_strains")
    
    # 1. GeneContent
    gc_output = output_base / "genecontent"
    gc_results = run_genecontent(pam_subset, strain_subset, gene_subset, gc_output)
    
    # 2. GLOOME with GeneContent tree
    gloome_gc_output = output_base / "gloome_gc_tree"
    gloome_gc_results = run_gloome_with_tree(
        pam_subset, strain_subset, gene_subset,
        gc_results['tree_file'], gloome_gc_output, "GC tree"
    )
    
    # 3. GLOOME with own tree
    gloome_own_output = output_base / "gloome_own_tree"
    gloome_own_results = run_gloome_own_tree(pam_subset, strain_subset, gene_subset, gloome_own_output)
    
    # Compare
    compare_results(gc_results, gloome_gc_results, gloome_own_results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_base}")


if __name__ == "__main__":
    main()
