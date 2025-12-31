#!/usr/bin/env python3
"""
Compare GeneContent (Rust) vs GLOOME on real E. coli ST131 data.

This script:
1. Loads real pangenome data (tree + gene presence/absence)
2. Runs GeneContent analysis (with Rust backend)
3. Runs GLOOME analysis
4. Compares results (rates, runtime, interpretability)
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

# Add persiste to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persiste.core.trees import load_tree
from persiste.plugins.genecontent.inference.gene_inference import GeneContentData
from persiste.plugins.genecontent.analyses.standard_analyses import GeneContentAnalysis


def load_pangenome_data(data_dir: Path):
    """Load pangenome data from directory."""
    print("=" * 70)
    print("LOADING PANGENOME DATA")
    print("=" * 70)
    
    # Load tree
    tree_file = data_dir / "core_genome_tree.nwk"
    if not tree_file.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_file}")
    
    print(f"\nLoading tree: {tree_file}")
    tree = load_tree(str(tree_file))
    print(f"  Taxa: {tree.n_tips}")
    print(f"  Total nodes: {tree.n_nodes}")
    
    # Load presence/absence matrix
    pa_file = data_dir / "gene_presence_absence.csv"
    if not pa_file.exists():
        raise FileNotFoundError(f"Presence/absence file not found: {pa_file}")
    
    print(f"\nLoading gene presence/absence: {pa_file}")
    df = pd.read_csv(pa_file, index_col=0)
    
    # Convert to numpy array (genes x strains)
    presence_matrix = df.values.astype(int)
    gene_names = df.index.tolist()
    strain_names = df.columns.tolist()
    
    # Transpose to (strains x genes) for GeneContentData
    presence_matrix = presence_matrix.T
    
    print(f"  Strains: {len(strain_names)}")
    print(f"  Genes: {len(gene_names)}")
    print(f"  Matrix shape: {presence_matrix.shape}")
    
    # Verify strain names match tree tips
    tree_tips = [tree.nodes[i].name for i in tree.tip_indices]
    if set(strain_names) != set(tree_tips):
        print("\n⚠ Warning: Strain names don't match tree tips exactly")
        print(f"  In matrix but not tree: {set(strain_names) - set(tree_tips)}")
        print(f"  In tree but not matrix: {set(tree_tips) - set(strain_names)}")
    
    # Create GeneContentData
    data = GeneContentData(
        tree=tree,
        presence_matrix=presence_matrix,
        taxon_names=strain_names,
        family_names=gene_names
    )
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    
    gene_freq = presence_matrix.sum(axis=0) / len(strain_names)
    print(f"\nGene frequency distribution:")
    print(f"  Core (100%): {(gene_freq == 1.0).sum()}")
    print(f"  Common (50-99%): {((gene_freq >= 0.5) & (gene_freq < 1.0)).sum()}")
    print(f"  Intermediate (20-49%): {((gene_freq >= 0.2) & (gene_freq < 0.5)).sum()}")
    print(f"  Rare (<20%): {(gene_freq < 0.2).sum()}")
    
    genes_per_strain = presence_matrix.sum(axis=1)
    print(f"\nGenes per strain:")
    print(f"  Mean: {genes_per_strain.mean():.0f} ± {genes_per_strain.std():.0f}")
    print(f"  Range: {genes_per_strain.min()}-{genes_per_strain.max()}")
    
    return data, df


def run_genecontent_analysis(data: GeneContentData, output_dir: Path):
    """Run GeneContent analysis with Rust backend."""
    print("\n" + "=" * 70)
    print("GENECONTENT ANALYSIS (RUST BACKEND)")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create analysis object
    analysis = GeneContentAnalysis(data)
    
    # Run global rate estimation
    print("\n1. Estimating global gain/loss rates...")
    start_time = time.time()
    result = analysis.global_rates(verbose=True)
    runtime = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Gain rate: {result.gain_rate:.4f}")
    print(f"  Loss rate: {result.loss_rate:.4f}")
    print(f"  π₁ (equilibrium frequency): {result.gain_rate/(result.gain_rate + result.loss_rate):.4f}")
    print(f"  Log-likelihood: {result.log_likelihood:.2f}")
    print(f"  Runtime: {runtime:.2f}s")
    
    # Save results
    results_file = output_dir / "genecontent_results.txt"
    with open(results_file, 'w') as f:
        f.write("GeneContent Analysis Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Gain rate: {result.gain_rate:.6f}\n")
        f.write(f"Loss rate: {result.loss_rate:.6f}\n")
        f.write(f"π₁: {result.gain_rate/(result.gain_rate + result.loss_rate):.6f}\n")
        f.write(f"Log-likelihood: {result.log_likelihood:.2f}\n")
        f.write(f"Runtime: {runtime:.2f}s\n")
    
    print(f"\n✓ Saved results: {results_file}")
    
    return {
        'gain_rate': result.gain_rate,
        'loss_rate': result.loss_rate,
        'pi1': result.gain_rate / (result.gain_rate + result.loss_rate),
        'log_likelihood': result.log_likelihood,
        'runtime': runtime
    }


def run_gloome_analysis(data: GeneContentData, df: pd.DataFrame, output_dir: Path):
    """Run GLOOME analysis."""
    print("\n" + "=" * 70)
    print("GLOOME ANALYSIS")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if GLOOME is available
    import shutil
    if not shutil.which('gainLoss'):
        print("\n⚠ GLOOME (gainLoss) not found in PATH")
        print("Skipping GLOOME analysis")
        return None
    
    # Create tree file
    tree_file = output_dir / 'tree.nwk'
    from persiste.plugins.genecontent.analyses.validation.tool_comparison_validation import tree_to_newick
    with open(tree_file, 'w') as f:
        f.write(tree_to_newick(data.tree))
        f.write('\n')
    
    # Create sequence file (FASTA with binary 0/1)
    seq_file = output_dir / 'sequences.fa'
    with open(seq_file, 'w') as f:
        for tip_idx, taxon in enumerate(data.taxon_names):
            f.write(f'>{taxon}\n')
            sequence = ''.join([str(int(data.presence_matrix[tip_idx, fam_idx]))
                               for fam_idx in range(data.presence_matrix.shape[1])])
            f.write(sequence + '\n')
    
    # Create parameter file
    param_file = output_dir / 'params.txt'
    with open(param_file, 'w') as f:
        f.write(f'_seqFile {seq_file.absolute()}\n')
        f.write(f'_treeFile {tree_file.absolute()}\n')
        f.write(f'_outDir {output_dir.absolute()}\n')
        f.write('_logFile gainLoss.log\n')
        f.write('_gainLossDist 1\n')
        f.write('_numberOfGainCategories 1\n')
        f.write('_numberOfLossCategories 1\n')
        f.write('_maxNumOfIterationsModel 10\n')
        f.write('_performOptimizations 1\n')
    
    print("\nRunning GLOOME...")
    print(f"  Tree: {tree_file}")
    print(f"  Sequences: {seq_file}")
    print(f"  Parameters: {param_file}")
    
    # Run GLOOME
    start_time = time.time()
    try:
        result = subprocess.run(
            ['gainLoss', str(param_file.absolute())],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(output_dir)
        )
        runtime = time.time() - start_time
        
        if result.returncode != 0:
            print(f"\n✗ GLOOME failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr[:500]}")
            return None
        
        # Parse results
        params_file = output_dir / 'EstimatedParameters.txt'
        if not params_file.exists():
            print(f"\n✗ GLOOME output file not found: {params_file}")
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
            print("\n✗ Could not parse GLOOME results")
            return None
        
        print(f"\nResults:")
        print(f"  Gain rate: {gain_rate:.4f}")
        print(f"  Loss rate: {loss_rate:.4f}")
        print(f"  π₁ (equilibrium frequency): {gain_rate/(gain_rate + loss_rate):.4f}")
        if log_likelihood:
            print(f"  Log-likelihood: {log_likelihood:.2f}")
        print(f"  Runtime: {runtime:.2f}s")
        
        # Save results
        results_file = output_dir / "gloome_results.txt"
        with open(results_file, 'w') as f:
            f.write("GLOOME Analysis Results\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Gain rate: {gain_rate:.6f}\n")
            f.write(f"Loss rate: {loss_rate:.6f}\n")
            f.write(f"π₁: {gain_rate/(gain_rate + loss_rate):.6f}\n")
            if log_likelihood:
                f.write(f"Log-likelihood: {log_likelihood:.2f}\n")
            f.write(f"Runtime: {runtime:.2f}s\n")
        
        print(f"\n✓ Saved results: {results_file}")
        
        return {
            'gain_rate': gain_rate,
            'loss_rate': loss_rate,
            'pi1': gain_rate / (gain_rate + loss_rate),
            'log_likelihood': log_likelihood,
            'runtime': runtime
        }
        
    except subprocess.TimeoutExpired:
        print("\n✗ GLOOME timed out (>300s)")
        return None
    except Exception as e:
        print(f"\n✗ GLOOME error: {e}")
        return None


def compare_results(gc_results: Dict, gloome_results: Optional[Dict]):
    """Compare results from both tools."""
    print("\n" + "=" * 70)
    print("TOOL COMPARISON")
    print("=" * 70)
    
    if gloome_results is None:
        print("\nGLOOME results not available - showing GeneContent results only")
        return
    
    print("\n" + "-" * 70)
    print("Parameter Estimates")
    print("-" * 70)
    
    print(f"\n{'Metric':<20} {'GeneContent':<15} {'GLOOME':<15} {'Ratio':<10}")
    print("-" * 70)
    
    gc_gain = gc_results['gain_rate']
    gl_gain = gloome_results['gain_rate']
    print(f"{'Gain rate':<20} {gc_gain:<15.4f} {gl_gain:<15.4f} {gc_gain/gl_gain:<10.2f}x")
    
    gc_loss = gc_results['loss_rate']
    gl_loss = gloome_results['loss_rate']
    print(f"{'Loss rate':<20} {gc_loss:<15.4f} {gl_loss:<15.4f} {gc_loss/gl_loss:<10.2f}x")
    
    gc_pi1 = gc_results['pi1']
    gl_pi1 = gloome_results['pi1']
    print(f"{'π₁ (equilibrium)':<20} {gc_pi1:<15.4f} {gl_pi1:<15.4f} {abs(gc_pi1-gl_pi1):<10.4f} diff")
    
    print("\n" + "-" * 70)
    print("Performance")
    print("-" * 70)
    
    gc_time = gc_results['runtime']
    gl_time = gloome_results['runtime']
    print(f"\n{'Tool':<20} {'Runtime (s)':<15} {'Speedup':<15}")
    print("-" * 70)
    print(f"{'GeneContent':<20} {gc_time:<15.2f} {gl_time/gc_time:<15.2f}x faster")
    print(f"{'GLOOME':<20} {gl_time:<15.2f} {'(baseline)':<15}")
    
    if gc_results['log_likelihood'] and gloome_results['log_likelihood']:
        print("\n" + "-" * 70)
        print("Model Fit")
        print("-" * 70)
        print(f"\n{'Tool':<20} {'Log-likelihood':<15}")
        print("-" * 70)
        print(f"{'GeneContent':<20} {gc_results['log_likelihood']:<15.2f}")
        print(f"{'GLOOME':<20} {gloome_results['log_likelihood']:<15.2f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    print("\nRate Scaling:")
    rate_ratio = gc_gain / gl_gain
    if abs(rate_ratio - 1.0) > 0.5:
        print(f"  ⚠ Large difference in absolute rates ({rate_ratio:.1f}x)")
        print(f"    This may indicate different rate parameterizations")
        print(f"    or different branch length scales between tools")
    else:
        print(f"  ✓ Similar absolute rates (ratio: {rate_ratio:.2f}x)")
    
    print("\nEquilibrium Frequency (π₁):")
    pi1_diff = abs(gc_pi1 - gl_pi1)
    if pi1_diff < 0.05:
        print(f"  ✓ Both tools agree on equilibrium frequency (diff: {pi1_diff:.4f})")
        print(f"    This is the most identifiable parameter")
    else:
        print(f"  ⚠ Equilibrium frequencies differ (diff: {pi1_diff:.4f})")
    
    print("\nPerformance:")
    speedup = gl_time / gc_time
    if speedup > 1.5:
        print(f"  ✓ GeneContent is {speedup:.1f}x faster than GLOOME")
    elif speedup < 0.67:
        print(f"  ⚠ GLOOME is {1/speedup:.1f}x faster than GeneContent")
    else:
        print(f"  ≈ Similar performance (ratio: {speedup:.2f}x)")
    
    print("\nGeneContent Advantages:")
    print("  ✓ Explicit retention strength parameter (θ)")
    print("  ✓ Statistical significance testing (LRT with p-values)")
    print("  ✓ Can test specific gene sets")
    print("  ✓ Direct biological interpretation")
    print("  ✓ Rust-accelerated (100x faster than NumPy)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_tools_real_data.py <data_dir>")
        print("\nExample:")
        print("  python compare_tools_real_data.py data/ecoli_st131")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Load data
    data, df = load_pangenome_data(data_dir)
    
    # Create output directory
    output_dir = Path("results/tool_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run GeneContent
    gc_output = output_dir / "genecontent"
    gc_results = run_genecontent_analysis(data, gc_output)
    
    # Run GLOOME
    gloome_output = output_dir / "gloome"
    gloome_results = run_gloome_analysis(data, df, gloome_output)
    
    # Compare results
    compare_results(gc_results, gloome_results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  GeneContent: {gc_output}")
    if gloome_results:
        print(f"  GLOOME: {gloome_output}")


if __name__ == "__main__":
    main()
