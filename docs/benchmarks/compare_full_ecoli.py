#!/usr/bin/env python3
"""
Compare GeneContent (Rust) vs GLOOME on full E. coli dataset.

Dataset: 25,420 genes × 1,324 E. coli strains
Source: BMC Genomics 2022 supplementary file 2A
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persiste.plugins.genecontent import pam_interface
from persiste.plugins.genecontent.tree_inference import infer_tree_from_pam
from persiste.plugins.genecontent.analyses.validation.tool_comparison_validation import tree_to_newick


def load_full_ecoli_pam(pam_file: Path):
    """Load the full E. coli PAM."""
    print("Loading full E. coli PAM...")
    print(f"  File: {pam_file}")
    
    df = pd.read_csv(pam_file, sep='\t', index_col=0)
    
    print(f"  Genes: {df.shape[0]:,}")
    print(f"  Strains: {df.shape[1]:,}")
    
    # Convert to (strains × genes)
    pam = df.values.T.astype(int)
    strain_names = df.columns.tolist()
    gene_names = df.index.tolist()
    
    # Summary stats
    genes_per_strain = pam.sum(axis=1)
    gene_freq = pam.sum(axis=0) / pam.shape[0]
    
    print(f"\n  Genes per strain: {genes_per_strain.mean():.0f} ± {genes_per_strain.std():.0f}")
    print(f"  Core genes (100%): {(gene_freq == 1.0).sum():,}")
    print(f"  Rare genes (<20%): {(gene_freq < 0.2).sum():,}")
    
    return pam, strain_names, gene_names


def run_genecontent_full(pam, strain_names, gene_names, output_dir: Path):
    """Run GeneContent on full dataset."""
    print("\n" + "=" * 70)
    print("GENECONTENT ANALYSIS (FULL DATASET)")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Step 1: Infer tree
    print("\nStep 1: Inferring tree from PAM...")
    print(f"  Method: Jaccard distance + UPGMA")
    print(f"  Strains: {len(strain_names):,}")
    
    tree_start = time.time()
    tree, metadata = infer_tree_from_pam(pam, strain_names, method="jaccard_upgma")
    tree_time = time.time() - tree_start
    
    print(f"  ✓ Tree inferred in {tree_time:.2f}s")
    
    # Save tree
    tree_file = output_dir / "tree.nwk"
    with open(tree_file, 'w') as f:
        f.write(tree_to_newick(tree))
        f.write('\n')
    print(f"  Saved: {tree_file}")
    
    # Step 2: Fit model
    print("\nStep 2: Fitting GeneContent model...")
    print(f"  Genes: {len(gene_names):,}")
    print(f"  Using Rust acceleration")
    
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
    print(f"\n  Total time: {total_time:.2f}s")
    
    # Save results
    results_file = output_dir / "genecontent_results.txt"
    with open(results_file, 'w') as f:
        f.write("GeneContent Analysis - Full E. coli Dataset\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset:\n")
        f.write(f"  Strains: {len(strain_names):,}\n")
        f.write(f"  Genes: {len(gene_names):,}\n\n")
        f.write(f"Tree:\n")
        f.write(f"  Method: {metadata.method}\n")
        f.write(f"  Inference time: {tree_time:.2f}s\n\n")
        f.write(f"Results:\n")
        f.write(f"  Gain rate (λ): {result.gain_rate:.6f}\n")
        f.write(f"  Loss rate (μ): {result.loss_rate:.6f}\n")
        f.write(f"  π₁: {result.equilibrium_frequency:.6f}\n")
        f.write(f"  Log-likelihood: {result.log_likelihood:.2f}\n")
        f.write(f"  Fitting time: {fit_time:.2f}s\n")
        f.write(f"  Total time: {total_time:.2f}s\n")
    
    print(f"  Saved: {results_file}")
    
    return {
        'gain_rate': result.gain_rate,
        'loss_rate': result.loss_rate,
        'pi1': result.equilibrium_frequency,
        'log_likelihood': result.log_likelihood,
        'tree_time': tree_time,
        'fit_time': fit_time,
        'total_time': total_time,
        'tree_file': tree_file,
    }


def run_gloome_full(pam, strain_names, gene_names, tree_file: Path, output_dir: Path):
    """Run GLOOME on full dataset."""
    print("\n" + "=" * 70)
    print("GLOOME ANALYSIS (FULL DATASET)")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if GLOOME is available
    import shutil
    if not shutil.which('gainLoss'):
        print("\n⚠ GLOOME (gainLoss) not found in PATH")
        return None
    
    print("\nPreparing GLOOME input files...")
    
    # Create sequence file (FASTA with binary 0/1)
    seq_file = output_dir / 'sequences.fa'
    print(f"  Writing sequences: {seq_file}")
    with open(seq_file, 'w') as f:
        for i, taxon in enumerate(strain_names):
            f.write(f'>{taxon}\n')
            sequence = ''.join([str(int(pam[i, j])) for j in range(len(gene_names))])
            f.write(sequence + '\n')
    
    # Copy tree file
    gloome_tree = output_dir / 'tree.nwk'
    import shutil as sh
    sh.copy(tree_file, gloome_tree)
    print(f"  Using tree: {gloome_tree}")
    
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
    
    print(f"\nRunning GLOOME...")
    print(f"  Strains: {len(strain_names):,}")
    print(f"  Genes: {len(gene_names):,}")
    print(f"  (This may take several minutes...)")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            ['gainLoss', str(param_file.absolute())],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(output_dir)
        )
        runtime = time.time() - start_time
        
        if result.returncode != 0:
            print(f"\n✗ GLOOME failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr[:500]}")
            return None
        
        print(f"  ✓ GLOOME completed in {runtime:.2f}s")
        
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
        
        # Save results
        results_file = output_dir / "gloome_results.txt"
        with open(results_file, 'w') as f:
            f.write("GLOOME Analysis - Full E. coli Dataset\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Dataset:\n")
            f.write(f"  Strains: {len(strain_names):,}\n")
            f.write(f"  Genes: {len(gene_names):,}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Gain rate: {gain_rate:.6f}\n")
            f.write(f"  Loss rate: {loss_rate:.6f}\n")
            f.write(f"  π₁: {gain_rate/(gain_rate + loss_rate):.6f}\n")
            if log_likelihood:
                f.write(f"  Log-likelihood: {log_likelihood:.2f}\n")
            f.write(f"  Runtime: {runtime:.2f}s\n")
        
        print(f"  Saved: {results_file}")
        
        return {
            'gain_rate': gain_rate,
            'loss_rate': loss_rate,
            'pi1': gain_rate / (gain_rate + loss_rate),
            'log_likelihood': log_likelihood,
            'runtime': runtime,
        }
        
    except subprocess.TimeoutExpired:
        print("\n✗ GLOOME timed out (>1 hour)")
        return None
    except Exception as e:
        print(f"\n✗ GLOOME error: {e}")
        return None


def compare_results(gc_results, gloome_results):
    """Compare GeneContent and GLOOME results."""
    print("\n" + "=" * 70)
    print("COMPARISON: GENECONTENT vs GLOOME")
    print("=" * 70)
    
    if gloome_results is None:
        print("\nGLOOME results not available")
        return
    
    print("\n" + "-" * 70)
    print("Parameter Estimates")
    print("-" * 70)
    
    print(f"\n{'Metric':<25} {'GeneContent':<20} {'GLOOME':<20} {'Ratio':<15}")
    print("-" * 70)
    
    gc_gain = gc_results['gain_rate']
    gl_gain = gloome_results['gain_rate']
    print(f"{'Gain rate (λ)':<25} {gc_gain:<20.6f} {gl_gain:<20.6f} {gc_gain/gl_gain:<15.2f}x")
    
    gc_loss = gc_results['loss_rate']
    gl_loss = gloome_results['loss_rate']
    print(f"{'Loss rate (μ)':<25} {gc_loss:<20.6f} {gl_loss:<20.6f} {gc_loss/gl_loss:<15.2f}x")
    
    gc_pi1 = gc_results['pi1']
    gl_pi1 = gloome_results['pi1']
    print(f"{'π₁ (equilibrium)':<25} {gc_pi1:<20.6f} {gl_pi1:<20.6f} {abs(gc_pi1-gl_pi1):<15.6f} diff")
    
    gc_ratio = gc_loss / gc_gain
    gl_ratio = gl_loss / gl_gain
    print(f"{'Loss/Gain ratio':<25} {gc_ratio:<20.2f} {gl_ratio:<20.2f} {abs(gc_ratio-gl_ratio):<15.2f} diff")
    
    print("\n" + "-" * 70)
    print("Performance")
    print("-" * 70)
    
    gc_time = gc_results['total_time']
    gl_time = gloome_results['runtime']
    
    print(f"\n{'Tool':<25} {'Runtime':<20} {'Speedup':<15}")
    print("-" * 70)
    print(f"{'GeneContent':<25} {gc_time:<20.2f}s {gl_time/gc_time:<15.2f}x faster")
    print(f"{'GLOOME':<25} {gl_time:<20.2f}s {'(baseline)':<15}")
    
    print(f"\n  Breakdown (GeneContent):")
    print(f"    Tree inference: {gc_results['tree_time']:.2f}s")
    print(f"    Model fitting:  {gc_results['fit_time']:.2f}s")
    
    if gc_results['log_likelihood'] and gloome_results['log_likelihood']:
        print("\n" + "-" * 70)
        print("Model Fit")
        print("-" * 70)
        print(f"\n{'Tool':<25} {'Log-likelihood':<20}")
        print("-" * 70)
        print(f"{'GeneContent':<25} {gc_results['log_likelihood']:<20.2f}")
        print(f"{'GLOOME':<25} {gloome_results['log_likelihood']:<20.2f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    print("\nRate Scaling:")
    rate_ratio = gc_gain / gl_gain
    if abs(rate_ratio - 1.0) > 0.5:
        print(f"  ⚠ Large difference in absolute rates ({rate_ratio:.1f}x)")
        print(f"    Different rate parameterizations or branch length scales")
    else:
        print(f"  ✓ Similar absolute rates (ratio: {rate_ratio:.2f}x)")
    
    print("\nEquilibrium Frequency (π₁):")
    pi1_diff = abs(gc_pi1 - gl_pi1)
    if pi1_diff < 0.05:
        print(f"  ✓ Both tools agree on equilibrium (diff: {pi1_diff:.4f})")
        print(f"    This is the most identifiable parameter")
    else:
        print(f"  ⚠ Equilibrium frequencies differ (diff: {pi1_diff:.4f})")
    
    print("\nLoss/Gain Ratio:")
    ratio_diff = abs(gc_ratio - gl_ratio)
    if ratio_diff < 1.0:
        print(f"  ✓ Both tools agree on loss-dominated evolution")
        print(f"    GeneContent: {gc_ratio:.2f}x, GLOOME: {gl_ratio:.2f}x")
    else:
        print(f"  ⚠ Different loss/gain ratios (diff: {ratio_diff:.2f})")
    
    print("\nPerformance:")
    speedup = gl_time / gc_time
    if speedup > 2:
        print(f"  ✓ GeneContent is {speedup:.1f}x faster than GLOOME")
    else:
        print(f"  ≈ Similar performance (ratio: {speedup:.2f}x)")


def main():
    print("=" * 70)
    print("FULL E. COLI DATASET COMPARISON")
    print("=" * 70)
    print("\nGeneContent (Rust) vs GLOOME")
    print("Dataset: 25,420 genes × 1,324 strains")
    print("Source: BMC Genomics 2022")
    
    # Load data
    data_dir = Path("data/ecoli_real")
    pam_file = data_dir / "Supplementary File 2A.txt"
    
    if not pam_file.exists():
        print(f"\n✗ PAM file not found: {pam_file}")
        return
    
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70 + "\n")
    
    pam, strain_names, gene_names = load_full_ecoli_pam(pam_file)
    
    # Create output directory
    output_dir = Path("results/ecoli_full_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run GeneContent
    gc_output = output_dir / "genecontent"
    gc_results = run_genecontent_full(pam, strain_names, gene_names, gc_output)
    
    # Run GLOOME
    gloome_output = output_dir / "gloome"
    gloome_results = run_gloome_full(
        pam, strain_names, gene_names,
        gc_results['tree_file'], gloome_output
    )
    
    # Compare
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
