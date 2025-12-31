#!/usr/bin/env python3
"""
Test GLOOME with progressively larger datasets to find failure threshold.

Tests both:
1. GLOOME with its own tree inference
2. GLOOME with GeneContent-inferred tree
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time
import subprocess
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persiste.plugins.genecontent import pam_interface
from persiste.plugins.genecontent.tree_inference import infer_tree_from_pam
from persiste.plugins.genecontent.analyses.validation.tool_comparison_validation import tree_to_newick


def create_small_subset(pam, strain_names, gene_names, n_strains, n_genes, seed=42):
    """Create a small random subset."""
    np.random.seed(seed)
    strain_idx = np.random.choice(len(strain_names), min(n_strains, len(strain_names)), replace=False)
    gene_idx = np.random.choice(len(gene_names), min(n_genes, len(gene_names)), replace=False)
    
    pam_subset = pam[strain_idx, :][:, gene_idx]
    strain_subset = [strain_names[i] for i in strain_idx]
    gene_subset = [gene_names[i] for i in gene_idx]
    
    return pam_subset, strain_subset, gene_subset


def run_gloome_with_own_tree(pam, strain_names, gene_names, output_dir: Path):
    """Run GLOOME with its own tree inference."""
    print("\n" + "=" * 70)
    print("GLOOME WITH INTERNAL TREE INFERENCE")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not shutil.which('gainLoss'):
        print("\n⚠ GLOOME not available")
        return None
    
    # Create sequence file
    seq_file = output_dir / 'sequences.fa'
    print(f"\nWriting sequences: {seq_file}")
    with open(seq_file, 'w') as f:
        for i, taxon in enumerate(strain_names):
            # Clean taxon name - remove special characters
            clean_name = taxon.replace('_', '').replace('-', '')
            f.write(f'>{clean_name}\n')
            sequence = ''.join([str(int(pam[i, j])) for j in range(len(gene_names))])
            f.write(sequence + '\n')
    
    # Create parameter file WITHOUT tree (let GLOOME infer it)
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
    print(f"  Strains: {len(strain_names)}")
    print(f"  Genes: {len(gene_names)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            ['gainLoss', str(param_file.absolute())],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(output_dir)
        )
        runtime = time.time() - start_time
        
        if result.returncode != 0:
            print(f"\n✗ GLOOME failed (rc={result.returncode})")
            print(f"STDERR: {result.stderr[:500]}")
            
            # Check log for more info
            log_file = output_dir / 'gainLoss.log'
            if log_file.exists():
                with open(log_file) as f:
                    log_content = f.read()
                    print(f"\nLast 20 lines of log:")
                    print('\n'.join(log_content.split('\n')[-20:]))
            return None
        
        print(f"  ✓ GLOOME completed in {runtime:.2f}s")
        
        # Parse results
        params_file = output_dir / 'EstimatedParameters.txt'
        if not params_file.exists():
            print("\n✗ Output file not found")
            return None
        
        gain_rate = None
        loss_rate = None
        
        with open(params_file) as f:
            for line in f:
                if 'Gain Expectation' in line:
                    gain_rate = float(line.split('=')[1].strip())
                elif 'Loss Expectation' in line:
                    loss_rate = float(line.split('=')[1].strip())
        
        if gain_rate is None or loss_rate is None:
            print("\n✗ Could not parse results")
            return None
        
        print(f"\nResults:")
        print(f"  Gain rate: {gain_rate:.6f}")
        print(f"  Loss rate: {loss_rate:.6f}")
        print(f"  π₁: {gain_rate/(gain_rate + loss_rate):.6f}")
        print(f"  Runtime: {runtime:.2f}s")
        
        return {
            'gain_rate': gain_rate,
            'loss_rate': loss_rate,
            'pi1': gain_rate / (gain_rate + loss_rate),
            'runtime': runtime,
        }
        
    except subprocess.TimeoutExpired:
        print("\n✗ GLOOME timed out")
        return None
    except Exception as e:
        print(f"\n✗ GLOOME error: {e}")
        return None


def run_gloome_with_gc_tree(pam, strain_names, gene_names, tree_file: Path, output_dir: Path):
    """Run GLOOME with GeneContent-inferred tree."""
    print("\n" + "=" * 70)
    print("GLOOME WITH GENECONTENT TREE")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not shutil.which('gainLoss'):
        print("\n⚠ GLOOME not available")
        return None
    
    # Read tree and get taxon names from it
    with open(tree_file) as f:
        tree_newick = f.read().strip()
    
    # Create sequence file with EXACT same names as in tree
    seq_file = output_dir / 'sequences.fa'
    print(f"\nWriting sequences: {seq_file}")
    with open(seq_file, 'w') as f:
        for i, taxon in enumerate(strain_names):
            f.write(f'>{taxon}\n')
            sequence = ''.join([str(int(pam[i, j])) for j in range(len(gene_names))])
            f.write(sequence + '\n')
    
    # Copy tree
    gloome_tree = output_dir / 'tree.nwk'
    shutil.copy(tree_file, gloome_tree)
    
    # Create parameter file WITH tree
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
    
    print(f"Running GLOOME (with provided tree)...")
    print(f"  Strains: {len(strain_names)}")
    print(f"  Genes: {len(gene_names)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            ['gainLoss', str(param_file.absolute())],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(output_dir)
        )
        runtime = time.time() - start_time
        
        if result.returncode != 0:
            print(f"\n✗ GLOOME failed (rc={result.returncode})")
            print(f"STDERR: {result.stderr[:500]}")
            
            log_file = output_dir / 'gainLoss.log'
            if log_file.exists():
                with open(log_file) as f:
                    log_content = f.read()
                    print(f"\nLast 20 lines of log:")
                    print('\n'.join(log_content.split('\n')[-20:]))
            return None
        
        print(f"  ✓ GLOOME completed in {runtime:.2f}s")
        
        # Parse results
        params_file = output_dir / 'EstimatedParameters.txt'
        if not params_file.exists():
            print("\n✗ Output file not found")
            return None
        
        gain_rate = None
        loss_rate = None
        
        with open(params_file) as f:
            for line in f:
                if 'Gain Expectation' in line:
                    gain_rate = float(line.split('=')[1].strip())
                elif 'Loss Expectation' in line:
                    loss_rate = float(line.split('=')[1].strip())
        
        if gain_rate is None or loss_rate is None:
            print("\n✗ Could not parse results")
            return None
        
        print(f"\nResults:")
        print(f"  Gain rate: {gain_rate:.6f}")
        print(f"  Loss rate: {loss_rate:.6f}")
        print(f"  π₁: {gain_rate/(gain_rate + loss_rate):.6f}")
        print(f"  Runtime: {runtime:.2f}s")
        
        return {
            'gain_rate': gain_rate,
            'loss_rate': loss_rate,
            'pi1': gain_rate / (gain_rate + loss_rate),
            'runtime': runtime,
        }
        
    except subprocess.TimeoutExpired:
        print("\n✗ GLOOME timed out")
        return None
    except Exception as e:
        print(f"\n✗ GLOOME error: {e}")
        return None


def run_genecontent(pam, strain_names, gene_names, output_dir: Path):
    """Run GeneContent analysis."""
    print("\n" + "=" * 70)
    print("GENECONTENT ANALYSIS")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nInferring tree...")
    tree_start = time.time()
    tree, metadata = infer_tree_from_pam(pam, strain_names, method="jaccard_upgma")
    tree_time = time.time() - tree_start
    print(f"  ✓ Tree inferred in {tree_time:.2f}s")
    
    # Save tree
    tree_file = output_dir / "tree.nwk"
    with open(tree_file, 'w') as f:
        f.write(tree_to_newick(tree))
        f.write('\n')
    
    print(f"Fitting model...")
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
    print(f"  Gain rate: {result.gain_rate:.6f}")
    print(f"  Loss rate: {result.loss_rate:.6f}")
    print(f"  π₁: {result.equilibrium_frequency:.6f}")
    print(f"  Total time: {total_time:.2f}s")
    
    return {
        'gain_rate': result.gain_rate,
        'loss_rate': result.loss_rate,
        'pi1': result.equilibrium_frequency,
        'total_time': total_time,
        'tree_file': tree_file,
    }


def test_size(pam, strain_names, gene_names, n_strains, n_genes, output_base: Path):
    """Test a specific dataset size."""
    print("\n" + "=" * 70)
    print(f"TESTING SIZE: {n_strains} strains × {n_genes} genes")
    print("=" * 70)
    
    # Create subset
    pam_subset, strain_subset, gene_subset = create_small_subset(
        pam, strain_names, gene_names, n_strains, n_genes
    )
    
    print(f"\nSubset created: {len(strain_subset)} × {len(gene_subset)}")
    
    output_dir = output_base / f"{n_strains}x{n_genes}"
    
    # Run GeneContent
    gc_output = output_dir / "genecontent"
    gc_results = run_genecontent(pam_subset, strain_subset, gene_subset, gc_output)
    
    # Run GLOOME with own tree
    gloome_own_output = output_dir / "gloome_own_tree"
    gloome_own_results = run_gloome_with_own_tree(
        pam_subset, strain_subset, gene_subset, gloome_own_output
    )
    
    # Run GLOOME with GC tree
    gloome_gc_output = output_dir / "gloome_gc_tree"
    gloome_gc_results = run_gloome_with_gc_tree(
        pam_subset, strain_subset, gene_subset,
        gc_results['tree_file'], gloome_gc_output
    )
    
    # Compare
    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)
    
    if gloome_own_results and gloome_gc_results:
        print(f"\n{'Tool':<30} {'Gain':<12} {'Loss':<12} {'π₁':<12} {'Time':<10}")
        print("-" * 70)
        print(f"{'GeneContent':<30} {gc_results['gain_rate']:<12.4f} {gc_results['loss_rate']:<12.4f} {gc_results['pi1']:<12.4f} {gc_results['total_time']:<10.2f}s")
        print(f"{'GLOOME (own tree)':<30} {gloome_own_results['gain_rate']:<12.4f} {gloome_own_results['loss_rate']:<12.4f} {gloome_own_results['pi1']:<12.4f} {gloome_own_results['runtime']:<10.2f}s")
        print(f"{'GLOOME (GC tree)':<30} {gloome_gc_results['gain_rate']:<12.4f} {gloome_gc_results['loss_rate']:<12.4f} {gloome_gc_results['pi1']:<12.4f} {gloome_gc_results['runtime']:<10.2f}s")
        
        return True  # Success
    else:
        print("\n⚠ GLOOME failed at this size")
        return False  # Failure


def main():
    print("=" * 70)
    print("GLOOME SCALING TEST")
    print("=" * 70)
    
    # Load data
    data_dir = Path("data/ecoli_real")
    pam_file = data_dir / "Supplementary File 2A.txt"
    
    print("\nLoading full dataset...")
    df = pd.read_csv(pam_file, sep='\t', index_col=0)
    pam = df.values.T.astype(int)
    strain_names = df.columns.tolist()
    gene_names = df.index.tolist()
    
    print(f"  Full dataset: {len(strain_names):,} × {len(gene_names):,}")
    
    output_base = Path("results/gloome_scaling")
    
    # Test progressively larger sizes
    test_sizes = [
        (20, 100),   # Tiny
        (50, 500),   # Small
        (100, 1000), # Medium
        (200, 2000), # Large
    ]
    
    for n_strains, n_genes in test_sizes:
        success = test_size(pam, strain_names, gene_names, n_strains, n_genes, output_base)
        if not success:
            print(f"\n✗ GLOOME failed at {n_strains} × {n_genes}")
            print("Stopping here.")
            break
    
    print("\n" + "=" * 70)
    print("SCALING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
