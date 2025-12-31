#!/usr/bin/env python3
"""
Test GLOOME scaling with increasing number of strains (all genes).

Tests: 100 strains, then 400 strains if successful.
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


def run_genecontent(pam, strain_names, gene_names, output_dir: Path):
    """Run GeneContent analysis."""
    print("\n" + "=" * 70)
    print("GENECONTENT")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nDataset: {len(strain_names)} strains × {len(gene_names):,} genes")
    
    tree_start = time.time()
    tree, metadata = infer_tree_from_pam(pam, strain_names, method="jaccard_upgma")
    tree_time = time.time() - tree_start
    print(f"  Tree: {tree_time:.2f}s")
    
    tree_file = output_dir / "tree.nwk"
    with open(tree_file, 'w') as f:
        f.write(tree_to_newick(tree))
        f.write('\n')
    
    fit_start = time.time()
    result = pam_interface.fit(
        pam=pam, tree=tree, taxon_names=strain_names,
        gene_names=gene_names, use_rust=True, verbose=False,
    )
    fit_time = time.time() - fit_start
    print(f"  Fit: {fit_time:.2f}s")
    
    total_time = tree_time + fit_time
    print(f"  Total: {total_time:.2f}s")
    print(f"  λ={result.gain_rate:.4f}, μ={result.loss_rate:.4f}, π₁={result.equilibrium_frequency:.4f}")
    
    return {
        'gain': result.gain_rate,
        'loss': result.loss_rate,
        'pi1': result.equilibrium_frequency,
        'time': total_time,
        'tree_file': tree_file,
    }


def run_gloome(pam, strain_names, gene_names, tree_file: Path, output_dir: Path):
    """Run GLOOME with provided tree."""
    print("\n" + "=" * 70)
    print("GLOOME (with GeneContent tree)")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not shutil.which('gainLoss'):
        print("  ✗ Not available")
        return None
    
    print(f"\nDataset: {len(strain_names)} strains × {len(gene_names):,} genes")
    
    seq_file = output_dir / 'sequences.fa'
    print("  Writing sequences...")
    with open(seq_file, 'w') as f:
        for i, taxon in enumerate(strain_names):
            f.write(f'>{taxon}\n')
            f.write(''.join([str(int(pam[i, j])) for j in range(len(gene_names))]) + '\n')
    
    gloome_tree = output_dir / 'tree.nwk'
    shutil.copy(tree_file, gloome_tree)
    
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
    
    print("  Running GLOOME...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['gainLoss', str(param_file.absolute())],
            capture_output=True, text=True, timeout=3600, cwd=str(output_dir)
        )
        runtime = time.time() - start_time
        
        if result.returncode != 0:
            print(f"  ✗ Failed (rc={result.returncode})")
            if result.returncode == -6:
                print("  Segmentation fault - dataset too large")
            return None
        
        print(f"  ✓ Completed in {runtime:.2f}s")
        
        params_file = output_dir / 'EstimatedParameters.txt'
        if not params_file.exists():
            print("  ✗ Output not found")
            return None
        
        gain, loss = None, None
        with open(params_file) as f:
            for line in f:
                if 'Gain Expectation' in line:
                    gain = float(line.split('=')[1].strip())
                elif 'Loss Expectation' in line:
                    loss = float(line.split('=')[1].strip())
        
        if gain is None or loss is None:
            print("  ✗ Could not parse")
            return None
        
        print(f"  λ={gain:.4f}, μ={loss:.4f}, π₁={gain/(gain+loss):.4f}")
        
        return {
            'gain': gain,
            'loss': loss,
            'pi1': gain/(gain+loss),
            'time': runtime,
        }
        
    except subprocess.TimeoutExpired:
        print("  ✗ Timed out (>1 hour)")
        return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def test_size(pam, strain_names, gene_names, n_strains, output_base: Path):
    """Test a specific number of strains."""
    print("\n" + "=" * 70)
    print(f"TESTING: {n_strains} strains × ALL genes")
    print("=" * 70)
    
    # Create subset
    np.random.seed(42)
    strain_idx = np.random.choice(len(strain_names), n_strains, replace=False)
    
    pam_subset = pam[strain_idx, :]
    strain_subset = [strain_names[i] for i in strain_idx]
    gene_subset = gene_names
    
    print(f"\nSubset: {len(strain_subset)} strains × {len(gene_subset):,} genes")
    
    genes_per_strain = pam_subset.sum(axis=1)
    gene_freq = pam_subset.sum(axis=0) / pam_subset.shape[0]
    
    print(f"  Genes per strain: {genes_per_strain.mean():.0f} ± {genes_per_strain.std():.0f}")
    print(f"  Core genes: {(gene_freq == 1.0).sum():,}")
    print(f"  Rare genes: {(gene_freq < 0.2).sum():,}")
    
    output_dir = output_base / f"{n_strains}_strains"
    
    # Run GeneContent
    gc_output = output_dir / "genecontent"
    gc_results = run_genecontent(pam_subset, strain_subset, gene_subset, gc_output)
    
    # Run GLOOME
    gloome_output = output_dir / "gloome"
    gloome_results = run_gloome(
        pam_subset, strain_subset, gene_subset,
        gc_results['tree_file'], gloome_output
    )
    
    # Compare
    if gloome_results:
        print("\n" + "-" * 70)
        print("COMPARISON")
        print("-" * 70)
        print(f"\n{'Tool':<20} {'Gain':<10} {'Loss':<10} {'π₁':<10} {'Time':<10}")
        print("-" * 70)
        print(f"{'GeneContent':<20} {gc_results['gain']:<10.4f} {gc_results['loss']:<10.4f} {gc_results['pi1']:<10.4f} {gc_results['time']:<10.2f}s")
        print(f"{'GLOOME':<20} {gloome_results['gain']:<10.4f} {gloome_results['loss']:<10.4f} {gloome_results['pi1']:<10.4f} {gloome_results['time']:<10.2f}s")
        
        pi1_diff = abs(gc_results['pi1'] - gloome_results['pi1'])
        speedup = gloome_results['time'] / gc_results['time']
        print(f"\n  π₁ agreement: {pi1_diff:.4f} ({pi1_diff/gc_results['pi1']*100:.1f}%)")
        print(f"  GeneContent speedup: {speedup:.1f}x")
        
        return True  # Success
    else:
        print("\n⚠ GLOOME failed at this size")
        return False  # Failure


def main():
    print("=" * 70)
    print("GLOOME STRAIN SCALING TEST")
    print("=" * 70)
    print("\nTest: Increasing strains with ALL 25,420 genes")
    
    # Load data
    data_dir = Path("data/ecoli_real")
    pam_file = data_dir / "Supplementary File 2A.txt"
    
    print("\nLoading E. coli dataset...")
    df = pd.read_csv(pam_file, sep='\t', index_col=0)
    pam = df.values.T.astype(int)
    strain_names = df.columns.tolist()
    gene_names = df.index.tolist()
    
    print(f"  Full: {len(strain_names):,} strains × {len(gene_names):,} genes")
    
    output_base = Path("results/gloome_strain_scaling")
    
    # Test 50 strains
    success_50 = test_size(pam, strain_names, gene_names, 50, output_base)
    
    if success_50:
        print("\n✓ 50 strains successful, testing 100...")
        success_100 = test_size(pam, strain_names, gene_names, 100, output_base)
        
        if success_100:
            print("\n✓ 100 strains successful!")
        else:
            print("\n✗ GLOOME failed at 100 strains")
    else:
        print("\n✗ GLOOME failed at 50 strains, stopping here")
    
    print("\n" + "=" * 70)
    print("SCALING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
