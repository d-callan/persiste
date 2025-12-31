#!/usr/bin/env python3
"""
Compare GeneContent vs GLOOME on a REPRESENTATIVE subset of E. coli data.

Uses stratified sampling to preserve gene frequency distribution.
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


def create_representative_subset(pam, strain_names, gene_names, 
                                 n_strains=200, n_genes=5000, seed=42):
    """
    Create a representative subset using stratified sampling.
    
    Preserves gene frequency distribution by sampling proportionally
    from each frequency class.
    """
    print("\n" + "=" * 70)
    print("CREATING REPRESENTATIVE SUBSET")
    print("=" * 70)
    
    np.random.seed(seed)
    
    # Calculate gene frequencies
    gene_freq = pam.sum(axis=0) / pam.shape[0]
    
    # Define frequency classes
    core = gene_freq == 1.0
    common = (gene_freq >= 0.5) & (gene_freq < 1.0)
    intermediate = (gene_freq >= 0.2) & (gene_freq < 0.5)
    rare = gene_freq < 0.2
    
    print(f"\nFull dataset gene distribution:")
    print(f"  Core (100%):        {core.sum():6,} ({core.sum()/len(gene_freq)*100:5.1f}%)")
    print(f"  Common (50-99%):    {common.sum():6,} ({common.sum()/len(gene_freq)*100:5.1f}%)")
    print(f"  Intermediate (20-49%): {intermediate.sum():6,} ({intermediate.sum()/len(gene_freq)*100:5.1f}%)")
    print(f"  Rare (<20%):        {rare.sum():6,} ({rare.sum()/len(gene_freq)*100:5.1f}%)")
    
    # Calculate proportional sample sizes
    n_core = int(n_genes * core.sum() / len(gene_freq))
    n_common = int(n_genes * common.sum() / len(gene_freq))
    n_intermediate = int(n_genes * intermediate.sum() / len(gene_freq))
    n_rare = n_genes - n_core - n_common - n_intermediate  # Remainder
    
    print(f"\nTarget subset distribution ({n_genes} genes):")
    print(f"  Core:        {n_core:6,} ({n_core/n_genes*100:5.1f}%)")
    print(f"  Common:      {n_common:6,} ({n_common/n_genes*100:5.1f}%)")
    print(f"  Intermediate: {n_intermediate:6,} ({n_intermediate/n_genes*100:5.1f}%)")
    print(f"  Rare:        {n_rare:6,} ({n_rare/n_genes*100:5.1f}%)")
    
    # Sample from each class
    core_idx = np.where(core)[0]
    common_idx = np.where(common)[0]
    intermediate_idx = np.where(intermediate)[0]
    rare_idx = np.where(rare)[0]
    
    selected_core = np.random.choice(core_idx, min(n_core, len(core_idx)), replace=False)
    selected_common = np.random.choice(common_idx, min(n_common, len(common_idx)), replace=False)
    selected_intermediate = np.random.choice(intermediate_idx, min(n_intermediate, len(intermediate_idx)), replace=False)
    selected_rare = np.random.choice(rare_idx, min(n_rare, len(rare_idx)), replace=False)
    
    gene_idx = np.concatenate([selected_core, selected_common, selected_intermediate, selected_rare])
    
    # Random sample of strains
    strain_idx = np.random.choice(len(strain_names), n_strains, replace=False)
    
    # Create subset
    pam_subset = pam[strain_idx, :][:, gene_idx]
    strain_names_subset = [strain_names[i] for i in strain_idx]
    gene_names_subset = [gene_names[i] for i in gene_idx]
    
    # Verify distribution
    subset_freq = pam_subset.sum(axis=0) / pam_subset.shape[0]
    subset_core = (subset_freq == 1.0).sum()
    subset_common = ((subset_freq >= 0.5) & (subset_freq < 1.0)).sum()
    subset_intermediate = ((subset_freq >= 0.2) & (subset_freq < 0.5)).sum()
    subset_rare = (subset_freq < 0.2).sum()
    
    print(f"\nActual subset distribution:")
    print(f"  Core:        {subset_core:6,} ({subset_core/len(gene_idx)*100:5.1f}%)")
    print(f"  Common:      {subset_common:6,} ({subset_common/len(gene_idx)*100:5.1f}%)")
    print(f"  Intermediate: {subset_intermediate:6,} ({subset_intermediate/len(gene_idx)*100:5.1f}%)")
    print(f"  Rare:        {subset_rare:6,} ({subset_rare/len(gene_idx)*100:5.1f}%)")
    
    print(f"\n✓ Representative subset created:")
    print(f"  Strains: {n_strains}")
    print(f"  Genes: {len(gene_idx)}")
    print(f"  Preserves frequency distribution")
    
    return pam_subset, strain_names_subset, gene_names_subset


def run_genecontent(pam, strain_names, gene_names, output_dir: Path):
    """Run GeneContent analysis."""
    print("\n" + "=" * 70)
    print("GENECONTENT ANALYSIS")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Infer tree
    print("\nInferring tree...")
    tree_start = time.time()
    tree, metadata = infer_tree_from_pam(pam, strain_names, method="jaccard_upgma")
    tree_time = time.time() - tree_start
    print(f"  ✓ Tree inferred in {tree_time:.2f}s")
    
    # Save tree
    tree_file = output_dir / "tree.nwk"
    with open(tree_file, 'w') as f:
        f.write(tree_to_newick(tree))
        f.write('\n')
    
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
        f.write(f"Strains: {len(strain_names)}\n")
        f.write(f"Genes: {len(gene_names)}\n")
        f.write(f"Gain rate: {result.gain_rate:.6f}\n")
        f.write(f"Loss rate: {result.loss_rate:.6f}\n")
        f.write(f"π₁: {result.equilibrium_frequency:.6f}\n")
        f.write(f"Log-likelihood: {result.log_likelihood:.2f}\n")
        f.write(f"Tree time: {tree_time:.2f}s\n")
        f.write(f"Fit time: {fit_time:.2f}s\n")
        f.write(f"Total time: {total_time:.2f}s\n")
    
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


def run_gloome(pam, strain_names, gene_names, tree_file: Path, output_dir: Path):
    """Run GLOOME analysis."""
    print("\n" + "=" * 70)
    print("GLOOME ANALYSIS")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    import shutil
    if not shutil.which('gainLoss'):
        print("\n⚠ GLOOME not available")
        return None
    
    # Create input files
    print("\nPreparing input files...")
    seq_file = output_dir / 'sequences.fa'
    with open(seq_file, 'w') as f:
        for i, taxon in enumerate(strain_names):
            f.write(f'>{taxon}\n')
            sequence = ''.join([str(int(pam[i, j])) for j in range(len(gene_names))])
            f.write(sequence + '\n')
    
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
    
    print("Running GLOOME...")
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
            return None
        
        print(f"  ✓ GLOOME completed in {runtime:.2f}s")
        
        # Parse results
        params_file = output_dir / 'EstimatedParameters.txt'
        if not params_file.exists():
            print("\n✗ Output file not found")
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
            print("\n✗ Could not parse results")
            return None
        
        print(f"\nResults:")
        print(f"  Gain rate: {gain_rate:.6f}")
        print(f"  Loss rate: {loss_rate:.6f}")
        print(f"  π₁: {gain_rate/(gain_rate + loss_rate):.6f}")
        if log_likelihood:
            print(f"  Log-likelihood: {log_likelihood:.2f}")
        print(f"  Runtime: {runtime:.2f}s")
        
        return {
            'gain_rate': gain_rate,
            'loss_rate': loss_rate,
            'pi1': gain_rate / (gain_rate + loss_rate),
            'log_likelihood': log_likelihood,
            'runtime': runtime,
        }
        
    except Exception as e:
        print(f"\n✗ GLOOME error: {e}")
        return None


def compare_results(gc_results, gloome_results, full_results):
    """Compare results."""
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("Subset vs Full Dataset")
    print("-" * 70)
    
    print(f"\n{'Metric':<20} {'Subset':<15} {'Full':<15} {'Ratio':<10}")
    print("-" * 70)
    print(f"{'Gain rate':<20} {gc_results['gain_rate']:<15.4f} {full_results['gain']:<15.4f} {gc_results['gain_rate']/full_results['gain']:<10.2f}x")
    print(f"{'Loss rate':<20} {gc_results['loss_rate']:<15.4f} {full_results['loss']:<15.4f} {gc_results['loss_rate']/full_results['loss']:<10.2f}x")
    print(f"{'π₁':<20} {gc_results['pi1']:<15.4f} {full_results['pi1']:<15.4f} {abs(gc_results['pi1']-full_results['pi1']):<10.4f} diff")
    
    if gloome_results:
        print("\n" + "-" * 70)
        print("GeneContent vs GLOOME (Subset)")
        print("-" * 70)
        
        print(f"\n{'Metric':<20} {'GeneContent':<15} {'GLOOME':<15} {'Ratio':<10}")
        print("-" * 70)
        print(f"{'Gain rate':<20} {gc_results['gain_rate']:<15.4f} {gloome_results['gain_rate']:<15.4f} {gc_results['gain_rate']/gloome_results['gain_rate']:<10.2f}x")
        print(f"{'Loss rate':<20} {gc_results['loss_rate']:<15.4f} {gloome_results['loss_rate']:<15.4f} {gc_results['loss_rate']/gloome_results['loss_rate']:<10.2f}x")
        print(f"{'π₁':<20} {gc_results['pi1']:<15.4f} {gloome_results['pi1']:<15.4f} {abs(gc_results['pi1']-gloome_results['pi1']):<10.4f} diff")
        print(f"{'Runtime':<20} {gc_results['total_time']:<15.2f}s {gloome_results['runtime']:<15.2f}s {gloome_results['runtime']/gc_results['total_time']:<10.2f}x")


def main():
    print("=" * 70)
    print("REPRESENTATIVE SUBSET COMPARISON")
    print("=" * 70)
    
    # Load full data
    data_dir = Path("data/ecoli_real")
    pam_file = data_dir / "Supplementary File 2A.txt"
    
    print("\nLoading full dataset...")
    df = pd.read_csv(pam_file, sep='\t', index_col=0)
    pam = df.values.T.astype(int)
    strain_names = df.columns.tolist()
    gene_names = df.index.tolist()
    
    print(f"  Full dataset: {len(strain_names):,} strains × {len(gene_names):,} genes")
    
    # Create representative subset
    pam_subset, strain_subset, gene_subset = create_representative_subset(
        pam, strain_names, gene_names,
        n_strains=200, n_genes=5000
    )
    
    # Run analyses
    output_dir = Path("results/representative_subset")
    
    gc_output = output_dir / "genecontent"
    gc_results = run_genecontent(pam_subset, strain_subset, gene_subset, gc_output)
    
    gloome_output = output_dir / "gloome"
    gloome_results = run_gloome(
        pam_subset, strain_subset, gene_subset,
        gc_results['tree_file'], gloome_output
    )
    
    # Full dataset results for comparison
    full_results = {
        'gain': 7.389056,
        'loss': 0.006835,
        'pi1': 0.999076,
    }
    
    compare_results(gc_results, gloome_results, full_results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
