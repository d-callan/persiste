#!/usr/bin/env python3
"""
Compare GeneContent, GLOOME, and BadiRate on E. coli subsets.
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


def create_subset(pam, strain_names, gene_names, n_strains, n_genes, seed=42):
    """Create a random subset."""
    np.random.seed(seed)
    strain_idx = np.random.choice(len(strain_names), min(n_strains, len(strain_names)), replace=False)
    gene_idx = np.random.choice(len(gene_names), min(n_genes, len(gene_names)), replace=False)
    
    pam_subset = pam[strain_idx, :][:, gene_idx]
    strain_subset = [strain_names[i] for i in strain_idx]
    gene_subset = [gene_names[i] for i in gene_idx]
    
    return pam_subset, strain_subset, gene_subset


def run_genecontent(pam, strain_names, gene_names, output_dir: Path):
    """Run GeneContent."""
    print("\n" + "=" * 70)
    print("GENECONTENT")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    start = time.time()
    tree, metadata = infer_tree_from_pam(pam, strain_names, method="jaccard_upgma")
    tree_time = time.time() - start
    
    tree_file = output_dir / "tree.nwk"
    with open(tree_file, 'w') as f:
        f.write(tree_to_newick(tree))
        f.write('\n')
    
    start = time.time()
    result = pam_interface.fit(
        pam=pam, tree=tree, taxon_names=strain_names,
        gene_names=gene_names, use_rust=True, verbose=False,
    )
    fit_time = time.time() - start
    
    print(f"  Tree: {tree_time:.2f}s, Fit: {fit_time:.2f}s, Total: {tree_time + fit_time:.2f}s")
    print(f"  λ={result.gain_rate:.4f}, μ={result.loss_rate:.4f}, π₁={result.equilibrium_frequency:.4f}")
    
    return {
        'gain': result.gain_rate,
        'loss': result.loss_rate,
        'pi1': result.equilibrium_frequency,
        'time': tree_time + fit_time,
        'tree_file': tree_file,
    }


def run_gloome(pam, strain_names, gene_names, tree_file: Path, output_dir: Path):
    """Run GLOOME."""
    print("\n" + "=" * 70)
    print("GLOOME")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not shutil.which('gainLoss'):
        print("  ✗ Not available")
        return None
    
    seq_file = output_dir / 'sequences.fa'
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
    
    start = time.time()
    try:
        result = subprocess.run(
            ['gainLoss', str(param_file.absolute())],
            capture_output=True, text=True, timeout=600, cwd=str(output_dir)
        )
        runtime = time.time() - start
        
        if result.returncode != 0:
            print(f"  ✗ Failed (rc={result.returncode})")
            return None
        
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
        
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  λ={gain:.4f}, μ={loss:.4f}, π₁={gain/(gain+loss):.4f}")
        
        return {'gain': gain, 'loss': loss, 'pi1': gain/(gain+loss), 'time': runtime}
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def run_badirate(pam, strain_names, gene_names, tree_file: Path, output_dir: Path):
    """Run BadiRate."""
    print("\n" + "=" * 70)
    print("BADIRATE")
    print("=" * 70)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    badirate_script = Path("~/Documents/badirate/BadiRate.pl").expanduser()
    if not badirate_script.exists():
        print("  ✗ Not available")
        return None
    
    # Copy tree
    br_tree = output_dir / 'tree.nwk'
    shutil.copy(tree_file, br_tree)
    
    # Create family size file (TSV format)
    fam_file = output_dir / 'families.tsv'
    with open(fam_file, 'w') as f:
        f.write('FAM_ID\t' + '\t'.join(strain_names) + '\n')
        for j in range(len(gene_names)):
            row = [gene_names[j]] + [str(int(pam[i, j])) for i in range(len(strain_names))]
            f.write('\t'.join(row) + '\n')
    
    start = time.time()
    try:
        result = subprocess.run(
            ['perl', str(badirate_script), '-treefile', str(br_tree),
             '-sizefile', str(fam_file), '-rmodel', 'GD'],
            capture_output=True, text=True, timeout=600, cwd=str(output_dir)
        )
        runtime = time.time() - start
        
        if 'Can\'t locate' in result.stderr or 'BEGIN failed' in result.stderr:
            print("  ✗ Perl dependencies missing")
            return None
        
        if result.returncode != 0:
            print(f"  ✗ Failed (rc={result.returncode})")
            return None
        
        # Parse output
        gain, loss = None, None
        for line in result.stdout.split('\n'):
            if 'gain' in line.lower() and 'rate' in line.lower():
                parts = line.split()
                for part in parts:
                    try:
                        val = float(part)
                        if 0 < val < 100:
                            gain = val
                            break
                    except:
                        continue
            elif ('death' in line.lower() or 'loss' in line.lower()) and 'rate' in line.lower():
                parts = line.split()
                for part in parts:
                    try:
                        val = float(part)
                        if 0 < val < 100:
                            loss = val
                            break
                    except:
                        continue
        
        if gain is None or loss is None:
            print("  ✗ Could not parse output")
            print(f"  Output preview: {result.stdout[:500]}")
            return None
        
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  λ={gain:.4f}, μ={loss:.4f}, π₁={gain/(gain+loss):.4f}")
        
        return {'gain': gain, 'loss': loss, 'pi1': gain/(gain+loss), 'time': runtime}
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def compare_results(gc, gloome, badirate, size_str):
    """Compare results."""
    print("\n" + "=" * 70)
    print(f"COMPARISON: {size_str}")
    print("=" * 70)
    
    tools = []
    if gc:
        tools.append(('GeneContent', gc))
    if gloome:
        tools.append(('GLOOME', gloome))
    if badirate:
        tools.append(('BadiRate', badirate))
    
    if len(tools) < 2:
        print("\n⚠ Need at least 2 tools to compare")
        return
    
    print(f"\n{'Tool':<15} {'Gain (λ)':<12} {'Loss (μ)':<12} {'π₁':<12} {'Time':<10}")
    print("-" * 70)
    for name, res in tools:
        print(f"{name:<15} {res['gain']:<12.4f} {res['loss']:<12.4f} {res['pi1']:<12.4f} {res['time']:<10.2f}s")
    
    # Agreement analysis
    if len(tools) >= 2:
        print("\nAgreement:")
        pi1_vals = [res['pi1'] for _, res in tools]
        pi1_range = max(pi1_vals) - min(pi1_vals)
        print(f"  π₁ range: {pi1_range:.4f} ({pi1_range/np.mean(pi1_vals)*100:.1f}% of mean)")
        
        if gc and gloome:
            print(f"  GeneContent vs GLOOME: {gc['time']/gloome['time']:.1f}x faster")
        if gc and badirate:
            print(f"  GeneContent vs BadiRate: {gc['time']/badirate['time']:.1f}x faster")


def main():
    print("=" * 70)
    print("THREE-TOOL COMPARISON: GeneContent vs GLOOME vs BadiRate")
    print("=" * 70)
    
    # Load data
    data_dir = Path("data/ecoli_real")
    pam_file = data_dir / "Supplementary File 2A.txt"
    
    print("\nLoading E. coli dataset...")
    df = pd.read_csv(pam_file, sep='\t', index_col=0)
    pam = df.values.T.astype(int)
    strain_names = df.columns.tolist()
    gene_names = df.index.tolist()
    
    print(f"  Full: {len(strain_names):,} × {len(gene_names):,}")
    
    # Test on small subset
    test_sizes = [
        (50, 500, "50 strains × 500 genes"),
        (100, 1000, "100 strains × 1,000 genes"),
    ]
    
    for n_strains, n_genes, size_str in test_sizes:
        print("\n" + "=" * 70)
        print(f"TESTING: {size_str}")
        print("=" * 70)
        
        pam_sub, strain_sub, gene_sub = create_subset(
            pam, strain_names, gene_names, n_strains, n_genes
        )
        
        output_base = Path(f"results/three_tool_comparison/{n_strains}x{n_genes}")
        
        # Run all tools
        gc_res = run_genecontent(pam_sub, strain_sub, gene_sub, output_base / "genecontent")
        gloome_res = run_gloome(pam_sub, strain_sub, gene_sub, gc_res['tree_file'], output_base / "gloome")
        badirate_res = run_badirate(pam_sub, strain_sub, gene_sub, gc_res['tree_file'], output_base / "badirate")
        
        compare_results(gc_res, gloome_res, badirate_res, size_str)
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
