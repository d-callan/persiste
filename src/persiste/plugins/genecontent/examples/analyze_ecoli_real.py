#!/usr/bin/env python3
"""
Analyze real E. coli PAM data using PAM-only pipeline.

Data from: BMC Genomics 2022 supplementary file 2A
- 25,421 genes
- 1,325 E. coli strains
"""

import sys
from pathlib import Path
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persiste.plugins.genecontent import pam_interface


def load_ecoli_pam(pam_file: Path):
    """Load the E. coli PAM from space-separated text file."""
    print(f"Loading PAM from: {pam_file}")
    print("  (This may take a moment for 25k genes × 1.3k strains...)")
    
    # Load with pandas to handle headers
    import pandas as pd
    df = pd.read_csv(pam_file, sep='\t', index_col=0)
    
    print(f"  Shape: {df.shape}")
    print(f"  Genes: {df.shape[0]:,}")
    print(f"  Strains: {df.shape[1]:,}")
    
    # Convert to numpy array and transpose to (strains × genes)
    pam = df.values.T.astype(int)
    
    # Get names from dataframe
    strain_names = df.columns.tolist()
    gene_names = df.index.tolist()
    
    # Print summary statistics
    print(f"\nPAM Statistics:")
    genes_per_strain = pam.sum(axis=1)
    print(f"  Genes per strain: {genes_per_strain.mean():.0f} ± {genes_per_strain.std():.0f}")
    print(f"  Range: {genes_per_strain.min()}-{genes_per_strain.max()}")
    
    gene_freq = pam.sum(axis=0) / pam.shape[0]
    print(f"\n  Gene frequency distribution:")
    print(f"    Core (100%): {(gene_freq == 1.0).sum():,}")
    print(f"    Common (50-99%): {((gene_freq >= 0.5) & (gene_freq < 1.0)).sum():,}")
    print(f"    Intermediate (20-49%): {((gene_freq >= 0.2) & (gene_freq < 0.5)).sum():,}")
    print(f"    Rare (<20%): {(gene_freq < 0.2).sum():,}")
    
    return pam, strain_names, gene_names


def main():
    print("=" * 70)
    print("REAL E. COLI PAM ANALYSIS")
    print("=" * 70)
    print("\nData source: BMC Genomics 2022 supplementary file 2A")
    print("Paper: Comparative genomics of E. coli")
    
    # Load data
    data_dir = Path("data/ecoli_real")
    pam_file = data_dir / "Supplementary File 2A.txt"
    
    if not pam_file.exists():
        print(f"\n✗ PAM file not found: {pam_file}")
        return
    
    print("\n" + "=" * 70)
    print("STEP 1: Load PAM")
    print("=" * 70 + "\n")
    
    pam, strain_names, gene_names = load_ecoli_pam(pam_file)
    
    # Subsample for faster analysis (optional)
    print("\n" + "=" * 70)
    print("STEP 2: Subsample for Analysis")
    print("=" * 70)
    
    # Use a subset for faster initial analysis
    n_strains_subset = 100
    n_genes_subset = 5000
    
    print(f"\nSubsampling for faster analysis:")
    print(f"  Strains: {n_strains_subset} (from {len(strain_names):,})")
    print(f"  Genes: {n_genes_subset} (from {len(gene_names):,})")
    
    # Random subsample
    np.random.seed(42)
    strain_idx = np.random.choice(len(strain_names), n_strains_subset, replace=False)
    gene_idx = np.random.choice(len(gene_names), n_genes_subset, replace=False)
    
    pam_subset = pam[strain_idx, :][:, gene_idx]
    strain_names_subset = [strain_names[i] for i in strain_idx]
    gene_names_subset = [gene_names[i] for i in gene_idx]
    
    print(f"  Subset shape: {pam_subset.shape}")
    
    # Run analysis
    print("\n" + "=" * 70)
    print("STEP 3: Infer Tree from PAM")
    print("=" * 70)
    print("\nUsing Jaccard distance + UPGMA...")
    print("(Tree inference should take seconds for 100 strains)")
    
    tree_start = time.time()
    
    # Just infer tree to check timing
    from persiste.plugins.genecontent.tree_inference import infer_tree_from_pam
    tree, metadata = infer_tree_from_pam(pam_subset, strain_names_subset, method="jaccard_upgma")
    
    tree_time = time.time() - tree_start
    print(f"\n✓ Tree inferred in {tree_time:.2f}s")
    print(f"  Tips: {tree.n_tips}")
    print(f"  Total nodes: {tree.n_nodes}")
    
    # Run full analysis
    print("\n" + "=" * 70)
    print("STEP 4: Fit GeneContent Model")
    print("=" * 70)
    print("\nFitting with Rust acceleration...")
    
    fit_start = time.time()
    
    result = pam_interface.fit(
        pam=pam_subset,
        taxon_names=strain_names_subset,
        gene_names=gene_names_subset,
        tree=tree,  # Use pre-computed tree
        tree_method="jaccard_upgma",
        use_rust=True,
        verbose=False,
    )
    
    fit_time = time.time() - fit_start
    
    print(f"\n✓ Model fitted in {fit_time:.2f}s")
    print(f"  Total time: {tree_time + fit_time:.2f}s")
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    result.print_summary()
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    print(f"\nGain rate: {result.gain_rate:.4f}")
    print(f"  → Genes gained per unit branch length")
    
    print(f"\nLoss rate: {result.loss_rate:.4f}")
    print(f"  → Genes lost per unit branch length")
    
    print(f"\nEquilibrium frequency: {result.equilibrium_frequency:.4f}")
    print(f"  → Expected proportion of genes present at equilibrium")
    print(f"  → π₁ = λ/(λ+μ) = {result.gain_rate:.4f}/({result.gain_rate:.4f}+{result.loss_rate:.4f})")
    
    turnover = result.gain_rate + result.loss_rate
    print(f"\nTurnover rate: {turnover:.4f}")
    print(f"  → Total rate of gene content change")
    
    if result.gain_rate > result.loss_rate:
        print(f"\n✓ Gain-dominated: genomes tend to accumulate genes")
    else:
        print(f"\n✓ Loss-dominated: genomes tend to lose genes")
    
    # Baseline diagnostics
    print("\n" + "=" * 70)
    print("BASELINE DIAGNOSTICS")
    print("=" * 70 + "\n")
    
    result.inference.get_baseline_diagnostics(verbose=True)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run on full dataset (all 1,325 strains × 25,421 genes)")
    print("  2. Test for retention bias on specific gene sets")
    print("  3. Compare with GLOOME results")
    print("  4. Analyze gene frequency classes separately")


if __name__ == "__main__":
    main()
