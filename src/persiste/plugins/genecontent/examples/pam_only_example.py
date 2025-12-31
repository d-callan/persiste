#!/usr/bin/env python3
"""
Example: PAM-only GeneContent analysis (no tree required).

This demonstrates the low-barrier-to-entry workflow where users only
need a presence/absence matrix. Tree inference is explicit and transparent.
"""

import sys
from pathlib import Path
import numpy as np

# Add persiste to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from persiste.plugins.genecontent import pam_interface


def example_1_minimal():
    """Minimal usage: PAM file only."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Minimal Usage (PAM file only)")
    print("=" * 70)
    
    # Create example PAM file
    print("\nCreating example PAM file...")
    pam_file = Path("example_pam.csv")
    
    # Simulate realistic E. coli-like data
    np.random.seed(42)
    n_strains = 20
    n_genes = 200
    
    # Core genes (40% present in all)
    n_core = int(n_genes * 0.4)
    core = np.ones((n_strains, n_core), dtype=int)
    
    # Accessory genes (60% variable)
    n_accessory = n_genes - n_core
    accessory = np.random.binomial(1, np.random.uniform(0.2, 0.8, n_accessory), 
                                   size=(n_strains, n_accessory))
    
    pam = np.hstack([core, accessory])
    
    # Save to CSV
    import pandas as pd
    strain_names = [f"strain_{i:03d}" for i in range(n_strains)]
    gene_names = ([f"core_{i:04d}" for i in range(n_core)] +
                  [f"acc_{i:04d}" for i in range(n_accessory)])
    
    df = pd.DataFrame(pam, index=strain_names, columns=gene_names)
    df.to_csv(pam_file)
    print(f"  Saved: {pam_file}")
    
    # Fit model (tree will be inferred automatically)
    print("\nFitting model...")
    result = pam_interface.fit(pam_file)
    
    print("\n✓ Analysis complete!")
    print(f"  Tree was inferred using: {result.tree_metadata.method}")
    print(f"  Gain rate: {result.gain_rate:.4f}")
    print(f"  Loss rate: {result.loss_rate:.4f}")
    
    # Clean up
    pam_file.unlink()


def example_2_explicit_control():
    """Explicit control over tree inference method."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Explicit Control (choose tree method)")
    print("=" * 70)
    
    # Create example data
    np.random.seed(123)
    pam = np.random.binomial(1, 0.5, size=(15, 150))
    taxon_names = [f"isolate_{i}" for i in range(15)]
    gene_names = [f"gene_{i}" for i in range(150)]
    
    # Compare different tree inference methods
    methods = ["jaccard_upgma", "hamming_upgma"]
    
    for method in methods:
        print(f"\n--- Using {method} ---")
        result = pam_interface.fit(
            pam=pam,
            taxon_names=taxon_names,
            gene_names=gene_names,
            tree_method=method,
            verbose=False,
        )
        print(f"  Gain rate: {result.gain_rate:.4f}")
        print(f"  Loss rate: {result.loss_rate:.4f}")
        print(f"  π₁: {result.equilibrium_frequency:.4f}")


def example_3_with_provided_tree():
    """Using provided tree instead of inference."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: With Provided Tree")
    print("=" * 70)
    
    # Create example data
    np.random.seed(456)
    n_taxa = 8
    pam = np.random.binomial(1, 0.5, size=(n_taxa, 100))
    taxon_names = [f"tip{i}" for i in range(n_taxa)]
    gene_names = [f"gene_{i}" for i in range(100)]
    
    # Create a simple tree file
    import tempfile
    newick = "(((tip0:1,tip1:1):1,(tip2:1,tip3:1):1):1,((tip4:1,tip5:1):1,(tip6:1,tip7:1):1):1):0;"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.nwk', delete=False) as f:
        f.write(newick)
        tree_file = f.name
    
    print(f"\nUsing provided tree: {tree_file}")
    
    # Fit with provided tree
    result = pam_interface.fit(
        pam=pam,
        tree=tree_file,
        taxon_names=taxon_names,
        gene_names=gene_names,
        verbose=False,
    )
    
    print(f"\n✓ Tree source: {result.tree_metadata.source}")
    print(f"  Gain rate: {result.gain_rate:.4f}")
    print(f"  Loss rate: {result.loss_rate:.4f}")
    
    # Clean up
    Path(tree_file).unlink()


def example_4_retention_test():
    """Testing retention bias on specific genes."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Retention Bias Testing")
    print("=" * 70)
    
    # Create example data with retention bias
    np.random.seed(789)
    n_strains = 25
    n_genes = 150
    
    # Most genes have normal gain/loss
    pam = np.random.binomial(1, 0.5, size=(n_strains, n_genes))
    
    # But first 20 genes are "retained" (present in most strains)
    pam[:, :20] = np.random.binomial(1, 0.9, size=(n_strains, 20))
    
    taxon_names = [f"strain_{i}" for i in range(n_strains)]
    gene_names = [f"gene_{i:04d}" for i in range(n_genes)]
    
    # Test if first 20 genes show retention
    retained_genes = gene_names[:20]
    
    print(f"\nTesting retention bias for {len(retained_genes)} genes...")
    
    comparison = pam_interface.fit_with_retention_test(
        pam=pam,
        retained_families=retained_genes,
        taxon_names=taxon_names,
        gene_names=gene_names,
        verbose=False,
    )
    
    print(f"\n✓ Test complete!")
    print(f"  Evidence strength: {comparison.evidence_strength}")
    print(f"  ΔLL: {comparison.delta_ll:.2f}")
    print(f"  p-value: {comparison.lrt_result.pvalue:.4f}")


def example_5_real_dataset():
    """Using real E. coli dataset (if available)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Real E. coli Dataset")
    print("=" * 70)
    
    # Check if real dataset exists
    data_dir = Path("data/ecoli_st131")
    pam_file = data_dir / "gene_presence_absence.csv"
    
    if not pam_file.exists():
        print(f"\n⚠ Dataset not found: {pam_file}")
        print("  Run: python scripts/download_ecoli_st131.py")
        return
    
    print(f"\nLoading real E. coli ST131 data...")
    
    # Fit model (no tree provided, will infer)
    result = pam_interface.fit(
        pam=pam_file,
        tree_method="jaccard_upgma",
        verbose=True,
    )
    
    print("\n✓ Real dataset analysis complete!")
    
    # Can do further analysis with result.inference
    print("\nBaseline diagnostics:")
    result.inference.get_baseline_diagnostics(verbose=True)


def main():
    """Run all examples."""
    print("=" * 70)
    print("PAM-ONLY GENECONTENT EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate the low-barrier-to-entry workflow")
    print("where users only need a presence/absence matrix.")
    print("\nTree inference is explicit and transparent:")
    print("  - Default: Jaccard distance + UPGMA")
    print("  - Fast: milliseconds to seconds")
    print("  - Metadata preserved in results")
    
    try:
        example_1_minimal()
        example_2_explicit_control()
        example_3_with_provided_tree()
        example_4_retention_test()
        example_5_real_dataset()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETE")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
