"""
PAM-only interface for GeneContent analysis.

Provides a simple, user-friendly API for analyzing presence/absence matrices
without requiring a phylogenetic tree. Tree inference is explicit and transparent.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union
import numpy as np
import pandas as pd

from persiste.core.trees import TreeStructure
from .tree_inference import (
    infer_tree_from_pam,
    TreeInferenceMetadata,
    validate_inferred_tree,
)
from .inference.gene_inference import GeneContentData, GeneContentInference
from .constraints.gene_constraint import GeneContentConstraint, NullConstraint


@dataclass
class PAMAnalysisResult:
    """
    Result from PAM-only analysis.
    
    Attributes:
        inference: GeneContentInference object for further analysis
        data: GeneContentData with tree and PAM
        tree_metadata: Metadata about tree source (provided or inferred)
        gain_rate: Estimated gain rate
        loss_rate: Estimated loss rate
        equilibrium_frequency: π₁ = gain/(gain+loss)
        log_likelihood: Model log-likelihood
    """
    inference: GeneContentInference
    data: GeneContentData
    tree_metadata: TreeInferenceMetadata
    gain_rate: float
    loss_rate: float
    equilibrium_frequency: float
    log_likelihood: float
    
    def print_summary(self):
        """Print analysis summary."""
        print("=" * 70)
        print("GENECONTENT ANALYSIS SUMMARY")
        print("=" * 70)
        
        print(f"\nData:")
        print(f"  Taxa: {self.data.n_taxa}")
        print(f"  Gene families: {self.data.n_families}")
        
        print(f"\nTree:")
        print(f"  Source: {self.tree_metadata.source}")
        if self.tree_metadata.source == "inferred":
            print(f"  Method: {self.tree_metadata.method}")
            print(f"  Distance metric: {self.tree_metadata.distance_metric}")
            print(f"  Clustering: {self.tree_metadata.clustering_method}")
            print(f"  ⚠ Tree was inferred from PAM - results approximate phylogeny")
        
        print(f"\nGlobal rates:")
        print(f"  Gain rate (λ): {self.gain_rate:.4f}")
        print(f"  Loss rate (μ): {self.loss_rate:.4f}")
        print(f"  Equilibrium frequency (π₁): {self.equilibrium_frequency:.4f}")
        print(f"  Log-likelihood: {self.log_likelihood:.2f}")
        
        print("\n" + "=" * 70)


def load_pam(
    pam_file: Union[str, Path],
    taxon_col: Optional[str] = None,
    gene_col: Optional[str] = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load presence/absence matrix from file.
    
    Supports:
    - CSV/TSV with taxa as rows or columns
    - Roary/Panaroo gene_presence_absence.csv format
    
    Args:
        pam_file: Path to PAM file
        taxon_col: Column name for taxon IDs (if taxa are rows)
        gene_col: Column name for gene IDs (if genes are rows)
        
    Returns:
        pam: Binary matrix (n_taxa × n_genes)
        taxon_names: List of taxon names
        gene_names: List of gene names
    """
    pam_file = Path(pam_file)
    
    # Read file
    if pam_file.suffix == '.csv':
        df = pd.read_csv(pam_file, index_col=0)
    elif pam_file.suffix in ['.tsv', '.txt']:
        df = pd.read_csv(pam_file, sep='\t', index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {pam_file.suffix}")
    
    # Determine orientation (taxa as rows or columns)
    # Heuristic: if first column name looks like a taxon, taxa are columns
    first_col = df.columns[0]
    if any(x in str(first_col).lower() for x in ['strain', 'sample', 'isolate', 'genome']):
        # Taxa are columns, genes are rows
        pam = df.values.T.astype(int)
        taxon_names = df.columns.tolist()
        gene_names = df.index.tolist()
    else:
        # Taxa are rows, genes are columns
        pam = df.values.astype(int)
        taxon_names = df.index.tolist()
        gene_names = df.columns.tolist()
    
    return pam, taxon_names, gene_names


def fit(
    pam: Union[str, Path, np.ndarray],
    tree: Optional[Union[str, Path, TreeStructure]] = None,
    taxon_names: Optional[list[str]] = None,
    gene_names: Optional[list[str]] = None,
    tree_method: Literal["jaccard_upgma", "hamming_upgma", "jaccard_nj"] = "jaccard_upgma",
    constraint: Optional[GeneContentConstraint] = None,
    use_rust: bool = True,
    verbose: bool = True,
) -> PAMAnalysisResult:
    """
    Fit GeneContent model to presence/absence matrix.
    
    This is the main entry point for PAM-only analysis. It handles:
    1. Loading PAM from file or array
    2. Tree inference (if tree not provided)
    3. Model fitting with Rust acceleration
    4. Result packaging with metadata
    
    Args:
        pam: Presence/absence matrix as:
            - File path (CSV/TSV)
            - NumPy array (n_taxa × n_genes)
        tree: Optional phylogenetic tree as:
            - File path (Newick format)
            - TreeStructure object
            - None (will infer from PAM)
        taxon_names: Taxon names (required if pam is array)
        gene_names: Gene names (required if pam is array)
        tree_method: Tree inference method if tree=None:
            - "jaccard_upgma": Jaccard distance + UPGMA (default, recommended)
            - "hamming_upgma": Hamming distance + UPGMA
            - "jaccard_nj": Jaccard distance + Neighbor-Joining
        constraint: Optional constraint model (default: global rates)
        use_rust: Use Rust acceleration (default: True)
        verbose: Print progress and results
        
    Returns:
        PAMAnalysisResult with fitted model and metadata
        
    Examples:
        Minimal usage (PAM file only):
        >>> result = fit("gene_presence.csv")
        
        With explicit tree:
        >>> result = fit("gene_presence.csv", tree="phylogeny.nwk")
        
        With constraint:
        >>> from persiste.plugins.genecontent.constraints import RetentionBiasConstraint
        >>> constraint = RetentionBiasConstraint(retained_families=["gene1", "gene2"])
        >>> result = fit("gene_presence.csv", constraint=constraint)
        
        From array:
        >>> pam = np.random.binomial(1, 0.5, size=(10, 100))
        >>> taxa = [f"strain_{i}" for i in range(10)]
        >>> genes = [f"gene_{i}" for i in range(100)]
        >>> result = fit(pam, taxon_names=taxa, gene_names=genes)
    """
    # Load PAM
    if isinstance(pam, (str, Path)):
        if verbose:
            print(f"Loading PAM from: {pam}")
        pam_array, taxon_names, gene_names = load_pam(pam)
    elif isinstance(pam, np.ndarray):
        if taxon_names is None or gene_names is None:
            raise ValueError("taxon_names and gene_names required when pam is array")
        pam_array = pam.astype(int)
    else:
        raise TypeError(f"pam must be file path or numpy array, got {type(pam)}")
    
    if verbose:
        print(f"  Taxa: {len(taxon_names)}")
        print(f"  Genes: {len(gene_names)}")
    
    # Load or infer tree
    if tree is None:
        if verbose:
            print(f"\nInferring tree from PAM using {tree_method}...")
        tree_obj, tree_metadata = infer_tree_from_pam(
            pam_array, taxon_names, method=tree_method
        )
        
        # Validate inferred tree
        validation = validate_inferred_tree(tree_obj, pam_array)
        if not validation["valid"]:
            raise ValueError(f"Inferred tree validation failed: {validation['errors']}")
        
        if verbose and validation["warnings"]:
            print("  Warnings:")
            for warning in validation["warnings"]:
                print(f"    ⚠ {warning}")
        
    elif isinstance(tree, (str, Path)):
        if verbose:
            print(f"\nLoading tree from: {tree}")
        from persiste.core.trees import load_tree
        tree_obj = load_tree(str(tree))
        tree_metadata = TreeInferenceMetadata(source="provided")
        
    elif isinstance(tree, TreeStructure):
        tree_obj = tree
        tree_metadata = TreeInferenceMetadata(source="provided")
    else:
        raise TypeError(f"tree must be file path, TreeStructure, or None, got {type(tree)}")
    
    if verbose:
        print(f"  Tree: {tree_obj.n_tips} tips, {tree_obj.n_nodes} total nodes")
    
    # Create GeneContentData
    data = GeneContentData(
        tree=tree_obj,
        presence_matrix=pam_array,
        taxon_names=taxon_names,
        family_names=gene_names,
    )
    
    # Create inference object
    inference = GeneContentInference(data, use_rust=use_rust)
    
    # Fit model
    if verbose:
        print(f"\nFitting model...")
        if use_rust:
            print("  Using Rust acceleration")
    
    if constraint is None:
        # Fit null model (global rates)
        result = inference.fit_null()
    else:
        # Fit with constraint
        result = inference.fit_with_constraint(constraint)
    
    # Extract results
    gain_rate = np.exp(result.parameters['log_gain'])
    loss_rate = np.exp(result.parameters['log_loss'])
    equilibrium_freq = gain_rate / (gain_rate + loss_rate)
    
    # Package results
    analysis_result = PAMAnalysisResult(
        inference=inference,
        data=data,
        tree_metadata=tree_metadata,
        gain_rate=gain_rate,
        loss_rate=loss_rate,
        equilibrium_frequency=equilibrium_freq,
        log_likelihood=result.log_likelihood,
    )
    
    if verbose:
        print("\n" + "=" * 70)
        analysis_result.print_summary()
    
    return analysis_result


def fit_with_retention_test(
    pam: Union[str, Path, np.ndarray],
    retained_families: list[str],
    tree: Optional[Union[str, Path, TreeStructure]] = None,
    taxon_names: Optional[list[str]] = None,
    gene_names: Optional[list[str]] = None,
    tree_method: Literal["jaccard_upgma", "hamming_upgma", "jaccard_nj"] = "jaccard_upgma",
    use_rust: bool = True,
    verbose: bool = True,
):
    """
    Convenience function for retention bias testing.
    
    Args:
        pam: Presence/absence matrix
        retained_families: List of gene families to test for retention
        tree: Optional phylogenetic tree
        taxon_names: Taxon names (if pam is array)
        gene_names: Gene names (if pam is array)
        tree_method: Tree inference method
        use_rust: Use Rust acceleration
        verbose: Print results
        
    Returns:
        Tuple of (null_result, alt_result, comparison_result)
    """
    from .constraints.gene_constraint import RetentionBiasConstraint
    
    # Fit null model
    null_result = fit(
        pam=pam,
        tree=tree,
        taxon_names=taxon_names,
        gene_names=gene_names,
        tree_method=tree_method,
        constraint=None,
        use_rust=use_rust,
        verbose=verbose,
    )
    
    # Fit with retention constraint
    if verbose:
        print(f"\nTesting retention bias for {len(retained_families)} families...")
    
    constraint = RetentionBiasConstraint(retained_families=set(retained_families))
    comparison = null_result.inference.compare_to_null(
        constraint=constraint,
        verbose=verbose,
    )
    
    return comparison
