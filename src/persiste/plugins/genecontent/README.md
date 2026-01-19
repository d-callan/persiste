# GeneContent Plugin

Phylogenetic analysis of gene content evolution in pangenomes using binary trait models.

## Core Functionality

### Main Modules

- **`pam_interface.py`** - Primary interface for PAM (Presence/Absence Matrix) analysis
- **`strain_diagnostics.py`** - Core utilities for detecting strain heterogeneity
- **`recipes/`** - User-facing recipe package
  - `heterogeneity.py`: Strain-focused diagnostics & stratified modeling
  - `selective.py`: Constraint-driven hypothesis tests & baseline checks

### Supporting Modules

- **`constraint.py`** - Parameter constraints for model fitting
- **`inference.py`** - Core inference engine
- **`data.py`** - Data structures for gene content analysis

## Strain Heterogeneity Framework

The key insight: **Parameter estimates in pangenome analysis are highly sensitive to strain-level heterogeneity in accessory gene content.**

### Quick Start

```python
from persiste.plugins.genecontent.pam_interface import fit
from persiste.plugins.genecontent.strain_diagnostics import diagnose_strain_heterogeneity
from persiste.plugins.genecontent.recipes import (
    run_heterogeneity_diagnostic,
    test_pathway_retention,
)

# Load your PAM data
pam, taxon_names, gene_names = load_your_data()

from persiste.plugins.genecontent.recipes import run_heterogeneity_diagnostic

# Step 0: Optional summary stats
diag = diagnose_strain_heterogeneity(pam)
diag.print_report()

# Step 1: Run heterogeneity diagnostic (checks for outlier-driven regimes)
scan = run_heterogeneity_diagnostic(pam, taxon_names, gene_names)
scan.print_summary()

# Step 2: Test a selective hypothesis (e.g., pathway retention)
pathway_map = {"glycolysis": {"geneA", "geneB"}}
test_res = test_pathway_retention(pam, taxon_names, gene_names, pathway_map)
test_res.print_summary()
```

### Analysis Recipes

Recipes live under `persiste.plugins.genecontent.recipes` and are grouped by focus area:

1. **Heterogeneity diagnostics (`recipes.heterogeneity`)**
   - `run_heterogeneity_diagnostic` – user entry point for `strain_heterogeneity_scan`
   - `strain_heterogeneity_scan` – detailed diagnostic comparing strain subsets
   - `stratified_regime_modeling` – descriptive comparison of high/low-accessory regimes

2. **Selective tests (`recipes.selective`)**
   - `run_baseline_diagnostics` – sanity-check null model goodness
   - `test_selective_hypothesis` – generic LRT wrapper for any `GeneContentConstraint`
   - `test_pathway_retention` – convenience wrapper for pathway coherence analyses
   - `test_environmental_gradient` – tests for rate shifts along continuous metadata

### Constraint types (v1):
1. Global retention bias: Some genes are selectively retained
2. Host/environment association: Retained only in specific hosts
3. Functional group coherence: Pathway-level retention
4. Genome reduction bias: Lineage-specific loss acceleration
5. Environmental gradient: Rates scale with continuous metadata
6. Genomic cluster linkage: Coordinated loss/gain for adjacent genes

## Usage Examples

### Pathway Coherence
```python
from persiste.plugins.genecontent.constraints.gene_constraint import PathwayCoherenceConstraint

constraint = PathwayCoherenceConstraint(
    pathway_map={"glycolysis": {"geneA", "geneB"}},
    pathway_effects={"glycolysis": (0.5, -1.0)}  # Boost gain, reduce loss
)
```

### Environmental Gradient
```python
from persiste.plugins.genecontent.constraints.gene_constraint import EnvironmentalGradientConstraint

constraint = EnvironmentalGradientConstraint(
    metadata_key="temperature",
    gain_slope=0.1,
    loss_slope=-0.05,
    reference_value=25.0
)
```

### Genomic Cluster
```python
from persiste.plugins.genecontent.constraints.gene_constraint import GenomicClusterConstraint

constraint = GenomicClusterConstraint(
    clusters={"operon1": {"geneH", "geneI"}},
    coordinated_loss_bias=-0.8
)
```

### Why This Matters

**E. coli case study (1,324 strains):**
- Cloud genes per strain: 29 - 1,068 (37x range!)
- Top 10% of strains contribute 17.7% of cloud gene observations
- Removing top 10% causes **>4000-fold regime shift** (λ/μ from 1081 → 0.26)

**Conclusion:** Different strains can have fundamentally different gene dynamics. Always check for heterogeneity!

## Standard Analysis Workflow

```python
# 1. Load data
from persiste.plugins.genecontent.pam_interface import fit

result = fit(
    pam="path/to/pangenome.csv",
    tree=None,  # Will infer tree
    taxon_names=taxa,
    gene_names=genes,
    tree_method='jaccard_upgma',
    use_rust=True  # Rust acceleration enabled by default
)

# 2. View results
result.print_summary()

# 3. Access parameters
print(f"Gain rate (λ): {result.gain_rate}")
print(f"Loss rate (μ): {result.loss_rate}")
print(f"Stationary frequency (π₁): {result.equilibrium_frequency}")
```

## Key Parameters

- **`gain_rate` (λ)** - Rate of gene family gain
- **`loss_rate` (μ)** - Rate of gene family loss
- **`equilibrium_frequency` (π₁)** - Stationary frequency = λ/(λ+μ)
  - More interpretable than λ/μ ratio
  - More stable across sampling strategies
  - **Always report π₁ prominently!**

## Tree Inference

Three methods available:
- `jaccard_upgma` - UPGMA with Jaccard distance (default)
- `hamming_upgma` - UPGMA with Hamming distance
- `jaccard_nj` - Neighbor-joining with Jaccard distance

Rust acceleration enabled by default for distance matrix computation (5-6x speedup).

## Exploratory Scripts

See `exploratory/` subdirectory for scripts used during framework development:
- E. coli case studies
- Validation simulations
- Asymmetric split tests
- Old gene-frequency-focused approaches (superseded)

These are preserved for reference but not part of core functionality.

## Documentation

- **`STRAIN_HETEROGENEITY_FRAMEWORK.md`** (repository root) - Complete framework guide
- **`exploratory/README.md`** - Exploratory script documentation

## References

- Tettelin et al. (2005) - Open pangenome concept
- Rouli et al. (2015) - Core/shell/cloud classification
