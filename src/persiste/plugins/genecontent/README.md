# GeneContent Plugin

Phylogenetic analysis of gene content evolution in pangenomes using binary trait models.

## Core Functionality

### Main Modules

- **`pam_interface.py`** - Primary interface for PAM (Presence/Absence Matrix) analysis
- **`strain_diagnostics.py`** - Diagnostic tools for detecting strain heterogeneity
- **`strain_recipes.py`** - Two-recipe framework for handling heterogeneous datasets

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
from persiste.plugins.genecontent.strain_recipes import strain_heterogeneity_scan

# Load your PAM data
pam, taxon_names, gene_names = load_your_data()

# Step 1: Run diagnostic
diag = diagnose_strain_heterogeneity(pam)
diag.print_report()

# Step 2: Run heterogeneity scan (ALWAYS RECOMMENDED)
scan = strain_heterogeneity_scan(pam, taxon_names, gene_names)
scan.print_summary()

# Step 3: Decide based on scan results
max_shift = max(abs(v) for v in scan.parameter_shifts.values())

if max_shift > 100:
    # EXTREME heterogeneity - use stratified modeling
    from persiste.plugins.genecontent.strain_recipes import stratified_regime_modeling
    
    stratified = stratified_regime_modeling(pam, taxon_names, gene_names)
    stratified.print_summary()
else:
    # Standard global model
    result = fit(pam, taxon_names=taxon_names, gene_names=gene_names)
    result.print_summary()
```

### Two-Recipe Framework

**Recipe 1: Strain Heterogeneity Scan (DIAGNOSTIC)**
- Tests parameter stability by removing outlier strains
- **This is the hypothesis test** - parameter shifts >100% indicate significant heterogeneity
- Should be run as standard first step for any pangenome analysis
- Returns: `HeterogeneityScanResult` with parameter shifts and interpretation

**Recipe 2: Stratified Regime Modeling (DESCRIPTIVE)**
- Models high-accessory and low-accessory strains separately
- Provides descriptive comparison of evolutionary regimes
- NOT a formal statistical test - use for biological interpretation
- Use when Recipe 1 detects significant heterogeneity
- Returns: `StratifiedRegimeResult` with separate regime estimates

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
