# Exploratory Scripts for Strain Heterogeneity Analysis

This directory contains scripts used during the development and validation of the strain heterogeneity framework. These scripts are preserved for reference and reproducibility but are not part of the core plugin functionality.

## Contents

### Validation Scripts
- `analyze_strain_heterogeneity.py` - E. coli case study showing strain-level heterogeneity
- `test_asymmetric_splits.py` - Tests different stratification thresholds (10%, 20%, 25%, 50%)
- `validate_sampling_bias.py` - Initial validation simulation

### Demonstration Scripts
- `demonstrate_sampling_bias.py` - Real data demonstration (superseded by controlled simulation)
- `simulate_sampling_bias_phenomenon.py` - Controlled simulation with heterogeneous gene classes

### Comparison Scripts
- `compare_all_genes_subset.py` - Full vs subset comparison
- `compare_representative_subset.py` - Representative subset analysis

### Benchmark Scripts
- `benchmark_full_ecoli_rust.py` - Rust acceleration benchmark on full E. coli dataset
- `benchmark_rust_distance.py` - Rust distance matrix computation benchmark
- `benchmark_tree_inference.py` - Tree inference method benchmarks
- `profile_tree_inference.py` - Profiling for tree inference optimization

### Superseded Modules (Gene-Frequency Focused)
- `diagnostics.py` - Old gene-frequency diagnostics (superseded by `strain_diagnostics.py`)
- `recipes.py` - Old gene-frequency recipes (superseded by `strain_recipes.py`)

These modules focused on gene frequency (singletons, rare genes) rather than strain heterogeneity. The new strain-focused approach is more biologically motivated and statistically principled.

## Key Findings

### E. coli Strain Heterogeneity (from `analyze_strain_heterogeneity.py`)

**Dataset:** 1,324 strains × 25,420 genes

**Strain heterogeneity:**
- Cloud genes per strain: 29 - 1,068 (37x range!)
- Top 10% of strains contribute 17.7% of all cloud gene observations

**Parameter sensitivity:**
```
Dataset                    Strains    λ        μ        π₁      λ/μ
Full dataset               1,324      7.39     0.007    0.999   1081
Remove top 10%             1,192      0.17     0.65     0.209   0.26
Remove top 25%             1,002      0.13     0.81     0.138   0.16
```

**Conclusion:** Removing just 10% of high-cloud strains causes >4000-fold regime shift!

### Asymmetric Splits (from `test_asymmetric_splits.py`)

Tested different stratification thresholds:
- Top 10% vs Bottom 90%
- Top 20% vs Bottom 80%
- Top 25% vs Bottom 75%
- Median (50/50)

**Finding:** All splits show both groups are loss-dominated, but with different rates. The median split dilutes the outlier effect most.

## Usage

These scripts are not intended for production use. They were used to:
1. Discover the strain heterogeneity phenomenon
2. Validate the framework on real data
3. Test different analytical approaches
4. Generate figures and results for documentation

For production analysis, use the main framework:
- `strain_diagnostics.py` - Diagnostic tools
- `strain_recipes.py` - Recipe 1 (scan) and Recipe 2 (stratified modeling)

## References

See `STRAIN_HETEROGENEITY_FRAMEWORK.md` in the repository root for the complete framework documentation.
