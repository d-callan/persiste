# Benchmark and Comparison Results

This directory contains performance benchmarks and tool comparisons for the GeneContent plugin.

## Contents

### Documentation
- **`ECOLI_FULL_ANALYSIS.md`** - Full E. coli dataset (1,324 strains) analysis results
- **`GLOOME_COMPARISON.md`** - Detailed comparison with GLOOME tool
- **`TOOL_COMPARISON_FINAL.md`** - Final comparison across multiple tools

### Scripts
- **`compare_all_tools.py`** - Compare multiple phylogenetic tools
- **`compare_full_ecoli.py`** - Full E. coli dataset comparison
- **`compare_tools_real_data.py`** - Tool comparison on real datasets
- **`test_gloome_scaling.py`** - GLOOME scaling behavior tests
- **`test_gloome_strain_scaling.py`** - GLOOME strain-level scaling tests

## Key Findings

### Performance
- **Rust acceleration:** 5-6x speedup for distance matrix computation
- **Full E. coli runtime:** ~5.6s with Rust (vs ~62s baseline)
- **Tree inference:** UPGMA methods significantly faster than NJ

### Tool Comparisons
- GeneContent provides comparable accuracy to GLOOME
- Significantly faster for large datasets
- Better handling of strain heterogeneity

### Strain Heterogeneity
- E. coli shows extreme heterogeneity (CV=0.40)
- Top 10% of strains drive >4000-fold parameter shift
- Stratified modeling reveals distinct regimes

## Usage

These scripts and documents are preserved for reproducibility and reference. For current analysis workflows, see:
- `/scripts/example_strain_heterogeneity_workflow.py`
- `/STRAIN_HETEROGENEITY_FRAMEWORK.md`
