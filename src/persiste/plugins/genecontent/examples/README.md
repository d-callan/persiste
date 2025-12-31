# GeneContent Plugin Examples

This directory contains examples demonstrating the GeneContent plugin.

## Production Workflow Examples

### **`example_strain_heterogeneity_workflow.py`** ⭐
**THE RECOMMENDED TEMPLATE** for pangenome analysis.

Demonstrates complete workflow:
1. Loading PAM data
2. Running diagnostic scan
3. Running heterogeneity scan (ALWAYS RECOMMENDED)
4. Decision logic based on scan results
5. Using stratified modeling when needed

**Use this as your template for analysis!**

### **`analyze_ecoli_real.py`**
Analysis of the full E. coli dataset (1,324 strains × 25,420 genes).
Shows real-world application on a large dataset.

## Basic API Examples

- **`pam_only_example.py`** - Simple PAM-only analysis (no tree required)
- **`basic_example.py`** - Basic usage with tree inference
- **`inference_example.py`** - Detailed inference workflow
- **`likelihood_example.py`** - Likelihood computation examples

**Note:** Basic examples demonstrate the API but do not include the **strain heterogeneity framework**.

## For Production Analysis

**Always use the complete workflow:**
- `example_strain_heterogeneity_workflow.py` - **Start here**
- `/STRAIN_HETEROGENEITY_FRAMEWORK.md` - Complete framework documentation

## Quick Start

```python
# Basic PAM analysis
from persiste.plugins.genecontent.pam_interface import fit

result = fit(
    pam="pangenome.csv",
    taxon_names=taxa,
    gene_names=genes,
    tree_method='jaccard_upgma'
)

result.print_summary()
```

## Important Note

**Always run the heterogeneity scan** before interpreting results:

```python
from persiste.plugins.genecontent.strain_recipes import strain_heterogeneity_scan

scan = strain_heterogeneity_scan(pam, taxa, genes)
scan.print_summary()

# If parameter shifts >100%, use stratified modeling
```

See the main workflow example for complete details.
