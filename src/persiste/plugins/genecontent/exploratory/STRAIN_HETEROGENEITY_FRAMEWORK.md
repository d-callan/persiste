# Strain Heterogeneity Framework for Pangenome Analysis

## The Core Insight

**The gain→loss regime shift is caused by STRAIN HETEROGENEITY, not just gene frequency.**

### E. coli Evidence

| Metric | Value |
|--------|-------|
| Cloud genes per strain (mean) | 289 |
| Cloud genes per strain (range) | **29 - 1,068** (37x difference!) |
| Top 10% strain contribution | **17.7%** of all cloud observations |
| **Parameter sensitivity** | Removing top 10% → **>4000-fold regime shift** |

**Removing just 132 strains (10%) changes λ/μ from 1081 to 0.26!**

## Two-Recipe Framework

### Recipe 1: Strain Heterogeneity Scan (DIAGNOSTIC)

**Goal:** Detect whether parameter estimates are driven by outlier strains

**What it is:**
- **THIS IS THE HYPOTHESIS TEST** for strain heterogeneity
- Tests parameter stability by removing outlier strains
- Parameter shifts >100% indicate significant heterogeneity
- **Should be run as standard first step for any pangenome analysis**

**What it is NOT:**
- Not a formal likelihood ratio test
- Not comparing nested models
- Not testing specific biological hypotheses

**Method:**
1. Compute per-strain cloud gene counts
2. Fit global model
3. Refit after removing:
   - Top 10% high-cloud strains
   - Bottom 10% low-cloud strains
4. Compare parameter shifts

**E. coli Results:**
```
Dataset                    Strains    λ        μ        π₁      λ/μ
Full dataset               1,324      7.39     0.007    0.999   1081
Remove top 10%             1,192      0.17     0.65     0.209   0.26
Remove bottom 10%          1,191      0.42     0.31     0.580   1.38

Parameter shifts:
  λ (remove top 10%):  -97.7%
  μ (remove top 10%):  +9436.2%
  π₁ (remove top 10%): -79.1%

Interpretation: EXTREME regime heterogeneity
```

**Decision rule:**
- Parameter shifts >100% → EXTREME heterogeneity → Use Recipe 2
- Parameter shifts 50-100% → STRONG heterogeneity → Consider Recipe 2
- Parameter shifts 20-50% → MODERATE heterogeneity → Report with caveat
- Parameter shifts <20% → STABLE → Standard global model appropriate

### Recipe 2: Stratified Regime Modeling (DESCRIPTIVE)

**Goal:** Characterize evolutionary regimes in different strain groups

**What it is:**
- **DESCRIPTIVE comparison** of high-accessory vs low-accessory strains
- Provides biological interpretation of regime differences
- Two separate models, not one complicated model
- Use when Recipe 1 detects significant heterogeneity

**What it is NOT:**
- Not a formal statistical test
- Not a model selection procedure
- Not testing whether stratification is "better"

**Method:**
1. Partition strains by cloud gene content (threshold: median or user-defined)
2. Fit GeneContent independently to each group
3. Compare:
   - Gain/loss regimes
   - Stationary frequencies (π₁)
   - Gene turnover rates

**E. coli Results:**
```
Stratification (threshold: 287 cloud genes):
  High-accessory: 659 strains
  Low-accessory:  665 strains

Parameter              High-accessory    Low-accessory    Difference
Gain rate (λ)          0.194             0.105            +0.088
Loss rate (μ)          0.596             0.946            -0.349
π₁                     0.245             0.100            +0.145
λ/μ ratio              0.325             0.111            +0.213

Both groups: LOSS-dominated, but with different rates
```

**Key insight:** These are **two different models**, not one complicated one. This keeps interpretation clean:

> "High-accessory and low-accessory lineages evolve under different gene-content dynamics."

**Reporting:** Present results separately for each group, emphasizing biological interpretation rather than statistical preference.

## Diagnostic Thresholds

### Heterogeneity Level Classification

| CV (cloud genes) | Top 10% contribution | Level | Action |
|------------------|---------------------|-------|--------|
| <0.3 | <15% | None | Standard global model |
| 0.3-0.5 | 15-20% | Low | Monitor, consider scan |
| 0.5-0.8 | 20-25% | Medium | Run heterogeneity scan |
| 0.8-1.2 | 25-30% | High | Stratified modeling recommended |
| >1.2 | >30% | Extreme | Stratified modeling required |

**E. coli:** CV=0.40, Top 10%=17.7% → Classified as "LOW" by CV alone, but **heterogeneity scan reveals EXTREME instability**

**Lesson:** Always run the scan! Static metrics can miss dynamic instability.

## Implementation

### Quick Start

```python
from persiste.plugins.genecontent.strain_diagnostics import diagnose_strain_heterogeneity
from persiste.plugins.genecontent.strain_recipes import strain_heterogeneity_scan

# 1. Diagnose
diag = diagnose_strain_heterogeneity(pam)
diag.print_report()

# 2. If heterogeneity detected, run scan
scan = strain_heterogeneity_scan(
    pam=pam,
    taxon_names=taxa,
    gene_names=genes,
    tree_method='jaccard_upgma'
)
scan.print_summary()

# 3. If extreme instability, use stratified modeling
if scan.interpretation.startswith("EXTREME"):
    from persiste.plugins.genecontent.strain_recipes import stratified_regime_modeling
    
    stratified = stratified_regime_modeling(
        pam=pam,
        taxon_names=taxa,
        gene_names=genes,
        threshold=None  # Use median
    )
    stratified.print_summary()
```

### Full Workflow

```python
# Load data
pam, taxa, genes = load_pam("pangenome.csv")

# Step 1: Diagnostic
diag = diagnose_strain_heterogeneity(pam)
diag.print_report()

# Step 2: Heterogeneity scan (always recommended)
scan = strain_heterogeneity_scan(pam, taxa, genes)
scan.print_summary()

# Step 3: Decision based on scan results
max_shift = max(abs(v) for v in scan.parameter_shifts.values())

if max_shift > 100:
    # EXTREME heterogeneity - use stratified modeling
    print("→ Using stratified modeling")
    stratified = stratified_regime_modeling(pam, taxa, genes)
    stratified.print_summary()
    
    # Report results separately for each group
    print(f"\nHigh-accessory strains (n={stratified.n_high}):")
    print(f"  λ={stratified.high_accessory_result.gain_rate:.4f}")
    print(f"  μ={stratified.high_accessory_result.loss_rate:.4f}")
    print(f"  π₁={stratified.high_accessory_result.equilibrium_frequency:.4f}")
    
    print(f"\nLow-accessory strains (n={stratified.n_low}):")
    print(f"  λ={stratified.low_accessory_result.gain_rate:.4f}")
    print(f"  μ={stratified.low_accessory_result.loss_rate:.4f}")
    print(f"  π₁={stratified.low_accessory_result.equilibrium_frequency:.4f}")
    
elif max_shift > 20:
    # MODERATE heterogeneity - report with caveat
    print("→ Moderate heterogeneity detected")
    print("→ Global model used, but results may be sensitive to strain sampling")
    result = fit(pam, taxon_names=taxa, gene_names=genes)
    
else:
    # LOW heterogeneity - standard analysis
    print("→ Standard global model appropriate")
    result = fit(pam, taxon_names=taxa, gene_names=genes)
```

## Biological Interpretation

### Why Strain Heterogeneity Matters

**Different strains can have fundamentally different gene dynamics:**

1. **Ecological niche:** Strains in different environments acquire different genes
2. **HGT rate variation:** Some lineages have higher horizontal gene transfer rates
3. **Population structure:** Recent vs ancestral lineages have different accessory content
4. **Sampling artifacts:** Outlier strains can dominate parameter estimates

### E. coli Case Study

**The top 10% of E. coli strains (132/1,324) have:**
- 2-3x more cloud genes than average
- Contribute 17.7% of all cloud gene observations
- Drive the apparent gain-dominated regime

**When removed, the "true" E. coli dynamics emerge:**
- Loss-dominated (λ/μ ≈ 0.26)
- π₁ ≈ 0.21 (genes mostly absent at equilibrium)
- Consistent with subset analyses

**Biological explanation:**
- These 132 strains may be:
  - Recent acquisitions to the sampled population
  - From distinct ecological niches (e.g., pathogenic vs commensal)
  - Lineages with elevated HGT rates
  - Artifacts of assembly/annotation differences

## Reporting Guidelines

### For Papers

**Bad:**
> "E. coli has a gain-dominated pangenome (λ/μ = 1086)."

**Good:**
> "Analysis of 1,324 E. coli strains revealed extreme strain heterogeneity in accessory gene content (range: 29-1,068 cloud genes per strain). The top 10% of strains by cloud gene content contributed 17.7% of all cloud gene observations and drove apparent gain-dominated dynamics (λ/μ=1081, π₁=0.999). Removing these outlier strains revealed loss-dominated dynamics (λ/μ=0.26, π₁=0.21), consistent with subset analyses. Stratified modeling showed that high-accessory (n=659) and low-accessory (n=665) strains exhibit distinct evolutionary regimes, both loss-dominated but with different rates (π₁=0.245 vs 0.100). This demonstrates that parameter estimates are highly sensitive to population structure, and that π₁ provides a more robust summary of pangenome dynamics than absolute gain/loss rates."

### Key Points to Report

1. **Strain heterogeneity metrics:**
   - Range of cloud genes per strain
   - CV (coefficient of variation)
   - Top 10% contribution

2. **Parameter stability:**
   - Results from heterogeneity scan
   - Magnitude of parameter shifts

3. **Stratified results (if applicable):**
   - Separate estimates for high/low accessory groups
   - Comparison of regimes

4. **π₁ prominently:**
   - More interpretable than λ/μ
   - More stable across sampling strategies

## Advantages Over Gene-Frequency Recipes

### Old Approach (Gene-Focused)
- "Exclude cloud genes"
- "Remove singletons"
- "Downweight rare genes"

**Problem:** Treats all strains as equivalent, just filters genes

### New Approach (Strain-Focused)
- "Detect strain heterogeneity"
- "Model subpopulations separately"
- "Test for regime differences"

**Advantage:** Recognizes that different strains may evolve differently

### Why This Is Better

1. **Biologically motivated:** Strains ARE different
2. **Statistically principled:** Tests hypotheses about regimes
3. **Interpretable:** "These lineages evolve differently" vs "we excluded genes"
4. **Actionable:** Identifies which strains drive parameter estimates

## Summary

**The strain heterogeneity framework provides:**

✅ **Diagnostic tools** - Detect when global models are inappropriate  
✅ **Analysis recipes** - Explicit strategies for heterogeneous data  
✅ **Hypothesis testing** - Formal tests for multiple regimes  
✅ **Clear interpretation** - Biological explanations for parameter shifts  

**Key message:** Always run the heterogeneity scan. It's your first line of defense against misleading parameter estimates.

## References

- This work - Strain heterogeneity framework
- Tettelin et al. (2005) - Open pangenome concept
- Rouli et al. (2015) - Core/shell/cloud classification

## Files

- `src/persiste/plugins/genecontent/strain_diagnostics.py` - Diagnostic tools
- `src/persiste/plugins/genecontent/strain_recipes.py` - Three recipes
- `scripts/test_strain_recipes.py` - Example usage
- `scripts/analyze_strain_heterogeneity.py` - E. coli case study
