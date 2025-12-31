# Cross-Plugin Analysis Recipes

This directory contains recipes that **orchestrate** multiple PersiSTE plugins for integrated analyses.

## Philosophy

**Single-plugin recipes** (in `src/persiste/plugins/*/recipes/`):
- Self-contained analyses using one plugin
- Plugin-specific questions
- Example: "Are copy number changes suppressed?" (CopyNumberDynamics only)

**Cross-plugin recipes** (in `recipes/`):
- **Orchestration protocols**, NOT joint likelihood models
- Run plugins sequentially, integrate results biologically
- Example: "Do genes with stable presence also have stable dosage?" (GeneContent + CopyNumberDynamics)

## Key Design Principle: Orchestration, Not Joint Models

Cross-plugin recipes are **analysis protocols**, not models:

1. **Run Plugin A** (e.g., GeneContent for retention classification)
2. **Run Plugin B conditional on A** (e.g., CopyNumberDynamics on present genes)
3. **Integrate results biologically** (pattern classification, interpretation)

### Why Not Joint Likelihoods?

For presence × dosage:
- **Presence/absence events** are rare but decisive
- **CN changes** are frequent but subtle
- **Forcing them into one CTMC dilutes both signals**

By separating them:
- GeneContent captures structural evolution (rare, decisive)
- CopyNumberDynamics captures quantitative tuning (frequent, subtle)
- Recipe ties them together **biologically**, not mathematically

This is more powerful than a joint model and mirrors how people actually think.

## Structure

```
recipes/                                    # Cross-plugin recipes (this directory)
├── README.md                              # This file
├── __init__.py                            # Exports cross-plugin recipes
├── gene_dosage_effect.py                  # GeneContent + CopyNumberDynamics
├── strain_heterogeneity.py                # Future: multi-plugin strain analysis
└── host_conditioned_dosage.py             # Future: environment-dependent dosage

src/persiste/plugins/copynumber/recipes/   # Single-plugin recipes
├── recipe_0_null.py                       # Null CN dynamics
├── recipe_1_dosage_stability.py           # Dosage stability scan
├── recipe_2_amplification_bias.py         # Amplification bias test
├── recipe_3_lineage_volatility.py         # Lineage-conditioned volatility
└── recipe_4_joint_presence_dosage.py      # Interface definition (future)

src/persiste/plugins/genecontent/recipes/  # Single-plugin recipes (future)
├── recipe_0_null.py                       # Null gene content dynamics
├── recipe_1_conservation.py               # Gene conservation/lability
└── recipe_2_clade_specific.py             # Clade-specific gene content
```

## Current Status

**v1 (implemented):**
- CopyNumberDynamics single-plugin recipes (0-3)
- Cross-plugin recipe: `gene_dosage_effect` (orchestration framework)

**v2 (planned):**
- Full GeneContent integration in `gene_dosage_effect`
- Additional cross-plugin recipes (strain heterogeneity, host-conditioned dosage)
- GeneContent single-plugin recipes

## Usage

### Single-Plugin Recipes

Import from their respective plugins:

```python
from persiste.plugins.copynumber.recipes import (
    null_cn_dynamics,
    dosage_stability_scan,
    amplification_bias_test,
)

# Run single-plugin analysis
report = dosage_stability_scan(
    cn_matrix="data/cn_matrix.tsv",
    tree="data/tree.nwk",
)
```

### Cross-Plugin Recipes

Import from top-level `recipes/`:

```python
from persiste.recipes import gene_dosage_effect

# Run cross-plugin orchestration
report = gene_dosage_effect(
    cn_matrix="data/cn_matrix.tsv",
    family_names=families,
    taxon_names=taxa,
    tree="data/tree.nwk",
)

# Access integrated results
essential_genes = [
    fam for fam, pattern in report.integrated_patterns.items()
    if pattern == "essential_dosage_sensitive"
]
```

## Example: Gene Dosage Effect Recipe

This recipe orchestrates GeneContent + CopyNumberDynamics:

**Step 1: Gene Retention Analysis (GeneContent)**
- Classify genes: core-like, accessory-like, lineage-restricted
- Output: retention regime, π₁, λ, μ

**Step 2: Conditional CN Analysis (CopyNumberDynamics)**
- Analyze CN dynamics only where gene is present
- Test: dosage stability, amplification bias

**Step 3: Integrative Interpretation**
- High retention + strong buffering → Essential dosage-sensitive
- High retention + amplification bias → Adaptive dosage modulation
- Low retention + high amplification → Mobile elements / selfish genes

**Result:** Biological pattern classification without joint likelihood

## Design Principles

1. **Interpretable results** - Every recipe produces human-readable interpretation
2. **Biological framing** - Descriptive, not causal statements
3. **Clear scope** - Each recipe answers one specific question
4. **Composable** - Recipes can be chained in workflows
5. **Production code** - No test-only shortcuts

## See Also

- `src/persiste/plugins/copynumber/examples/recipe_demo.py` - Complete workflow example
- `src/persiste/plugins/copynumber/validation/VALIDATION_RESULTS_v2.md` - Validation details
- `src/persiste/plugins/copynumber/README.md` - Plugin documentation
