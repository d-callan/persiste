# PERSISTE

**P**ersistence **E**vidence via **R**ate **S**ignatures **I**n **S**tate **T**ransition **E**volution

A generalized, **exploratory** framework for detecting constraint signatures in life-like systems, inspired by HyPhy's statistical philosophy but conceptually reframed around baseline vs. constraint.

## Overview

PERSISTE detects constraint by comparing observed transition rates against a baseline generative process. The framework is domain-agnostic—plugins provide domain-specific instantiations.

**Current status:** this repository is primarily an exploration of the underlying framework ideas. The plugins should be treated as research prototypes, not robust scientific software.

- **Phylogenetics** (`persiste-phylo`): FEL-like selection analysis as a **non-rigorous proof-of-concept** that the reframed framework can reproduce HyPhy-adjacent workflows. It is **not** intended to be a drop-in replacement for HyPhy.
- **Assembly Theory** (`persiste-assembly`): an exploratory sandbox driven by curiosity about assembly theory, and an example of framework flexibility.

- **Gene content** (`persiste-genecontent`): Pangenome gain/loss dynamics with strain heterogeneity framework. **Production-ready** for research use.

Coming soon (exploratory):

- **Copy number**: copy number dynamics

## What this project is (and isn’t)

- **Exploration / prototyping**
  The goal is to explore a general “baseline vs. constraint” inference pattern across domains.
- **Not production-ready**
  Expect incomplete APIs, rough edges, and changing interfaces.
- **Not yet robust**
  The methods and implementations here should not be treated as fully validated or publication-grade without independent verification.

## Core Concept

For each transition `r` between states:
- **Baseline rate** `λ_r`: what would happen without constraint
- **Constraint parameter** `θ_r`: multiplicative modifier on baseline
- **Effective rate**: `λ_r · θ_r`

Where:
- `θ_r = 1` → unconstrained
- `θ_r < 1` → suppressed (constraint)
- `θ_r > 1` → facilitated

## Installation

```bash
# This is currently a research prototype repository.
# Installation is typically via editable installs during development.

pip install -e .
```

## Quick Start

```python
import persiste as ps
"""Illustrative sketch (API is evolving)."""

# Define state space
states = ps.StateSpace.from_list(['A', 'B', 'C'])

# Define transition graph (topology)
graph = ps.TransitionGraph.complete(states)

# Define baseline process (opportunity)
baseline = ps.Baseline.uniform(rate=1.0)

# Define constraint model (hypothesis)
model = ps.ConstraintModel(
    states=states,
    baseline=baseline,
    graph=graph,
    constraint_structure='per_transition',
    allow_facilitation=True
)

# Load data (observed counts)
data = ps.ObservedTransitions(
    counts={('A', 'B'): 10, ('B', 'C'): 2},
    exposure=1.0
)

 # Define observation model (statistics)
obs_model = ObservationModel(graph)  # placeholder

# Fit model (inference)
# Dispatches to ConstraintInference engine
result = model.fit(data, obs_model=obs_model, method='MLE')

print(f"Log-likelihood: {result.log_likelihood}")
print(f"AIC: {result.aic}")

# Inspect fitted parameters
print("Fitted parameters:", result.parameters)

# Test for constraint (hypothesis testing)
null_model = ps.ConstraintModel(states, baseline, graph) # Unconstrained
null_result = null_model.fit(data, obs_model=obs_model)

lrt = model.test(
    data,
    null_result=null_result,
    alternative_result=result,
    obs_model=obs_model,
    method='LRT'
)

print(f"p-value: {lrt.pvalue}")
```

## GeneContent Plugin - Production Ready

The **GeneContent plugin** provides phylogenetic analysis of pangenome gain/loss dynamics with a novel strain heterogeneity framework.

### Quick Start

```python
from persiste.plugins.genecontent.pam_interface import fit
from persiste.plugins.genecontent.strain_recipes import strain_heterogeneity_scan

# Load your presence/absence matrix
result = fit(
    pam="pangenome.csv",
    taxon_names=taxa,
    gene_names=genes,
    tree_method='jaccard_upgma'
)

# Check for strain heterogeneity (ALWAYS RECOMMENDED)
scan = strain_heterogeneity_scan(pam, taxa, genes)
scan.print_summary()
```

### Key Features

- **Strain heterogeneity detection** - Identifies when parameter estimates are driven by outlier strains
- **Two-recipe framework** - Diagnostic scan + stratified modeling for heterogeneous datasets
- **Rust acceleration** - 5-6x speedup for large datasets
- **Validated on real data** - E. coli case study with 1,324 strains

### Documentation

- **[STRAIN_HETEROGENEITY_FRAMEWORK.md](STRAIN_HETEROGENEITY_FRAMEWORK.md)** - Complete framework guide
- **[src/persiste/plugins/genecontent/README.md](src/persiste/plugins/genecontent/README.md)** - Plugin documentation
- **[src/persiste/plugins/genecontent/examples/](src/persiste/plugins/genecontent/examples/)** - Example scripts and workflows

## Repository Structure

```
persiste/
├── src/persiste/
│   ├── core/                          # Core framework (exploratory)
│   └── plugins/
│       ├── genecontent/               # GeneContent plugin (production-ready)
│       │   ├── pam_interface.py       # Main interface
│       │   ├── strain_diagnostics.py  # Heterogeneity diagnostics
│       │   ├── strain_recipes.py      # Two-recipe framework
│       │   ├── examples/              # Example scripts & workflows
│       │   ├── exploratory/           # Development/validation scripts
│       │   └── README.md
│       ├── assembly/                  # Assembly Theory plugin (exploratory)
│       │   └── examples/              # Assembly examples
│       └── phylo/                     # Phylogenetics plugin (proof-of-concept)
├── docs/                              # Documentation archive
│   ├── benchmarks/                    # Performance comparisons
│   └── development/                   # Development history
├── rust/                              # Rust acceleration
└── STRAIN_HETEROGENEITY_FRAMEWORK.md  # Main framework documentation
```

## Development Status

### Production-Ready
- **GeneContent plugin** - Pangenome gain/loss analysis with strain heterogeneity framework

### Exploratory / Pre-alpha
- **Phylo plugin (FEL)** - Lightweight validation that framework can support HyPhy-adjacent workflows
- **Assembly plugin** - Exploratory implementation for assembly theory
- **Core framework** - General baseline vs. constraint pattern (evolving)

If you want to use exploratory plugins for serious scientific inference, plan to independently validate results and assumptions.

## License

MIT (TBD)

## Citation

(TBD)
