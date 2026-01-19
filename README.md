# PERSISTE

**P**ersistence **E**vidence via **R**ate **S**ignatures **I**n **S**tate **T**ransition **E**volution

A generalized, **exploratory** framework for detecting constraint signatures in life-like systems, inspired by HyPhy's statistical philosophy but conceptually reframed around baseline vs. constraint.

## Overview

PERSISTE detects constraint by comparing observed transition rates against a baseline generative process. The framework is domain-agnostic—plugins provide domain-specific instantiations.

**Current status:** this repository is primarily an exploration of the underlying framework ideas. The plugins should be treated as research prototypes, not robust scientific software.

- **Phylogenetics** (`persiste-phylo`): FEL-like selection analysis as a **non-rigorous proof-of-concept** that the reframed framework can reproduce HyPhy-adjacent workflows. It is **not** intended to be a replacement for HyPhy.
- **Assembly Theory** (`persiste-assembly`): an exploratory sandbox driven by curiosity about assembly theory, and an example of framework flexibility.
- **Gene content** (`persiste-genecontent`): Pangenome gain/loss dynamics with strain heterogeneity framework.
- **Copy number**: (`persiste-copynumber`):copy number dynamics

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

## Phylo Plugin – Proof-of-Concept Interface

The phylo plugin now mirrors the assembly interface pattern: you provide a tree and codon alignment, and the shared `ConstraintInference` engine fits the global ω (dN/dS) parameter via the MG94 baseline and the `PhyloCTMCObservationModel`.

```python
from persiste.core.trees import TreeStructure
from persiste.plugins.phylo import (
    CodonStateSpace,
    PhyloModelConfig,
    fit_global_omega,
    sequences_to_codon_alignment,
)

# 1) Prepare inputs
tree = TreeStructure.from_newick("((A:0.1,B:0.1):0.2,(C:0.1,D:0.1):0.2);")
codon_space = CodonStateSpace.universal()
sequences = {
    "A": "ATGATGATG",
    "B": "ATGTTTATG",
    "C": "ATGATGTTT",
    "D": "ATGTTTTTT",
}
alignment, taxa = sequences_to_codon_alignment(sequences, codon_space, tree.tip_names)

# 2) Fit the global ω constraint using shared inference APIs
result = fit_global_omega(
    tree=tree,
    alignment=alignment,
    config=PhyloModelConfig(initial_omega=1.0, kappa=2.0),
    inference_kwargs={"options": {"maxiter": 25}},
)

print("ω̂ =", result.parameters["omega"])
print("log-likelihood =", result.log_likelihood)
```

Key helper functions exposed through `persiste.plugins.phylo`:

1. `sequences_to_codon_alignment` – convert FASTA-like mappings into codon index matrices with explicit taxon ordering.
2. `load_codon_alignment` – read codon FASTA files and convert them to alignments in one step.
3. `build_phylo_components` – construct the codon space, MG94 baseline, transition graph, and `PhyloCTMCObservationModel`.
4. `fit_global_omega` – configure the MG94 + OmegaConstraint + PhyloCTMC pipeline and dispatch to `ConstraintInference`.

These helpers keep the proof-of-concept focused while ensuring the phylo workflow exercises the same APIs that other plugins rely on.

## GeneContent Plugin

The **GeneContent plugin** provides phylogenetic analysis of pangenome gain/loss dynamics with a novel strain heterogeneity framework.

### Quick Start

```python
from persiste.plugins.genecontent.pam_interface import fit
from persiste.plugins.genecontent.recipes import run_heterogeneity_diagnostic

# Load your presence/absence matrix
result = fit(
    pam="pangenome.csv",
    taxon_names=taxa,
    gene_names=genes,
    tree_method='jaccard_upgma'
)

# Check for strain heterogeneity (ALWAYS RECOMMENDED)
scan = run_heterogeneity_diagnostic(pam, taxa, genes)
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
│       │   ├── recipes/               # User-facing recipe package
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

### Beta 
- **GeneContent plugin** - Pangenome gain/loss analysis with strain heterogeneity framework
- **Core framework** - General baseline vs. constraint pattern

## Alpha
- ** CopyNumber plugin** - Copy number dynamics

### Exploratory / Pre-alpha
- **Phylo plugin (FEL)** - Lightweight validation that framework can support HyPhy-adjacent workflows
- **Assembly plugin** - Exploratory implementation for assembly theory

If you want to use exploratory plugins for serious scientific inference, plan to independently validate results and assumptions.

## License

MIT (TBD)

