# PERSISTE

**P**ersistence **E**vidence via **R**ate **S**ignatures **I**n **S**tate **T**ransition **E**volution

A generalized, **exploratory** framework for detecting constraint signatures in life-like systems, inspired by HyPhy's statistical philosophy but conceptually reframed around baseline vs. constraint.

## Overview

PERSISTE detects constraint by comparing observed transition rates against a baseline generative process. The framework is domain-agnostic—plugins provide domain-specific instantiations.

**Current status:** this repository is primarily an exploration of the underlying framework ideas. The plugins should be treated as research prototypes, not robust scientific software.

- **Phylogenetics** (`persiste-phylo`): FEL-like selection analysis as a **non-rigorous proof-of-concept** that the reframed framework can reproduce HyPhy-adjacent workflows. It is **not** intended to be a drop-in replacement for HyPhy.
- **Assembly Theory** (`persiste-assembly`): an exploratory sandbox driven by curiosity about assembly theory, and an example of framework flexibility.

Coming soon (exploratory):

- **Gene content**: gain/loss dynamics
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

## Documentation

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed design and roadmap.

## Development Status

🚧 **Pre-alpha / exploratory**

- **Phylo plugin (FEL)**
  Intended as a lightweight, not-very-rigorous validation that the framework can support HyPhy-adjacent workflows under a different conceptual framing.
- **Assembly plugin**
  An exploratory implementation motivated by curiosity about assembly theory and as a stress-test of framework flexibility.

If you want to use this for serious scientific inference, plan to independently validate results and assumptions.

## License

MIT (TBD)

## Citation

(TBD)
