# PERSISTE

**P**ersistence **E**vidence via **R**ate **S**ignatures **I**n **S**tate **T**ransition **E**volution

A generalized framework for detecting constraint signatures in life-like systems, inspired by HyPhy's statistical philosophy but built for modern compute infrastructure.

## Overview

PERSISTE detects constraint by comparing observed transition rates against a baseline generative process. The framework is domain-agnostic—plugins provide domain-specific instantiations for:

- **Phylogenetics** (`persiste-phylo`): HyPhy-compatible selection analysis
- **Assembly Theory** (`persiste-assembly`): Chemical constraint in prebiotic systems
- **Viral Quasispecies** (`persiste-quasispecies`): Population-level constraint
- **Ecological Networks** (`persiste-ecology`): Interaction constraint

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
# Core framework
pip install persiste

# Plugins
pip install persiste-phylo
pip install persiste-assembly
```

## Quick Start

```python
import persiste as ps
from persiste.core import ConstraintInference, PoissonObservationModel

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
obs_model = ObservationModel(graph)

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

🚧 **Pre-alpha**: Core architecture in development

## License

MIT (TBD)

## Citation

(TBD)
