# Assembly Plugin Examples

This directory contains examples demonstrating the Assembly Theory plugin.

## Contents

### Basic Demos
- **`assembly_demo.py`** - Basic assembly plugin demonstration
- **`assembly_interface_demo.py`** - Interface usage examples
- **`assembly_graph_demo.py`** - Assembly graph examples
- **`assembly_observation_demo.py`** - Observation model examples
- **`assembly_dynamics_demo.py`** - Dynamics examples

### Inference Examples
- **`assembly_inference_demo.py`** - Detailed inference workflow
- **`assembly_inference_mle.py`** - MLE inference examples
- **`assembly_full_demo.py`** - Complete workflow demonstration

### Validation & Testing
Validation scripts and results now live under **`plugins/assembly/validation/`**:

- **`validation/scripts/`** – Shared utilities (benchmark runner, dataset generation)
- **`validation/experiments/`** – Reboot validation suites (null recovery, robustness scenarios)
- **`validation/results/`** – Canonical outputs/logs for the current iteration

See that directory’s README for the latest workflow details.

### Analysis & Visualization
- **`assembly_model_comparison.py`** - Model comparison examples
- **`assembly_scaling_curves.py`** - Scaling curve analysis
- **`assembly_plot_scaling.py`** - Scaling visualization

## Note

The Assembly plugin is **exploratory** and not production-ready. These examples are preserved for reference and development purposes.

For production-ready pangenome analysis, see the **GeneContent plugin**:
- `/src/persiste/plugins/genecontent/examples/`
- `/STRAIN_HETEROGENEITY_FRAMEWORK.md`
