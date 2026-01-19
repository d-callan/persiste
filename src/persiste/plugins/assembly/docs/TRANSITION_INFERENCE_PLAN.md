# Plan: Transition-Aware Assembly Inference

## 1. Problem Statement
The current "Presence/Absence" model is insufficient for detecting "lifelike" signals (e.g., extreme reuse, founder constraints) because it treats observations as independent snapshots without considering the assembly process. Lifelike systems are characterized by specific *trajectories* through state space, where complex states depend on the prior existence of specific subassemblies.

## 2. Persiste Terminology: Constraints vs. Data
To maintain consistency with the `persiste` framework, we define:
- **Constraints**: Features we test for by comparing a constrained model against a baseline (neutral) model. Examples: `reuse_count`, `founder_bias`, `depth_gate`.
- **Primary Data**: Observed counts/abundances of compounds (e.g., from mass spec/metabolomics). This is the "snapshot" we aim to explain.
- **Structural Dependencies**: Inherent rules of Assembly Theory (e.g., subassembly dependency). These are **not** constraints themselves, but rather the "physics" of the system that the data must follow.

## 3. Inferred Trajectory Likelihood
Since real-world data often lacks explicit time or discovery order, we must **infer** the most likely assembly paths that explain the observed snapshot data.

### Key Insights
- **Assembly Index as Structural Data**: A compound with index $N$ cannot appear until all its required subassemblies exist. This is a hard structural requirement that the likelihood model uses to constrain valid paths.
- **Abundance as Latent Timing**: Counts in a snapshot are treated as a proxy for steady-state occupancy. A high-index compound with high abundance implies a highly efficient (likely constrained) assembly path.
- **Power of Reuse**: Lifelike systems show high likelihood when trajectories "jump" to high complexity once a key "founder" motif (a specific constraint) is active.

### Proposed Observation Model
We model the **Joint Probability of the Observed Abundance Snapshot**:
$$P(\text{Counts} | \theta) = \int_{\tau \in \text{Paths}} P(\text{Counts} | \tau) P(\tau | \theta) d\tau$$
Where $P(\tau | \theta)$ is the probability of a trajectory given the active **constraints**.

## 4. Latent Information Extraction
The likelihood model extracts latent information from the static abundance snapshot:
1. **Topological Sort**: Sort observed compounds by assembly index to define "valid" discovery windows.
2. **Path Mapping**: Identify "critical paths" in the assembly graph that must have been traversed.
3. **Density Contrast**: Compare observed abundance distributions against neutral baseline expectations to detect where specific **constraints** (like reuse) significantly improve the fit.

## 4. Implementation Roadmap

### Phase 1: Rust Backend Expansion (Completed)
- **PathStats**: Added `state_trajectory` and `arrival_times` to track the full sequence of state discoveries.
- **Python API**: Exposed these fields in `simulate_assembly_trajectories`.

### Phase 2: Counts-First Observation Model (`counts_model.py`)
- **Poisson Counts**: Model observed abundances directly using a Poisson likelihood centered on latent state occupancy.
- **Extensible Diagnostics**: Extract features (reuse, depth, etc.) directly from the weighted latent state distribution to avoid hardcoding constraint-specific logic in the likelihood.
- **Founder vs Derived Reuse Diagnostics**: 
  - *Founder Reuse*: Reuse events involving low-depth states (proxy for early discovery).
  - *Derived Reuse*: Reuse events involving high-depth states.
- **Deliverables**: `AssemblyCountsModel` providing both log-likelihood and structural diagnostics.

### Phase 3: Heterogeneity & Mixture Models
- **Mixture Detection**: Implement a recipe to detect if the data is a mixture of multiple constraint patterns (e.g., a "biotic" signal superimposed on an "abiotic" background).
- **Inferred Heterogeneity**: Use the discrepancy between predicted arrival times and observed abundances to flag "outlier" compounds that don't fit the dominant $\theta$ pattern.
- **EM Optimization**: Cluster observations into $K$ groups, each with its own $\theta$ vector.

### Phase 5: Lifelike Classifier Recipe
- Compute the Bayes Factor: $K = \frac{P(O | \theta_{lifelike})}{P(O | \theta_{null})}$.
- Significant signal in `reuse_count` or `founder_bias` indicates lifelike chemistry.
- **Heterogeneity Score**: Report a metric of how "consistent" the snapshot is with a single assembly process versus a mixture of processes.

## 5. Detailed Implementation Notes

### Transition Matrix Caching
To keep likelihood calculations tractable, we will cache the transition rates $Q(s_i, s_j | \theta)$ for all observed states and their immediate neighbors. This allows fast re-weighting without full re-simulation.

### Handling "Hidden" States
The likelihood must account for states that *must* have existed (as intermediates) but were not observed in the final snapshot. The assembly index provides the lower bound for these hidden states.

### Snapshot Likelihood Algorithm
1. Sample $N$ trajectories $\{\tau_i\}$ from Rust using $\theta_{ref}$.
2. For each trajectory, calculate the "Set Discovery Time" $T_{discovery}(O) = \max_{s \in O} (\text{arrival\_time}(s))$.
3. The likelihood is higher if $T_{discovery}(O)$ is small and if the path features (reuse, etc.) match the $\theta$ being evaluated.
