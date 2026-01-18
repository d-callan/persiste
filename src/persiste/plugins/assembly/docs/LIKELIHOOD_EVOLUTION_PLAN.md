# Likelihood Evolution Plan: From Static Snapshots to Generative Dynamics

This document outlines the strategic evolution of the assembly likelihood framework. We have successfully established a **Zero-Order Generative Likelihood** (Static Snapshot), which restores identifiability for presence/absence data. The next phase involves layering **First-Order** and **Second-Order** generative components to capture the rich, life-like dynamics of assembly processes.

## 1. Conceptual Framework

We view the likelihood function not as a monolithic regression target, but as a layered evaluation of increasingly rich data features.

$$ \log P(\text{Data} \mid \theta) = \underbrace{\log P(\text{States} \mid \theta)}_{\text{Zero-Order (Current)}} + \underbrace{\log P(\text{Counts} \mid \text{States}, \theta)}_{\text{First-Order (Next)}} + \underbrace{\log P(\text{Trajectory} \mid \theta)}_{\text{Second-Order (Future)}} $$

### Current State: The Static Snapshot (Zero-Order)
*   **Model:** Data is viewed as a set of terminal states sampled from the stationary distribution of the process under constraint $\theta$.
*   **Mechanism:** $\theta$ reweights the probability mass of the state space (e.g., boosting reused or deep states).
*   **Question Answered:** "Are the observed chemical species likely to arise from this reweighted distribution?"
*   **Strengths:** Excellent for compositional fingerprints ("what exists"). Fits the Persiste/HyPhy philosophy of reweighted substitution models.
*   **Limitations:** Ignores abundance, path multiplicity, time-ordering, and autocatalytic feedback.

---

## 2. Evolution Roadmap

### Phase 1: Generative Flux & Abundance (First-Order)
**Goal:** Capture the dynamic *flux* of the system, not just the static presence of states. This aligns with Persiste's core philosophy: modeling transitions between states.

*   **Mechanism:** Constraints act as biases on **transition rates**.
    *   **Input:** Any constraint (Reuse, Depth, Founder Bias, Class Consistency) modifies the probability of specific assembly steps.
    *   **Process:** These modified local transitions integrate over time to produce a global **flux** through the assembly graph.
    *   **Output:** The model predicts an expected **abundance** $\lambda_i(\theta)$ for every state $i$.
*   **Why this replaces summaries:**
    *   We do not need to calculate ad-hoc summaries like "mean reuse" or "average depth."
    *   The abundance vector $\vec{\lambda}(\theta)$ *is* the sufficient statistic. It naturally incorporates all constraints:
        *   *Reuse Constraint* $\to$ Amplifies flux to reused states.
        *   *Depth Constraint* $\to$ Shifts flux toward (or away from) deep states.
        *   *Founder Bias* $\to$ Concentrates flux in early-diverging lineages.
    *   Likelihood becomes $P(\text{Observed Counts} \mid \vec{\lambda}(\theta))$, capturing the full generative signature.

### Phase 2: Trajectory Structure (Second-Order)
**Goal:** Capture path dependence and irreversibility.

*   **Mechanism:** Evaluate the likelihood of the *pathways* implied by the data, not just the endpoints.
*   **Features:**
    *   **Transition Frequencies:** Multinomial likelihood over transition types (Join vs. Split vs. Decay).
    *   **Path Multiplicity:** High-reuse objects should have many incoming paths.
    *   **Depth Progression:** Expectation of how mass flows from shallow to deep over time.

---

## 3. Alignment with "Life-Like" Physics

This evolution directly addresses the gaps in identifying life-like chemistry:

| Feature | Current Model (Static) | Future Model (Dynamic) |
| :--- | :--- | :--- |
| **Selection** | ✅ Reweighted static probability | ✅ Reweighted probability |
| **Autocatalysis** | ❌ Indistinguishable from stability | ✅ Captured by overdispersed counts (burstiness) |
| **Irreversibility** | ❌ Equilibrium assumption | ✅ Trajectory/Transition likelihoods |
| **Heterogeneity** | ❌ Single global distribution | ✅ Mixture models / Per-trajectory latent classes |

## 4. Practical Implementation Plan

We will proceed incrementally, preserving the stability of the current fix.

### Immediate Term (Current)
*   **Action:** Keep the current "Static Snapshot" fix.
*   **Justification:** It provides a correct, principled baseline for identifiability. It separates "diffuse" (non-life) from "concentrated" (life-like) distributions.
*   **Validation:** Run deep simulations (depth 15-20) now. The current model is sufficient to detect the concentration of probability mass characteristic of selection.

### Near Term (Next Upgrade)
*   **Action:** Implement **Generalized Generative Abundance (Flux) Likelihood**.
*   **Concept:** Abandon "feature summaries" entirely.
    *   In Persiste, constraints (Reuse, Depth, Founder Bias, Class Consistency) act by modifying **transition rates**.
    *   Modified transitions alter the **flux** through the assembly graph.
    *   Flux determines the expected **abundance** $\lambda_i(\theta)$ of every specific state $i$.
*   **Mechanism:**
    1.  **Input:** The simulator runs with full constraints $\theta$.
    2.  **Output:** It produces expected abundances $\lambda_i(\theta)$ for all states, reflecting the integrated effect of *all* constraints (e.g., founder-biased states become abundant; deep states become rare or common depending on $\theta$).
    3.  **Likelihood:** Compare observed counts $k_{obs,i}$ directly to expected flux $\lambda_i(\theta)$ for all species $i$.
        $$ \log P(\text{Counts} \mid \theta) = \sum_{i \in \text{Observed}} \log P_{\text{count}}(k_{obs,i} \mid \lambda_i(\theta)) $$
*   **Why this meets requirements:**
    *   **All Constraints Included:** Whether the constraint is "Reuse" or "Symmetry Break B", its effect is fully captured in the resulting abundances.
    *   **No Summaries:** We never calculate "mean depth" or "average reuse". We just check if the specific high-abundance states predicted by the model match the data.
    *   **Models Transitions:** The abundances are the direct integration of the transition dynamics.

*   **Hypothesis Testing (Evidence accumulation):**
    *   How do we know if a specific constraint (e.g., "Reuse") is at play?
    *   We compare models using Likelihood Ratios: $LR = \frac{P(\text{Data} \mid \text{Flux}(\theta_{\text{reuse}}))}{P(\text{Data} \mid \text{Flux}(\theta_{\text{null}}))}$
    *   If the data contains high counts of reused parts, the "Reuse" flux model will predict those high counts correctly, yielding a higher likelihood than the null model (which predicts random diffuse counts).
    *   **Result:** We get a direct quantitative measure of evidence (Delta Log-Likelihood) for each physical mechanism.

### Long Term (Research)
*   **Action:** Investigate **Trajectory Likelihoods**.
*   **Conceptual Shift:** From $P(\text{State} \mid \theta)$ to $P(\text{History} \mid \theta)$.
    *   Current model asks: "Is this object stable/likely?"
    *   Trajectory model asks: "Is the pathway to create this object plausible?"
*   **Implementation Details:**
    1.  **Rust Simulator Upgrade:**
        *   Modify `simulate_assembly_trajectories` to return full **lineage trees** or **path histories**, not just final state IDs.
        *   *Data Structure:* `PathHistory` struct containing a sequence of `(state_id, transition_type, time_delta)` tuples.
    2.  **Path Integral / HMM Approach:**
        *   Since we observe only the final product $S_{\text{obs}}$, the true history is hidden.
        *   Likelihood becomes a sum over possible paths $\gamma$:
            $$ P(S_{\text{obs}} \mid \theta) = \sum_{\gamma \to S_{\text{obs}}} P(\gamma \mid \theta) $$
        *   *Approximation:* Instead of summing all paths (intractable), we use the cached paths from the simulator as an importance sample of the path space.
    3.  **Transition Probabilities:**
        *   Decompose path probability into individual steps:
            $$ P(\gamma \mid \theta) = \prod_{t} P(\text{transition}_t \mid \text{state}_t, \theta) $$
        *   This explicitly captures irreversibility (probability of A+B→C vs C→A+B).
    4.  **Asymmetry Detection:**
        *   This framework allows detecting "ratchets" — steps that are easy to take forward but hard to reverse, which creates the "time arrow" of life-like systems.

## 5. Deprecation & Cleanup

To support the move to a pure generative flux model, we must remove legacy code that relies on ad-hoc feature summaries. This code is now obsolete because the physical constraints are handled upstream in the simulation, not downstream in the likelihood scoring.

### Targets for Removal
*   **CLI Helpers (`src/persiste/plugins/assembly/cli.py`):**
    *   `_extract_feature_summary`: Computes global mean/variance of reuse from baseline.
    *   `_compute_constraint_features`: Weights feature stats by latent probabilities.
    *   `observation_summary` argument passing in `fit_assembly_constraints`.
*   **Likelihood Logic (`src/persiste/plugins/assembly/likelihood.py`):**
    *   `observation_summary` parameter in `compute_observation_ll`.
    *   Global fallback logic using `global_reuse_mean` and `variance_reuse_count`.
    *   Ad-hoc Gaussian scoring for "reuse conditioned on depth" inside the enriched block.
*   **Tests:**
    *   Any tests specifically validating the "summary statistics" extraction or the "Gaussian reuse score" logic.

### Goal State
The likelihood function should accept `observed_counts` and `expected_flux` (derived from `latent_states`), and nothing else. All physical intuition should be encoded in the *simulation* (which produces the flux), not in the *likelihood function* (which simply evaluates the match).

## 6. Summary

We are moving from asking **"Is this molecule possible?"** to asking **"Is this process plausible?"**.

The current fix answers the first question correctly for the first time. The next steps will layer on the answers to the second.
