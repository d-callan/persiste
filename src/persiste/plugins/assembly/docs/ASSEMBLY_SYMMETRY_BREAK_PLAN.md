# Assembly Symmetry-Break Implementation Plan

This document captures the end-to-end plan for introducing controlled asymmetries into the Assembly plugin and wiring them through simulator, constraints, observations, and inference (screening + stochastic phases).

## 1. Symmetry Break A — Depth-Gated Reuse (Time/Depth Non-Stationarity)
- **Status:** ✅ Simulator + dataset + stochastic likelihood + sweep verification complete. ΔLL now averages ~222 across 3 replicates when using enriched observation stats (depth × reuse) and the stochastic phase’s latent-weighted likelihood.
- **Key lessons:** Signals only appear once the likelihood terms are weighted by θ-dependent latent probabilities and the sweep reads the stochastic ΔLL; the deterministic screening approximation still misses this symmetry break because the non-stationarity only exists inside the simulator.
- **Next:** (a) Decide whether to expose a depth-gate-inspired feature at the constraint layer so screening can approximate it, or (b) allow “stochastic-only” inference mode in benchmarking so we do not block on deterministic upgrades.

### 1.1 Current Identifiability Snapshot (Jan 2026)
- **Depth-gated dataset (pr8 depth7 traj200 signal):** ΔLL ≈ 222, recovery = 100 %. Both screening and stochastic likelihoods accept enriched stats, but only the stochastic phase “sees” the depth gate. `stochastic_delta_ll` is now surfaced in sweeps so ΔLL lift is observable even when screening stays flat.
- **Legacy robustness tests (`assembly_robustness_tests.py`):** All four scenarios (wrong baseline, missing states, noisy counts, false positives) still pass with broadened-but-identifiable likelihood profiles and no spurious constraints.
- **Power-grid smoke dataset (`pr4_depth3_traj100_presence_null`):** Presence-only dataset still yields recovery = 1.0 once the likelihood tolerates `frequency=None` and we rely on stochastic ΔLL.
- **Baseline misspecification smoke scenarios (correct/mild/severe):** Sweeps distinguish the tiers—only the “correct” scenario passes the ΔLL threshold after the recent fixes; mild/severe remain below safety-adjusted thresholds.

### 1.2 Requirements for Sufficient Power (No Further Asymmetries Yet)
- Observation records must carry per-compound `mean_max_depth`, `mean_reuse_count`, and `frequency` (nullable). Null-safe defaults + latent-probability weighting are essential; otherwise ΔLL collapses to zero.
- Deterministic screening remains blind to simulator-only asymmetries. Until depth-bin constraint features exist, benchmarking should either skip screening or treat `stochastic_delta_ll` as the authoritative score.
- Importance-sampling cache health (trust radius, ESS) remains the gating factor for stochastic runtime; no regressions observed after the latest sweeps.

## 2. Symmetry Break B — Context-Dependent Reuse (Compound Classes)
| Layer | Implementation Plan |
| --- | --- |
| Simulator | ✅ Added optional `context_class_config` knobs to the Rust simulator + Python bindings. Generators can now pass primitive→class maps plus `same_class_theta` / `cross_class_theta` to make reuse bonuses/penalties class-aware without touching inference. Generator: `generate_context_class_dataset.py --same-class-theta 0.35 --cross-class-theta -0.25`. |
| Constraints | Add features such as `reuse_same_class`, `reuse_cross_class`, `class_switch_penalty`. Introduce a class-compatibility helper once we define its semantics. |
| Observations | ✅ `generate_context_class_dataset.py` emits per-compound `mean_same_class_reuse` / `mean_cross_class_reuse` and matching variances. Likelihood work is deferred; for now these stats document the injected asymmetry for power sweeps. |
| Screening | Extend steady-state model to compute class-wise occupancy and reuse predictions; use them when scoring candidate θ. |

## 3. Symmetry Break C — Founder Bias / Exchangeability Break
| Layer | Implementation Plan |
| --- | --- |
| Simulator | ✅ Optional `founder_bias_config` (rank threshold + founder bonus / late penalty) now lives in `SimulationConfig`, exposed through Python bindings and `generate_founder_bias_dataset.py --founder-rank-threshold 2 --founder-bonus-theta 0.4 --late-penalty-theta -0.2`. |
| Constraints | Introduce features like `founder_reuse`, `age_penalty`, `early_depth_bonus`. |
| Observations | ✅ Founder generator records `mean_founder_reuse`, `mean_derived_reuse`, and `mean_founder_rank` per compound plus global summaries so we can benchmark identifiability before wiring inference features. |
| Screening | Deterministic approximation must aggregate age bins or time slices so ΔLL reflects founder bias. |

## 4. Additional Asymmetries (Future Iterations)
- **Multiple θ regimes:** simulate regime labels per trajectory, record regime-conditioned stats (duration split, reuse split), extend likelihood accordingly.
- **Sequence / motif context:** track motif gains/losses or environmental conditions; add observation terms for motif-specific reuse or duration.

## 5. Screening Likelihood Roadmap
Current `SteadyStateAssemblyModel.approximate_log_likelihood` now accepts enriched stats but still cannot “see” simulator-only asymmetries (e.g., depth gate) because occupancy is computed from constraint weights alone. Updated steps:
1. **Short term:** expose `stochastic_delta_ll` in all reports (done) and allow sweeps to fall back to stochastic scoring when screening ΔLL stays flat.
2. **Medium term:** introduce explicit depth-bin or context-specific features in the constraint layer so the deterministic model can approximate simulator effects.
3. **Long term:** implement moment-matched reuse/depth expectations per bin so screening ΔLL reagrees with stochastic ΔLL for symmetry breaks expressible via constraints.

## 6. Placeholder Helper Interface
- `AssemblyConstraint.get_rate_multipliers` returns a 2-state placeholder matrix. Either (a) implement the real per-transition matrix needed by shared helpers, or (b) refactor helpers to call `constraint_contribution` directly and remove the placeholder.

## 7. Deliverable Sequence
1. ✅ Finalize stochastic observation likelihood and validate ΔLL lift with depth-gated dataset (complete; ΔLL ≈ 222, recovery 100%).
2. Extend screening likelihood/constraint features so deterministic ΔLL can surface depth-gated signal **and** propagate `max_depth` defaults for legacy datasets (alternatively: formalize stochastic-only benchmarking mode while this is in progress).
3. 🏗️ Simulator + generator support for Break B (context classes) landed via `ContextClassConfig` + `generate_context_class_dataset.py`. Next: wire constraint/screening features once we want inference-aware class effects.
4. Implement simulator + constraint + observation work for Break C (founder bias), each accompanied by tests/datasets and power-envelope sweeps.
5. Clean up placeholder helper interface, ensuring any new asymmetry features have well-defined semantics end-to-end.

This plan keeps simulator physics, constraint definitions, observation conditioning, and inference logic aligned, ensuring every symmetry break is both introduced and observable.
