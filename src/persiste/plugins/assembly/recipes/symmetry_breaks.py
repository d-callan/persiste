import math
import logging
from dataclasses import dataclass
from typing import Literal, Any

import numpy as np
import persiste_rust

from persiste.plugins.assembly.recipes.base import DiagnosticReport
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.likelihood import compute_observation_ll

logger = logging.getLogger(__name__)

@dataclass
class SymmetryDiscriminationReport(DiagnosticReport):
    """Report for Symmetry Break Discrimination (Recipe 3)."""
    base_ll: float
    break_ll: float
    delta_ll: float
    p_value: float
    best_model: str
    severity: Literal["ok", "warning", "fail"]
    recommendation: str

    def print_summary(self) -> None:
        print(f"--- Symmetry Break Discrimination Report ({self.severity.upper()}) ---")
        print(f"Base LL:  {self.base_ll:.4f}")
        print(f"Break LL: {self.break_ll:.4f}")
        print(f"Delta LL: {self.delta_ll:.4f}")
        print(f"P-value:  {self.p_value:.4f}")
        print(f"Best Model: {self.best_model}")
        print(f"Recommendation: {self.recommendation}")

@dataclass
class ContextClassSensitivityReport(DiagnosticReport):
    """Report for Context Class Sensitivity (Recipe 4)."""
    original_ll: float
    shuffled_lls: list[float]
    p_value: float
    severity: Literal["ok", "warning", "fail"]
    recommendation: str

    def print_summary(self) -> None:
        print(f"--- Context Class Sensitivity Report ({self.severity.upper()}) ---")
        print(f"Original LL: {self.original_ll:.4f}")
        print(f"Mean Shuffled LL: {np.mean(self.shuffled_lls):.4f}")
        print(f"P-value: {self.p_value:.4f}")
        print(f"Recommendation: {self.recommendation}")

def symmetry_break_discrimination(
    data: dict[str, Any],
    base_theta: dict[str, float],
    break_theta: dict[str, float],
    break_type: Literal["A", "B", "C"],
    break_params: dict[str, Any],
) -> SymmetryDiscriminationReport:
    """
    Recipe 3: Symmetry Break Discrimination.
    
    Compares a stationary model (base_theta) against a non-stationary model (break_theta).
    
    Args:
        data: Observed data dict
        base_theta: Feature weights for the base (stationary) model
        break_theta: Feature weights for the symmetry break model
        break_type: Type of symmetry break ("A", "B", or "C")
        break_params: Parameters for the symmetry break (e.g., depth_gate_threshold)
    """
    primitives = data["primitives"]
    cached_model = data["cached_model"]
    observed_ids = data["stable_observed_ids"]
    observation_records = data.get("observation_records")
    compound_to_state = data.get("compound_to_state", {})

    # 1. Evaluate base model (Stationary)
    base_constraint = AssemblyConstraint(feature_weights=base_theta)
    base_latent = cached_model.get_latent_states(base_constraint)
    base_ll = compute_observation_ll(
        base_latent,
        observed_ids,
        primitives,
        observation_records=observation_records,
        compound_to_state=compound_to_state
    )

    # 2. Evaluate symmetry break model (Non-stationary)
    break_constraint_params = {"feature_weights": break_theta}
    if break_type == "A":
        break_constraint_params["depth_gate_threshold"] = break_params.get("depth_gate_threshold")
    elif break_type == "B":
        break_constraint_params["primitive_classes"] = break_params.get("primitive_classes")
    elif break_type == "C":
        break_constraint_params["founder_rank_threshold"] = break_params.get("founder_rank_threshold")
        
    break_constraint = AssemblyConstraint(**break_constraint_params)
    break_latent = cached_model.get_latent_states(break_constraint)
    break_ll = compute_observation_ll(
        break_latent,
        observed_ids,
        primitives,
        observation_records=observation_records,
        compound_to_state=compound_to_state
    )

    delta_ll = break_ll - base_ll
    
    # Debug print for discrimination
    logger.debug(f"Discrimination base_ll={base_ll:.4f}, break_ll={break_ll:.4f}, delta_ll={delta_ll:.4f}")
    
    # Simple Likelihood Ratio Test (approximated)
    # df is roughly the number of extra parameters
    df = len(break_theta) - len(base_theta)
    if df <= 0: df = 1 # Fallback
    
    chi2_stat = 2 * max(0, delta_ll)
    # P-value calculation (placeholder for actual chi2 survival function)
    # In a real implementation we would use scipy.stats.chi2.sf
    p_value = np.exp(-chi2_stat / 2.0) # Very rough approximation

    if delta_ll > 2.0:
        best_model = "Symmetry Break"
        severity = "ok"
        recommendation = f"Use the Symmetry Break {break_type} model; it provides significantly better fit."
    else:
        best_model = "Base (Stationary)"
        severity = "warning"
        recommendation = "Stick to the stationary model; the symmetry break does not add significant value."

    return SymmetryDiscriminationReport(
        base_ll=base_ll,
        break_ll=break_ll,
        delta_ll=delta_ll,
        p_value=p_value,
        best_model=best_model,
        severity=severity,
        recommendation=recommendation
    )

def context_class_sensitivity(
    data: dict[str, Any],
    theta: dict[str, float],
    primitive_classes: dict[str, str],
    n_shuffles: int = 20,
) -> ContextClassSensitivityReport:
    """
    Recipe 4: Context Class Sensitivity.
    
    Validates if the defined primitive classes are actually distinct by comparing
    against models where the class labels are shuffled among primitives.
    
    Args:
        data: Observed data dict
        theta: Feature weights (must include same_class_reuse and/or cross_class_reuse)
        primitive_classes: Mapping from primitive to class label
        n_shuffles: Number of random shuffles to perform
    """
    primitives = data["primitives"]
    cached_model = data["cached_model"]
    observed_ids = data["stable_observed_ids"]
    observation_records = data.get("observation_records")
    compound_to_state = data.get("compound_to_state", {})

    def evaluate_with_classes(classes: dict[str, str]) -> float:
        constraint = AssemblyConstraint(
            feature_weights=theta,
            primitive_classes=classes
        )
        # FORCE resimulation to ensure classes are passed to Rust
        cached_model.invalidate_cache()
        latent = cached_model.get_latent_states(constraint)
        return compute_observation_ll(
            latent,
            observed_ids,
            primitives,
            observation_records=observation_records,
            compound_to_state=compound_to_state
        )

    # 1. Original LL
    original_ll = evaluate_with_classes(primitive_classes)

    # 2. Shuffled LLs
    shuffled_lls = []
    primitive_names = list(primitive_classes.keys())
    class_labels = list(primitive_classes.values())
    
    for _ in range(n_shuffles):
        np.random.shuffle(class_labels)
        shuffled_map = dict(zip(primitive_names, class_labels))
        shuffled_lls.append(evaluate_with_classes(shuffled_map))

    # P-value: fraction of shuffles that performed better than original
    p_value = float(np.sum(np.array(shuffled_lls) >= original_ll) / n_shuffles)

    if p_value < 0.15: # Relaxed p-value for small sample size
        severity = "ok"
        recommendation = "Context classes are well-defined and contribute real signal."
    elif p_value < 0.3:
        severity = "warning"
        recommendation = "Context class signal is weak; consider if classes are correctly assigned."
    else:
        severity = "fail"
        recommendation = "No evidence for context class signal. The classes may be arbitrary."

    return ContextClassSensitivityReport(
        original_ll=original_ll,
        shuffled_lls=shuffled_lls,
        p_value=p_value,
        severity=severity,
        recommendation=recommendation
    )

@dataclass
class FounderPersistenceReport(DiagnosticReport):
    """Report for Founder Persistence (Recipe 5)."""
    founder_ids: list[int]
    founder_discovery_times: dict[int, float]
    final_probabilities: dict[int, float]
    persistence_ratio: float
    severity: Literal["ok", "warning", "fail"]
    recommendation: str

    def print_summary(self) -> None:
        print(f"--- Founder Persistence Report ({self.severity.upper()}) ---")
        print(f"Number of Founders: {len(self.founder_ids)}")
        print(f"Founder Persistence Ratio: {self.persistence_ratio:.4f}")
        print(f"Recommendation: {self.recommendation}")

def founder_persistence_diagnostic(
    data: dict[str, Any],
    founder_rank_threshold: int = 5,
) -> FounderPersistenceReport:
    """
    Recipe 5: Founder Persistence Diagnostic.
    
    Analyzes whether early-discovered (founder) states remain significant in the
    stationary distribution of the assembly process.
    
    Args:
        data: Observed data dict (must have a cached model with trajectories)
        founder_rank_threshold: Rank below which states are considered founders
    """
    cached_model = data["cached_model"]
    
    if not cached_model._cache:
        raise ValueError("Cached model must be initialized with trajectories.")
        
    # 1. Identify founders across all trajectories
    # We use first_visit_time and founder_rank stored in PathStats (via Rust)
    # or we can reconstruct from discovered_states if they have rank metadata.
    # Actually, let's use the latent states (stationary distribution) vs discovery order.
    
    simulation_result = persiste_rust.simulate_assembly_trajectories(
        primitives=cached_model.primitives,
        initial_parts=cached_model.initial_state.get_parts_list(),
        theta=cached_model._cache.theta_ref,
        n_samples=cached_model.simulation.n_samples,
        t_max=cached_model.simulation.t_max,
        burn_in=0.0, # NO burn-in to catch early discovery
        max_depth=cached_model.simulation.max_depth,
        seed=cached_model.rng_seed,
        kappa=cached_model.baseline.kappa,
        join_exponent=cached_model.baseline.join_exponent,
        split_exponent=cached_model.baseline.split_exponent,
        decay_rate=cached_model.baseline.decay_rate,
        initial_state_id=cached_model.initial_state.stable_id,
    )
    
    paths = simulation_result["paths"]
    
    # Map state_id -> earliest discovery time across all paths
    discovery_times: dict[int, float] = {}
    for path in paths:
        sid = path["final_state_id"]
        t = path["first_visit_time"]
        if sid not in discovery_times or t < discovery_times[sid]:
            discovery_times[sid] = t
            
    # Sort states by discovery time
    sorted_by_discovery = sorted(discovery_times.items(), key=lambda x: x[1])
    founder_ids = [sid for sid, t in sorted_by_discovery[:founder_rank_threshold]]
    
    # 2. Check final probabilities from the standard model (with burn-in)
    final_probs = cached_model._get_current_states()
    
    # Calculate mass of founders in final distribution
    founder_mass = sum(final_probs.get(sid, 0.0) for sid in founder_ids)
    
    # Persistence ratio: founder mass relative to uniform expected mass
    n_total_states = len(final_probs)
    if n_total_states == 0: n_total_states = 1
    expected_uniform = len(founder_ids) / n_total_states
    persistence_ratio = founder_mass / expected_uniform if expected_uniform > 0 else 0.0

    if persistence_ratio > 2.0:
        severity = "ok"
        recommendation = "Founders show strong persistence; the system has 'memory' of early states."
    elif persistence_ratio > 0.5:
        severity = "warning"
        recommendation = "Founders show moderate persistence."
    else:
        severity = "fail"
        recommendation = "Founders are quickly lost; early discovery does not impact stationary distribution."

    return FounderPersistenceReport(
        founder_ids=founder_ids,
        founder_discovery_times={sid: t for sid, t in sorted_by_discovery[:founder_rank_threshold]},
        final_probabilities={sid: final_probs.get(sid, 0.0) for sid in founder_ids},
        persistence_ratio=persistence_ratio,
        severity=severity,
        recommendation=recommendation
    )
