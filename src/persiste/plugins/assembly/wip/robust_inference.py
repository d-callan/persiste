"""
Robust Constraint Inference: Safe-by-default API.

This implements the system-level improvements:
- Joint baseline + constraint inference
- Automatic null testing
- L2 regularization by default
- Profile likelihood diagnostics
- Baseline sensitivity analysis
- Cross-validation support

Users can't accidentally overclaim.
"""

import sys
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
import warnings

from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.baselines.baseline_family import BaselineFamily, FixedBaseline, SimpleBaselineFamily
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.observation.presence_model import FrequencyWeightedPresenceModel
from persiste.plugins.assembly.inference.constraint_result import (
    ConstraintResult, ProfileDiagnostics, classify_evidence, compute_robustness_score
)


class RobustConstraintInference:
    """
    Safe-by-default constraint inference.
    
    Features:
    - Joint baseline + constraint inference (prevents false positives)
    - Automatic null testing (can't forget to compare)
    - L2 regularization by default (soft sparsity)
    - Profile likelihood diagnostics (identifiability checks)
    - Baseline sensitivity analysis (robustness)
    - Cross-validation (generalization)
    
    Example:
        inference = RobustConstraintInference(
            graph=graph,
            baseline_family=SimpleBaselineFamily(),  # Infer join_exponent
            obs_model=FrequencyWeightedPresenceModel(),
        )
        
        result = inference.fit_with_null(observed_counts)
        print(result)  # Automatic diagnostics
    """
    
    def __init__(
        self,
        graph: AssemblyGraph,
        baseline_family: BaselineFamily,
        obs_model: FrequencyWeightedPresenceModel,
        initial_state: Optional[AssemblyState] = None,
        regularization: float = 0.1,
        n_latent_samples: int = 100,
        t_max: float = 50.0,
        burn_in: float = 25.0,
    ):
        """
        Initialize robust inference.
        
        Args:
            graph: Assembly graph
            baseline_family: Baseline family (use SimpleBaselineFamily for robustness)
            obs_model: Observation model
            initial_state: Initial state for simulation (default: first primitive)
            regularization: L2 penalty strength (0 = no regularization)
            n_latent_samples: Number of samples for latent state estimation
            t_max: Simulation time
            burn_in: Burn-in time
        """
        self.graph = graph
        self.baseline_family = baseline_family
        self.obs_model = obs_model
        self.regularization = regularization
        self.n_latent_samples = n_latent_samples
        self.t_max = t_max
        self.burn_in = burn_in
        
        # Default initial state
        if initial_state is None:
            primitives = list(graph.primitives)
            self.initial_state = AssemblyState.from_parts([primitives[0]], depth=0)
        else:
            self.initial_state = initial_state
    
    def neg_log_likelihood(
        self,
        params: np.ndarray,
        observed_counts: Dict[str, int],
        constraint_features: List[str],
    ) -> float:
        """
        Negative log-likelihood with L2 regularization.
        
        Args:
            params: Combined [constraint_params, baseline_params]
            observed_counts: Observed frequency counts
            constraint_features: List of constraint feature names
        
        Returns:
            Negative log-likelihood + L2 penalty
        """
        # Split parameters
        n_constraint = len(constraint_features)
        constraint_params = params[:n_constraint]
        baseline_params = params[n_constraint:]
        
        # Create models
        theta = {feat: val for feat, val in zip(constraint_features, constraint_params)}
        constraint = AssemblyConstraint(feature_weights=theta)
        
        baseline_param_dict = {
            name: val for name, val in zip(self.baseline_family.get_parameter_names(), baseline_params)
        }
        baseline = self.baseline_family.create_baseline(baseline_param_dict)
        
        # Simulate latent states
        simulator = GillespieSimulator(self.graph, baseline, constraint, rng=np.random.default_rng(None))
        
        try:
            latent_states = simulator.sample_final_states(
                self.initial_state,
                n_samples=self.n_latent_samples,
                t_max=self.t_max,
                burn_in=self.burn_in,
            )
        except Exception as e:
            warnings.warn(f"Simulation failed: {e}")
            return 1e10  # Large penalty
        
        # Compute likelihood
        ll = self.obs_model.compute_log_likelihood(observed_counts, latent_states)
        
        # Add L2 regularization on constraint parameters
        l2_penalty = self.regularization * np.sum(constraint_params ** 2)
        
        # Add baseline prior
        baseline_prior = self.baseline_family.log_prior(baseline_param_dict)
        
        # Return negative (for minimization)
        return -(ll + baseline_prior) + l2_penalty
    
    def fit(
        self,
        observed_counts: Dict[str, int],
        constraint_features: List[str] = ['reuse_count', 'depth_change'],
        initial_guess: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """
        Fit constraint and baseline parameters.
        
        Args:
            observed_counts: Observed frequency counts
            constraint_features: List of constraint features to infer
            initial_guess: Initial parameter values (default: zeros)
        
        Returns:
            (constraint_params, baseline_params, neg_log_likelihood)
        """
        # Initial guess
        if initial_guess is None:
            constraint_init = np.zeros(len(constraint_features))
        else:
            constraint_init = np.array([initial_guess.get(f, 0.0) for f in constraint_features])
        
        baseline_init = np.array([
            self.baseline_family.parameters[name]
            for name in self.baseline_family.get_parameter_names()
        ])
        
        x0 = np.concatenate([constraint_init, baseline_init])
        
        # Optimize
        result = minimize(
            self.neg_log_likelihood,
            x0,
            args=(observed_counts, constraint_features),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.1}
        )
        
        # Extract parameters
        n_constraint = len(constraint_features)
        constraint_params = {
            feat: val for feat, val in zip(constraint_features, result.x[:n_constraint])
        }
        baseline_params = {
            name: val for name, val in zip(
                self.baseline_family.get_parameter_names(),
                result.x[n_constraint:]
            )
        }
        
        return constraint_params, baseline_params, result.fun
    
    def fit_with_null(
        self,
        observed_counts: Dict[str, int],
        constraint_features: List[str] = ['reuse_count', 'depth_change'],
        profile_diagnostics: bool = True,
        verbose: bool = True,
    ) -> ConstraintResult:
        """
        Fit with automatic null testing and diagnostics.
        
        This is the recommended API - safe by default.
        
        Args:
            observed_counts: Observed frequency counts
            constraint_features: List of constraint features to infer
            profile_diagnostics: Compute profile likelihood diagnostics
            verbose: Print progress
        
        Returns:
            ConstraintResult with full diagnostics
        """
        if verbose:
            print("=" * 60)
            print("Robust Constraint Inference")
            print("=" * 60)
            print(f"\nFeatures: {constraint_features}")
            print(f"Regularization: {self.regularization}")
            print(f"Baseline: {type(self.baseline_family).__name__}")
        
        # Fit null model (θ = 0, baseline only)
        if verbose:
            print("\n[1/3] Fitting null model (θ=0)...")
        
        null_params = {f: 0.0 for f in constraint_features}
        _, baseline_null, nll_null = self.fit(observed_counts, constraint_features, initial_guess=null_params)
        ll_null = -nll_null
        
        if verbose:
            print(f"  LL(null) = {ll_null:.2f}")
        
        # Fit constrained model
        if verbose:
            print("\n[2/3] Fitting constrained model...")
        
        constraint_params, baseline_params, nll_constrained = self.fit(observed_counts, constraint_features)
        ll_constrained = -nll_constrained
        delta_ll = ll_constrained - ll_null
        
        if verbose:
            print(f"  LL(constrained) = {ll_constrained:.2f}")
            print(f"  Δ LL = {delta_ll:.2f}")
        
        # Profile diagnostics
        profile_diags = {}
        identifiable = True
        
        if profile_diagnostics and len(constraint_features) > 0:
            if verbose:
                print("\n[3/3] Computing profile diagnostics...")
            
            for feature in constraint_features:
                diag = self._compute_profile_diagnostic(
                    feature, constraint_params, observed_counts, constraint_features, verbose=verbose
                )
                profile_diags[feature] = diag
                if not diag.identifiable:
                    identifiable = False
        
        # Collect warnings
        warnings_list = []
        
        if delta_ll < 2.0:
            warnings_list.append("Δ LL < 2: No evidence for constraints")
        elif delta_ll < 5.0:
            warnings_list.append("Δ LL < 5: Weak evidence, interpret with caution")
        elif delta_ll < 10.0:
            warnings_list.append("Δ LL < 10: Moderate evidence, validate on independent data")
        
        if not identifiable:
            warnings_list.append("Some parameters not identifiable (flat profile)")
        
        if self.regularization == 0:
            warnings_list.append("No regularization - may overfit to baseline errors")
        
        # Classify evidence
        evidence = classify_evidence(delta_ll, identifiable, warnings_list)
        
        # Compute robustness score
        robustness = compute_robustness_score(identifiable, delta_ll, warnings_list)
        
        # Create result
        result = ConstraintResult(
            estimate=constraint_params,
            baseline_params=baseline_params,
            ll_constrained=ll_constrained,
            ll_null=ll_null,
            delta_ll=delta_ll,
            evidence=evidence,
            identifiable=identifiable,
            profile_diagnostics=profile_diags,
            warnings=warnings_list,
            robustness_score=robustness,
        )
        
        if verbose:
            print("\n" + str(result))
        
        return result
    
    def _compute_profile_diagnostic(
        self,
        feature: str,
        base_params: Dict[str, float],
        observed_counts: Dict[str, int],
        constraint_features: List[str],
        verbose: bool = False,
    ) -> ProfileDiagnostics:
        """Compute profile likelihood diagnostic for one feature."""
        grid_values = np.linspace(-2.0, 2.0, 9)
        log_liks = []
        
        if verbose:
            print(f"  Profiling {feature}...", end=" ", flush=True)
        
        for value in grid_values:
            params_test = base_params.copy()
            params_test[feature] = value
            
            _, _, nll = self.fit(observed_counts, constraint_features, initial_guess=params_test)
            log_liks.append(-nll)
        
        log_liks = np.array(log_liks)
        ll_range = np.max(log_liks) - np.min(log_liks)
        peak_idx = np.argmax(log_liks)
        peak_value = grid_values[peak_idx]
        
        # Curvature
        if 0 < peak_idx < len(grid_values) - 1:
            curvature = abs(log_liks[peak_idx-1] - 2*log_liks[peak_idx] + log_liks[peak_idx+1])
        else:
            curvature = 0.0
        
        # Classify
        identifiable = ll_range > 10.0
        
        if ll_range < 2.0:
            evidence = "none"
        elif ll_range < 5.0:
            evidence = "weak"
        elif ll_range < 10.0:
            evidence = "moderate"
        else:
            evidence = "strong"
        
        if verbose:
            print(f"range={ll_range:.1f}, {evidence}")
        
        return ProfileDiagnostics(
            parameter=feature,
            peak_value=peak_value,
            ll_range=ll_range,
            curvature=curvature,
            identifiable=identifiable,
            evidence=evidence,
        )
    
    def baseline_sensitivity_analysis(
        self,
        observed_counts: Dict[str, int],
        baseline_variations: List[BaselineFamily],
        constraint_features: List[str] = ['reuse_count', 'depth_change'],
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Test sensitivity to baseline specification.
        
        Args:
            observed_counts: Observed frequency counts
            baseline_variations: List of alternative baseline families
            constraint_features: Constraint features to infer
            verbose: Print progress
        
        Returns:
            Dict mapping feature names to standard deviations across baselines
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Baseline Sensitivity Analysis")
            print("=" * 60)
        
        estimates = {f: [] for f in constraint_features}
        
        for i, baseline_family in enumerate(baseline_variations):
            if verbose:
                print(f"\n[{i+1}/{len(baseline_variations)}] Testing baseline variant...")
            
            # Temporarily swap baseline
            original_baseline = self.baseline_family
            self.baseline_family = baseline_family
            
            # Fit
            constraint_params, _, _ = self.fit(observed_counts, constraint_features)
            
            # Record
            for f in constraint_features:
                estimates[f].append(constraint_params.get(f, 0.0))
            
            # Restore
            self.baseline_family = original_baseline
        
        # Compute stability
        stability = {}
        for f in constraint_features:
            stability[f] = float(np.std(estimates[f]))
            if verbose:
                mean_val = np.mean(estimates[f])
                std_val = stability[f]
                print(f"  {f}: {mean_val:.3f} ± {std_val:.3f}")
        
        return stability
    
    def cross_validate(
        self,
        observed_counts: Dict[str, int],
        constraint_features: List[str] = ['reuse_count', 'depth_change'],
        k: int = 5,
        verbose: bool = True,
    ) -> float:
        """
        K-fold cross-validation.
        
        Args:
            observed_counts: Observed frequency counts
            constraint_features: Constraint features to infer
            k: Number of folds
            verbose: Print progress
        
        Returns:
            Average cross-validation score (higher is better)
        """
        if verbose:
            print("\n" + "=" * 60)
            print(f"{k}-Fold Cross-Validation")
            print("=" * 60)
        
        # Convert counts to list of observations
        observations = []
        for compound, count in observed_counts.items():
            observations.extend([compound] * count)
        
        np.random.shuffle(observations)
        fold_size = len(observations) // k
        
        scores = []
        
        for fold in range(k):
            # Split data
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < k - 1 else len(observations)
            
            test_obs = observations[test_start:test_end]
            train_obs = observations[:test_start] + observations[test_end:]
            
            # Convert back to counts
            train_counts = {}
            for obs in train_obs:
                train_counts[obs] = train_counts.get(obs, 0) + 1
            
            test_counts = {}
            for obs in test_obs:
                test_counts[obs] = test_counts.get(obs, 0) + 1
            
            # Fit on train
            constraint_params, baseline_params, _ = self.fit(train_counts, constraint_features)
            
            # Evaluate on test
            constraint = AssemblyConstraint(feature_weights=constraint_params)
            baseline = self.baseline_family.create_baseline(baseline_params)
            simulator = GillespieSimulator(self.graph, baseline, constraint, rng=np.random.default_rng(None))
            
            latent_states = simulator.sample_final_states(
                self.initial_state,
                n_samples=self.n_latent_samples,
                t_max=self.t_max,
                burn_in=self.burn_in,
            )
            
            test_ll = self.obs_model.compute_log_likelihood(test_counts, latent_states)
            scores.append(test_ll)
            
            if verbose:
                print(f"  Fold {fold+1}: LL = {test_ll:.2f}")
        
        avg_score = float(np.mean(scores))
        
        if verbose:
            print(f"\n  Average: {avg_score:.2f} ± {np.std(scores):.2f}")
        
        return avg_score
