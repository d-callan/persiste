"""
Robust Constraint Inference V2: Fast by default with conditional robustness.

Key architectural principles:
1. Simulation is DECOUPLED from likelihood evaluation (cached states)
2. Robustness checks are CONDITIONAL (only when ΔLL is uncertain)
3. Fast/standard/thorough modes with proper staging
4. Inner loop is deterministic and fast
5. Outer loop handles stochastic validation

Performance:
- Fast mode: <30s (cached states, no profile diagnostics)
- Standard mode: 1-2min (cached states, conditional diagnostics)
- Thorough mode: 5-10min (fresh simulation, full diagnostics)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
import warnings

from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.baselines.baseline_family import BaselineFamily
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.observation.presence_model import FrequencyWeightedPresenceModel
from persiste.plugins.assembly.inference.constraint_result import (
    ConstraintResult, ProfileDiagnostics, classify_evidence, compute_robustness_score
)
from persiste.plugins.assembly.inference.state_cache import StateDistributionCache


class RobustConstraintInferenceV2:
    """
    Fast-by-default robust constraint inference.
    
    Architecture:
    - STAGE 1: Simulate once at null (θ=0) → cache states
    - STAGE 2: Fast optimization using cached states
    - STAGE 3: Conditional robustness checks (only if ΔLL uncertain)
    
    Modes:
    - fast: Cached states, no diagnostics (~30s)
    - standard: Cached states, conditional diagnostics (~1-2min)
    - thorough: Fresh simulation, full diagnostics (~5-10min)
    """
    
    def __init__(
        self,
        graph: AssemblyGraph,
        baseline_family: BaselineFamily,
        obs_model: FrequencyWeightedPresenceModel,
        initial_state: Optional[AssemblyState] = None,
        regularization: float = 0.1,
    ):
        self.graph = graph
        self.baseline_family = baseline_family
        self.obs_model = obs_model
        self.regularization = regularization
        
        if initial_state is None:
            primitives = list(graph.primitives)
            self.initial_state = AssemblyState.from_parts([primitives[0]], depth=0)
        else:
            self.initial_state = initial_state
        
        # State cache (populated on first use)
        self.cache: Optional[StateDistributionCache] = None
        self.cached_states: Optional[List[AssemblyState]] = None
    
    def _populate_cache(
        self,
        n_samples: int,
        t_max: float,
        burn_in: float,
        constraint_features: List[str],
        verbose: bool = False,
    ):
        """Populate cache by simulating at null (θ=0)."""
        if verbose:
            print(f"  Simulating {n_samples} trajectories at null (θ=0)...")
        
        # Create null model
        theta_null = {f: 0.0 for f in constraint_features}
        constraint_null = AssemblyConstraint(feature_weights=theta_null)
        
        baseline_params = {
            name: self.baseline_family.parameters[name]
            for name in self.baseline_family.get_parameter_names()
        }
        baseline = self.baseline_family.create_baseline(baseline_params)
        
        # Create cache
        self.cache = StateDistributionCache(
            self.graph,
            self.initial_state,
            t_max=t_max,
            burn_in=burn_in,
        )
        
        # Populate
        self.cache.populate(
            baseline,
            constraint_null,
            n_samples=n_samples,
            constraint_features=constraint_features,
            baseline_param_names=list(baseline_params.keys()),
        )
        
        # Store states for fast access
        self.cached_states = self.cache.get_final_states()
        
        if verbose:
            print(f"  Cached {len(self.cached_states)} states")
    
    def neg_log_likelihood(
        self,
        params: np.ndarray,
        observed_counts: Dict[str, int],
        constraint_features: List[str],
        use_cache: bool = True,
    ) -> float:
        """
        Negative log-likelihood (FAST: uses cached states).
        
        Args:
            params: Combined [constraint_params, baseline_params]
            observed_counts: Observed data
            constraint_features: Feature names
            use_cache: Use cached states (fast) vs fresh simulation (slow)
        """
        # Split parameters
        n_constraint = len(constraint_features)
        constraint_params = params[:n_constraint]
        baseline_params = params[n_constraint:]
        
        # Create models
        theta = {feat: val for feat, val in zip(constraint_features, constraint_params)}
        
        if use_cache and self.cached_states is not None:
            # FAST PATH: use cached states
            # Note: This is approximate - assumes θ is close to null
            latent_states = self.cached_states
        else:
            # SLOW PATH: fresh simulation
            constraint = AssemblyConstraint(feature_weights=theta)
            baseline_param_dict = {
                name: val for name, val in zip(self.baseline_family.get_parameter_names(), baseline_params)
            }
            baseline = self.baseline_family.create_baseline(baseline_param_dict)
            
            simulator = GillespieSimulator(self.graph, baseline, constraint, rng=np.random.default_rng(None))
            
            try:
                latent_states = simulator.sample_final_states(
                    self.initial_state,
                    n_samples=len(self.cached_states) if self.cached_states else 100,
                    t_max=50.0,
                    burn_in=25.0,
                )
            except Exception as e:
                warnings.warn(f"Simulation failed: {e}")
                return 1e10
        
        # Compute likelihood
        # Convert list of states to dict with uniform probabilities
        if isinstance(latent_states, list):
            n_states = len(latent_states)
            latent_states_dict = {state: 1.0/n_states for state in latent_states}
        else:
            latent_states_dict = latent_states
        
        ll = self.obs_model.compute_log_likelihood(observed_counts, latent_states_dict)
        
        # Add L2 regularization
        l2_penalty = self.regularization * np.sum(constraint_params ** 2)
        
        # Add baseline prior
        baseline_param_dict = {
            name: val for name, val in zip(self.baseline_family.get_parameter_names(), baseline_params)
        }
        baseline_prior = self.baseline_family.log_prior(baseline_param_dict)
        
        return -(ll + baseline_prior) + l2_penalty
    
    def fit(
        self,
        observed_counts: Dict[str, int],
        constraint_features: List[str] = ['reuse_count', 'depth_change'],
        initial_guess: Optional[Dict[str, float]] = None,
        use_cache: bool = True,
        max_iter: int = 100,
    ) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """
        Fit constraint and baseline parameters.
        
        Args:
            observed_counts: Observed data
            constraint_features: Features to infer
            initial_guess: Initial parameter values
            use_cache: Use cached states (fast)
            max_iter: Max optimizer iterations
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
            args=(observed_counts, constraint_features, use_cache),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 0.01, 'fatol': 0.1}
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
        mode: str = 'fast',
        verbose: bool = True,
    ) -> ConstraintResult:
        """
        Fit with automatic null testing and conditional diagnostics.
        
        Modes:
        - 'fast': Cached states, no diagnostics (~30s)
        - 'standard': Cached states, conditional diagnostics (~1-2min)
        - 'thorough': Fresh simulation, full diagnostics (~5-10min)
        
        Args:
            observed_counts: Observed data
            constraint_features: Features to infer
            mode: Inference mode
            verbose: Print progress
        """
        if mode not in {'fast', 'standard', 'thorough'}:
            raise ValueError("mode must be 'fast', 'standard', or 'thorough'")
        
        # Mode-specific settings
        settings = {
            'fast': {
                'n_samples': 50,
                't_max': 50.0,
                'burn_in': 25.0,
                'use_cache': True,  # Use cache for screening only
                'max_iter': 100,
                'run_diagnostics': False,
            },
            'standard': {
                'n_samples': 100,
                't_max': 50.0,
                'burn_in': 25.0,
                'use_cache': False,  # FIXED: Use fresh simulation for accuracy
                'max_iter': 200,
                'run_diagnostics': 'conditional',  # Only if ΔLL in uncertain range
            },
            'thorough': {
                'n_samples': 200,
                't_max': 50.0,
                'burn_in': 25.0,
                'use_cache': False,
                'max_iter': 300,
                'run_diagnostics': True,
            },
        }
        
        config = settings[mode]
        
        if verbose:
            print("=" * 60)
            print("Robust Constraint Inference V2")
            print("=" * 60)
            print(f"\nMode: {mode}")
            print(f"Features: {constraint_features}")
            print(f"Regularization: {self.regularization}")
            print(f"Baseline: {type(self.baseline_family).__name__}")
        
        # STAGE 1: Populate cache (if using cache and not already populated)
        if config['use_cache'] and self.cached_states is None:
            if verbose:
                print("\n[Stage 1] Populating state cache...")
            self._populate_cache(
                n_samples=config['n_samples'],
                t_max=config['t_max'],
                burn_in=config['burn_in'],
                constraint_features=constraint_features,
                verbose=verbose,
            )
        
        # STAGE 2: Fit null and constrained models
        if verbose:
            print("\n[Stage 2] Fitting models...")
            print("  [2.1] Null model (θ=0)...")
        
        null_params = {f: 0.0 for f in constraint_features}
        _, baseline_null, nll_null = self.fit(
            observed_counts,
            constraint_features,
            initial_guess=null_params,
            use_cache=config['use_cache'],
            max_iter=config['max_iter'],
        )
        ll_null = -nll_null
        
        if verbose:
            print(f"    LL(null) = {ll_null:.2f}")
            print("  [2.2] Constrained model...")
        
        constraint_params, baseline_params, nll_constrained = self.fit(
            observed_counts,
            constraint_features,
            use_cache=config['use_cache'],
            max_iter=config['max_iter'],
        )
        ll_constrained = -nll_constrained
        delta_ll = ll_constrained - ll_null
        
        if verbose:
            print(f"    LL(constrained) = {ll_constrained:.2f}")
            print(f"    Δ LL = {delta_ll:.2f}")
        
        # STAGE 3: Conditional diagnostics
        profile_diags = {}
        identifiable = True
        
        run_diagnostics = config['run_diagnostics']
        if run_diagnostics == 'conditional':
            # Only run if ΔLL is in uncertain range [2, 15]
            run_diagnostics = 2.0 <= delta_ll <= 15.0
            if verbose and run_diagnostics:
                print("\n[Stage 3] Δ LL in uncertain range → running diagnostics...")
            elif verbose:
                print("\n[Stage 3] Δ LL outside uncertain range → skipping diagnostics")
        
        if run_diagnostics and len(constraint_features) > 0:
            if verbose:
                print("  Computing profile diagnostics...")
            
            # Simplified profile: just check if estimate is stable
            for feature in constraint_features:
                # Test ±1 around estimate
                test_values = [
                    constraint_params[feature] - 1.0,
                    constraint_params[feature],
                    constraint_params[feature] + 1.0,
                ]
                
                lls = []
                for val in test_values:
                    test_params = constraint_params.copy()
                    test_params[feature] = val
                    _, _, nll = self.fit(
                        observed_counts,
                        constraint_features,
                        initial_guess=test_params,
                        use_cache=config['use_cache'],
                        max_iter=50,  # Fewer iterations for profile
                    )
                    lls.append(-nll)
                
                ll_range = max(lls) - min(lls)
                peak_idx = np.argmax(lls)
                
                # Simple curvature
                if peak_idx == 1:
                    curvature = abs(lls[0] - 2*lls[1] + lls[2])
                else:
                    curvature = 0.0
                
                identifiable_feature = ll_range > 5.0
                identifiable = identifiable and identifiable_feature
                
                if ll_range < 2.0:
                    evidence = "none"
                elif ll_range < 5.0:
                    evidence = "weak"
                elif ll_range < 10.0:
                    evidence = "moderate"
                else:
                    evidence = "strong"
                
                profile_diags[feature] = ProfileDiagnostics(
                    parameter=feature,
                    peak_value=test_values[peak_idx],
                    ll_range=ll_range,
                    curvature=curvature,
                    identifiable=identifiable_feature,
                    evidence=evidence,
                )
                
                if verbose:
                    print(f"    {feature}: range={ll_range:.1f}, {evidence}")
        
        # Collect warnings
        warnings_list = []
        
        if delta_ll < 2.0:
            warnings_list.append("Δ LL < 2: No evidence for constraints")
        elif delta_ll < 5.0:
            warnings_list.append("Δ LL < 5: Weak evidence, interpret with caution")
        elif delta_ll < 10.0:
            warnings_list.append("Δ LL < 10: Moderate evidence, validate on independent data")
        
        if not identifiable and profile_diags:
            warnings_list.append("Some parameters not identifiable (flat profile)")
        
        if self.regularization == 0:
            warnings_list.append("No regularization - may overfit to baseline errors")
        
        if config['use_cache']:
            warnings_list.append("Used cached states (fast but approximate)")
        
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
