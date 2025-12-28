"""
MLE inference for assembly constraints.

Minimal viable implementation:
- Simulate dynamics to get latent states
- Evaluate observation likelihood
- Optimize with scipy.optimize.minimize

No gradients, no Hessians, no fancy optimizers.
This is Phase 1 - we find out what's identifiable.
"""

from typing import Dict, Set, Optional, Callable
import numpy as np

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from persiste.plugins.assembly.states.assembly_state import AssemblyState
from persiste.plugins.assembly.baselines.assembly_baseline import AssemblyBaseline
from persiste.plugins.assembly.constraints.assembly_constraint import AssemblyConstraint
from persiste.plugins.assembly.graphs.assembly_graph import AssemblyGraph
from persiste.plugins.assembly.dynamics.gillespie import GillespieSimulator
from persiste.plugins.assembly.observation.presence_model import PresenceObservationModel


class AssemblyMLEInference:
    """
    MLE inference for assembly constraints.
    
    Pipeline: θ → λ_eff → P(state) → P(observations)
    
    This is the scientific heart of the system.
    """
    
    def __init__(
        self,
        graph: AssemblyGraph,
        baseline: AssemblyBaseline,
        obs_model: PresenceObservationModel,
        initial_state: AssemblyState,
        feature_names: list[str],
    ):
        """
        Initialize MLE inference.
        
        Args:
            graph: Assembly graph
            baseline: Baseline rate model
            obs_model: Observation model
            initial_state: Starting state for dynamics
            feature_names: List of feature names to fit
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for inference. Install with: pip install scipy")
        
        self.graph = graph
        self.baseline = baseline
        self.obs_model = obs_model
        self.initial_state = initial_state
        self.feature_names = feature_names
    
    def neg_log_likelihood(
        self,
        theta_vec: np.ndarray,
        observed_compounds: Set[str],
        n_samples: int = 50,
        t_max: float = 50.0,
        burn_in: float = 25.0,
        rng_seed: Optional[int] = None,
    ) -> float:
        """
        Negative log-likelihood for optimization.
        
        Pipeline:
        1. Unpack θ from vector
        2. Create constraint with θ
        3. Simulate dynamics to get latent states
        4. Evaluate observation likelihood
        5. Return -log P(observations | θ)
        
        Args:
            theta_vec: Parameter vector
            observed_compounds: Set of observed compounds
            n_samples: Number of simulation samples
            t_max: Simulation time
            burn_in: Burn-in time
            rng_seed: Random seed for reproducibility
            
        Returns:
            Negative log-likelihood
        """
        # Create constraint from parameters
        theta_dict = {name: float(theta_vec[i]) for i, name in enumerate(self.feature_names)}
        constraint = AssemblyConstraint(feature_weights=theta_dict)
        
        # Simulate dynamics to get latent states
        rng = np.random.default_rng(rng_seed)
        simulator = GillespieSimulator(self.graph, self.baseline, constraint, rng=rng)
        
        latent_states = simulator.sample_final_states(
            self.initial_state,
            n_samples=n_samples,
            t_max=t_max,
            burn_in=burn_in,
        )
        
        # Evaluate observation likelihood
        log_lik = self.obs_model.compute_log_likelihood(observed_compounds, latent_states)
        
        return -log_lik
    
    def fit(
        self,
        observed_compounds: Set[str],
        n_samples: int = 50,
        t_max: float = 50.0,
        burn_in: float = 25.0,
        bounds: Optional[list] = None,
        rng_seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Fit constraint parameters via MLE.
        
        Args:
            observed_compounds: Set of observed compounds
            n_samples: Number of simulation samples per likelihood eval
            t_max: Simulation time
            burn_in: Burn-in time
            bounds: Parameter bounds (default: no bounds)
            rng_seed: Random seed
            verbose: Print optimization progress
            
        Returns:
            Dict with fitted parameters and diagnostics
        """
        # Initial parameters (neutral)
        theta0 = np.zeros(len(self.feature_names))
        
        # Bounds (default: no bounds)
        if bounds is None:
            bounds = [(None, None)] * len(self.feature_names)
        
        if verbose:
            print(f"Starting MLE optimization...")
            print(f"  Features: {self.feature_names}")
            print(f"  Initial θ: {theta0}")
            print(f"  Samples per eval: {n_samples}")
        
        # Objective function
        def objective(theta):
            nll = self.neg_log_likelihood(
                theta,
                observed_compounds,
                n_samples=n_samples,
                t_max=t_max,
                burn_in=burn_in,
                rng_seed=rng_seed,
            )
            if verbose:
                theta_str = ', '.join(f"{self.feature_names[i]}={theta[i]:.3f}" for i in range(len(theta)))
                print(f"  Eval: θ=[{theta_str}] → -log L = {nll:.4f}")
            return nll
        
        # Optimize
        result = minimize(
            objective,
            theta0,
            method='Nelder-Mead',  # Derivative-free
            bounds=bounds,
            options={'maxiter': 100, 'disp': verbose},
        )
        
        # Extract results
        theta_mle = result.x
        theta_dict = {name: float(theta_mle[i]) for i, name in enumerate(self.feature_names)}
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"  Success: {result.success}")
            print(f"  Iterations: {result.nit}")
            print(f"  Final -log L: {result.fun:.4f}")
            print(f"  MLE: {theta_dict}")
        
        return {
            'theta_mle': theta_dict,
            'theta_vec': theta_mle,
            'neg_log_lik': result.fun,
            'success': result.success,
            'n_iterations': result.nit,
            'message': result.message,
        }
