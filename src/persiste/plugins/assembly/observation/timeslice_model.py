"""
Time-sliced presence observation model (Option B - VERY POWERFUL).

Instead of one snapshot:
    obs = {A, B, C}

Use time-sliced observations:
    obs = {
        t1: {A, B},
        t2: {A, B, C},
        t3: {A, D}
    }

You don't need exact times - just ordering.

This adds:
- Directional information
- Transient vs stable discrimination
- Constraint effect on dynamics

This is huge for assembly theory.
"""

from typing import Dict, Set, List, Any, Optional
import numpy as np

from persiste.core.observation_models import ObservationModel
from persiste.core.data import ObservedTransitions
from persiste.plugins.assembly.states.assembly_state import AssemblyState


class TimeSlicedPresenceModel(ObservationModel):
    """
    Time-sliced presence observation model.
    
    Observes presence/absence at multiple time points.
    Captures dynamics, not just equilibrium.
    
    Key advantages:
    - Directional information (what appears/disappears)
    - Transient vs stable discrimination
    - Constraint effects on dynamics (not just rates)
    
    Attributes:
        detection_prob: P(detect | present) at each time slice
        false_positive_prob: P(detect | absent) at each time slice
    """
    
    def __init__(
        self,
        graph: Any = None,
        detection_prob: float = 0.9,
        false_positive_prob: float = 0.01,
    ):
        """
        Initialize time-sliced presence model.
        
        Args:
            graph: TransitionGraph (unused, for interface compatibility)
            detection_prob: P(detect | present) per time slice
            false_positive_prob: P(detect | absent) per time slice
        """
        self.graph = graph
        self.detection_prob = detection_prob
        self.false_positive_prob = false_positive_prob
    
    def rate(self, i: int, j: int) -> float:
        """Get observation rate (unused)."""
        return 0.0
    
    def log_likelihood(
        self,
        data: ObservedTransitions,
        baseline: Any,
        graph: Any,
        parameters: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute log-likelihood of time-sliced observations.
        
        Args:
            data: ObservedTransitions with time_slices and trajectories
            baseline: Baseline model (unused)
            graph: TransitionGraph (unused)
            parameters: Optional parameters (unused)
            
        Returns:
            Log-likelihood
        """
        if not hasattr(data, 'time_slices') or not hasattr(data, 'trajectories'):
            raise ValueError(
                "TimeSlicedPresenceModel requires data with "
                "'time_slices' and 'trajectories' attributes"
            )
        
        time_slices = data.time_slices
        trajectories = data.trajectories
        
        return self.compute_log_likelihood(time_slices, trajectories)
    
    def compute_log_likelihood(
        self,
        time_slices: Dict[float, Set[str]],
        trajectories: List[Any],  # List of Trajectory objects
    ) -> float:
        """
        Compute likelihood of time-sliced observations.
        
        Model:
        1. For each time slice t_i, compute latent state distribution from trajectories
        2. For each compound, compute P(present | latent states at t_i)
        3. Likelihood = product over time slices of P(observations | latent states)
        
        Args:
            time_slices: Dict of time -> observed compounds at that time
            trajectories: List of simulated trajectories
            
        Returns:
            Log-likelihood
        """
        if not trajectories:
            return -np.inf
        
        log_lik = 0.0
        
        # For each time slice
        for t, observed_compounds in time_slices.items():
            # Get latent state distribution at time t from trajectories
            latent_states = self._get_state_distribution_at_time(trajectories, t)
            
            if not latent_states:
                return -np.inf
            
            # Compute marginal presence probability for each compound
            compound_probs = {}
            all_compounds = set()
            
            for state, prob in latent_states.items():
                for part in state.get_parts_list():
                    all_compounds.add(part)
                    compound_probs[part] = compound_probs.get(part, 0.0) + prob
            
            # Likelihood for this time slice
            for compound in all_compounds:
                p_present = compound_probs.get(compound, 0.0)
                
                if compound in observed_compounds:
                    # Observed: P(detect | present) * P(present) + P(detect | absent) * P(absent)
                    p_detect = (
                        self.detection_prob * p_present +
                        self.false_positive_prob * (1 - p_present)
                    )
                else:
                    # Not observed: P(not detect | present) * P(present) + P(not detect | absent) * P(absent)
                    p_detect = (
                        (1 - self.detection_prob) * p_present +
                        (1 - self.false_positive_prob) * (1 - p_present)
                    )
                
                if p_detect > 0:
                    log_lik += np.log(p_detect)
                else:
                    return -np.inf
        
        return log_lik
    
    def _get_state_distribution_at_time(
        self,
        trajectories: List[Any],
        t: float,
    ) -> Dict[AssemblyState, float]:
        """
        Get empirical state distribution at time t from trajectories.
        
        Args:
            trajectories: List of Trajectory objects
            t: Time point
            
        Returns:
            Dict of state -> empirical probability
        """
        states_at_t = []
        
        for traj in trajectories:
            # Find state at time t
            state = None
            for i, time in enumerate(traj.times):
                if time >= t:
                    state = traj.states[i]
                    break
            
            if state is None and traj.times:
                # If t is beyond trajectory, use final state
                state = traj.states[-1]
            
            if state is not None:
                states_at_t.append(state)
        
        # Count states
        state_counts = {}
        for state in states_at_t:
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Convert to probabilities
        total = len(states_at_t)
        if total == 0:
            return {}
        
        state_probs = {
            state: count / total
            for state, count in state_counts.items()
        }
        
        return state_probs
    
    def __str__(self) -> str:
        return f"TimeSlicedPresenceModel(detect={self.detection_prob:.2f}, false_pos={self.false_positive_prob:.3f})"
