"""
PERSISTE: Persistence Evidence via Rate Signatures In State Transition Evolution

A generalized framework for detecting constraint signatures in life-like systems.
"""

__version__ = "0.1.0"

from persiste.core.states import StateSpace
from persiste.core.transitions import TransitionGraph
from persiste.core.observation_models import (
    ObservationModel,
    PoissonObservationModel,
    CTMCObservationModel,
    MultinomialObservationModel,
)
from persiste.core.baseline import Baseline
from persiste.core.data import ObservedTransitions
from persiste.core.constraints import ConstraintModel
from persiste.core.inference import (
    ConstraintInference,
    ConstraintResult,
    ConstraintTestResult,
)
from persiste.plugins.registry import plugins

__all__ = [
    "StateSpace",
    "TransitionGraph",
    "ObservationModel",
    "PoissonObservationModel",
    "CTMCObservationModel",
    "MultinomialObservationModel",
    "Baseline",
    "ObservedTransitions",
    "ConstraintModel",
    "ConstraintInference",
    "ConstraintResult",
    "ConstraintTestResult",
    "plugins",
    "__version__",
]
