"""
Constraint model for global ω (dN/dS) inference in phylogenetics.

Provides a lightweight adapter that satisfies the ConstraintInference API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from persiste.core.baseline import Baseline


@dataclass
class OmegaConstraint:
    """
    Simple constraint model that scales nonsynonymous rates by ω.

    Attributes:
        baseline: MG94Baseline providing synonymous opportunities
        graph: CodonTransitionGraph describing allowed transitions
        omega: Current ω value (default 1.0)
    """

    baseline: Any
    graph: Any
    omega: float = 1.0
    allow_facilitation: bool = True

    def get_parameters(self) -> dict[str, float]:
        """Return current parameter dictionary."""
        return {"omega": self.omega}

    def set_parameters(self, params: dict[str, float]) -> None:
        """Update internal parameter dictionary."""
        self.omega = float(params.get("omega", self.omega))

    # ------------------------------------------------------------------
    # ConstraintInference compatibility methods
    # ------------------------------------------------------------------

    def pack(self, parameters: dict[str, float] | None = None) -> np.ndarray:
        """Pack ω into a 1D array."""
        params = parameters if parameters is not None else {"omega": self.omega}
        return np.array([float(params.get("omega", 1.0))], dtype=float)

    def unpack(self, vector: np.ndarray) -> dict[str, float]:
        """Unpack optimization vector into parameter dict."""
        if vector.size != 1:
            raise ValueError("OmegaConstraint expects a single-parameter vector.")
        return {"omega": float(vector[0])}

    def initial_parameters(self) -> np.ndarray:
        """Provide neutral starting value."""
        return self.pack()

    def num_free_parameters(self, parameters: dict[str, float] | None = None) -> int:
        """Number of free parameters (always 1)."""
        return 1

    def get_constrained_baseline(self, parameters: dict[str, float] | None = None) -> Baseline:
        """
        Return a Baseline whose rates incorporate ω scaling on nonsynonymous edges.
        """
        omega_value = float((parameters or {}).get("omega", self.omega))
        return _OmegaScaledBaseline(self.baseline, self.graph, omega_value)


class _OmegaScaledBaseline(Baseline):
    """
    Lightweight wrapper that preserves MG94 helpers while applying ω scaling.

    ConstraintInference expects a Baseline, but our observation model also
    needs MG94-specific helpers (rate matrices, matrix exponentials, codon space).
    This wrapper forwards all helper calls to the original MG94Baseline while
    overriding get_rate and exposing the current ω value.
    """

    def __init__(self, base: Any, graph: Any, omega: float):
        self._base = base
        self.graph = graph
        self.omega = omega
        super().__init__(rate_fn=self._rate_fn)

    def _rate_fn(self, i: int, j: int) -> float:
        rate = self._base.get_rate(i, j)
        if self.graph.is_nonsynonymous(i, j):
            return rate * self.omega
        return rate

    def build_rate_matrix(self, omega: float | None = None) -> np.ndarray:
        if not hasattr(self._base, "build_rate_matrix"):
            raise AttributeError("Underlying baseline lacks build_rate_matrix")
        omega_value = omega if omega is not None else self.omega
        return self._base.build_rate_matrix(omega_value)

    def build_rate_matrix_alpha_beta(self, alpha: float, beta: float) -> np.ndarray:
        if not hasattr(self._base, "build_rate_matrix_alpha_beta"):
            raise AttributeError(
                "Underlying baseline lacks build_rate_matrix_alpha_beta"
            )
        return self._base.build_rate_matrix_alpha_beta(alpha, beta)

    def matrix_exponential_fast(self, alpha: float, beta: float, t: float) -> np.ndarray:
        if not hasattr(self._base, "matrix_exponential_fast"):
            raise AttributeError("Underlying baseline lacks matrix_exponential_fast")
        return self._base.matrix_exponential_fast(alpha, beta, t)

    def __getattr__(self, item: str) -> Any:
        # Delegate attributes (codon_space, kappa, etc.) to the base MG94 object.
        return getattr(self._base, item)
