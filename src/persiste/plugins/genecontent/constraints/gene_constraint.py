"""
Constraint model for gene content evolution.

Purpose: Capture selective structure beyond baseline variability.

Constraint form:
    λ*_ij = λ_baseline(g) × exp(C(g, lineage; θ))

Where:
- C is cheap to compute
- θ are interpretable parameters

Constraint types (v1):
1. Global retention bias: Some genes are selectively retained
2. Host/environment association: Retained only in specific hosts
3. Functional group coherence: Pathway-level retention
4. Genome reduction bias: Lineage-specific loss acceleration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np

from persiste.core.constraint_utils import MultiplicativeConstraint


@dataclass
class ConstraintEffect:
    """
    Constraint effect on a specific gene family.

    The constraint modifies baseline rates:
        effective_rate = baseline_rate × exp(effect)

    Attributes:
        gain_effect: Log-multiplier on gain rate (0 = no effect)
        loss_effect: Log-multiplier on loss rate (0 = no effect)
        family_id: Gene family identifier
    """

    gain_effect: float = 0.0
    loss_effect: float = 0.0
    family_id: str = ""

    @property
    def gain_multiplier(self) -> float:
        """Multiplicative effect on gain rate."""
        return np.exp(self.gain_effect)

    @property
    def loss_multiplier(self) -> float:
        """Multiplicative effect on loss rate."""
        return np.exp(self.loss_effect)

    def is_neutral(self, tol: float = 1e-6) -> bool:
        """Check if constraint has no effect."""
        return abs(self.gain_effect) < tol and abs(self.loss_effect) < tol


class GeneContentConstraint(MultiplicativeConstraint, ABC):
    """
    Abstract base class for gene content constraint models.

    Constraints modify baseline rates to capture selective effects.
    The constraint contribution C(g, context; θ) is added to log-rates.
    """

    @abstractmethod
    def get_effect(self, family_id: str, context: Optional[Dict] = None) -> ConstraintEffect:
        """
        Get constraint effect for a gene family.

        Args:
            family_id: Gene family identifier
            context: Optional context (e.g., host, lineage, branch)

        Returns:
            ConstraintEffect with gain and loss effects
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """Get all constraint parameters as a dict."""
        pass

    @abstractmethod
    def set_parameters(self, params: Dict[str, float]):
        """Set constraint parameters from a dict."""
        pass

    @abstractmethod
    def n_parameters(self) -> int:
        """Number of free parameters."""
        pass

    def get_rate_multipliers(
        self,
        theta: float | None = None,
        *,
        family_idx: int | None = None,
        family_id: str | None = None,
        lineage_id: str | None = None,
        context: Optional[Dict] = None,
    ) -> Dict[tuple[int, int], float]:
        """Map constraint effects onto 2-state gain/loss transitions."""
        if family_id is None:
            raise ValueError("family_id is required for gene content constraints")
        effect = self.get_effect(family_id, context=context)
        return {
            (0, 1): effect.gain_multiplier,
            (1, 0): effect.loss_multiplier,
        }

    def log_prior(self) -> float:
        """Log prior on constraint parameters (default: 0)."""
        return 0.0

    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names for optimization."""
        return list(self.get_parameters().keys())

    def get_parameter_bounds(self, name: str) -> tuple:
        """Get bounds for a parameter. Override in subclasses."""
        return (-5.0, 5.0)  # Default: reasonable range for log-effects

    def get_initial_value(self, name: str) -> float:
        """Get initial value for a parameter. Override in subclasses."""
        return 0.0  # Default: no effect

    @classmethod
    def null_model(cls) -> "NullConstraint":
        """Return a null constraint (no effect)."""
        return NullConstraint()


@dataclass
class NullConstraint(GeneContentConstraint):
    """
    Null constraint: no selective effect.

    All rates are unchanged from baseline.
    Use as null hypothesis for LRT.
    """

    def get_effect(self, family_id: str, context: Optional[Dict] = None) -> ConstraintEffect:
        """No effect on any family."""
        return ConstraintEffect(family_id=family_id)

    def get_parameters(self) -> Dict[str, float]:
        """No parameters."""
        return {}

    def set_parameters(self, params: Dict[str, float]):
        """No parameters to set."""
        pass

    def n_parameters(self) -> int:
        """Zero parameters."""
        return 0


@dataclass
class PerFamilyConstraint(GeneContentConstraint):
    """
    Per-family constraint: each gene family has its own constraint effect.

    This is the most flexible model but has many parameters.
    Use with regularization to avoid overfitting.

    Attributes:
        effects: Dict mapping family_id -> (gain_effect, loss_effect)
        regularization: L2 regularization strength (0 = none)
    """

    effects: Dict[str, tuple] = field(default_factory=dict)
    regularization: float = 0.1

    def get_effect(self, family_id: str, context: Optional[Dict] = None) -> ConstraintEffect:
        """Get effect for a family."""
        if family_id in self.effects:
            gain_eff, loss_eff = self.effects[family_id]
        else:
            gain_eff, loss_eff = 0.0, 0.0

        return ConstraintEffect(gain_effect=gain_eff, loss_effect=loss_eff, family_id=family_id)

    def get_parameters(self) -> Dict[str, float]:
        """Get all parameters as flat dict."""
        params = {}
        for fam, (gain_eff, loss_eff) in self.effects.items():
            params[f"{fam}_gain"] = gain_eff
            params[f"{fam}_loss"] = loss_eff
        return params

    def set_parameters(self, params: Dict[str, float]):
        """Set parameters from flat dict."""
        # Group by family
        families = set()
        for key in params:
            if key.endswith("_gain") or key.endswith("_loss"):
                fam = key.rsplit("_", 1)[0]
                families.add(fam)

        for fam in families:
            gain_eff = params.get(f"{fam}_gain", 0.0)
            loss_eff = params.get(f"{fam}_loss", 0.0)
            self.effects[fam] = (gain_eff, loss_eff)

    def n_parameters(self) -> int:
        """2 parameters per family."""
        return 2 * len(self.effects)

    def log_prior(self) -> float:
        """L2 regularization prior."""
        if self.regularization <= 0:
            return 0.0

        penalty = 0.0
        for gain_eff, loss_eff in self.effects.values():
            penalty += gain_eff**2 + loss_eff**2

        return -0.5 * self.regularization * penalty


@dataclass
class RetentionBiasConstraint(GeneContentConstraint):
    """
    Global retention bias: some genes are selectively retained.

    Models the idea that certain gene families have reduced loss rates
    due to selective pressure (e.g., essential genes, core genome).

    Attributes:
        retained_families: Set of family IDs under retention constraint
        retention_strength: Log-reduction in loss rate (negative = reduced loss)
        prior_mean: Mean of Gaussian prior on retention_strength (default: 0)
        prior_std: Standard deviation of Gaussian prior (default: 2.0)
    """

    retained_families: Set[str] = field(default_factory=set)
    retention_strength: float = -1.0  # Default: ~2.7x reduction in loss rate
    prior_mean: float = 0.0  # Null hypothesis: no effect
    prior_std: float = 2.0  # Weak prior: allows effects up to ~6x rate change

    def get_effect(self, family_id: str, context: Optional[Dict] = None) -> ConstraintEffect:
        """Reduced loss rate for retained families."""
        if family_id in self.retained_families:
            return ConstraintEffect(
                gain_effect=0.0, loss_effect=self.retention_strength, family_id=family_id
            )
        return ConstraintEffect(family_id=family_id)

    def get_parameters(self) -> Dict[str, float]:
        """Single parameter: retention strength."""
        return {"retention_strength": self.retention_strength}

    def set_parameters(self, params: Dict[str, float]):
        """Set retention strength."""
        if "retention_strength" in params:
            self.retention_strength = params["retention_strength"]

    def n_parameters(self) -> int:
        """One parameter."""
        return 1

    def log_prior(self) -> float:
        """
        Gaussian prior on retention strength.

        Encodes the null hypothesis (θ = 0) while allowing deviations.
        This is Fix #3: hierarchical shrinkage to reduce variance and bias.

        Returns:
            Log-probability under N(prior_mean, prior_std²)
        """
        if self.prior_std <= 0:
            return 0.0  # Flat prior if std is non-positive

        # Gaussian log-probability: -0.5 * ((x - μ) / σ)² - log(σ√(2π))
        z = (self.retention_strength - self.prior_mean) / self.prior_std
        log_prob = -0.5 * z**2 - np.log(self.prior_std * np.sqrt(2 * np.pi))

        return log_prob


@dataclass
class HostAssociationConstraint(GeneContentConstraint):
    """
    Host/environment association: genes retained in specific hosts.

    Models the idea that certain genes are only beneficial in specific
    host or environmental contexts.

    Attributes:
        associations: Dict mapping family_id -> set of hosts where retained
        host_effects: Dict mapping host -> effect strength
        default_loss_increase: Increased loss rate when not in associated host
    """

    associations: Dict[str, Set[str]] = field(default_factory=dict)
    host_effects: Dict[str, float] = field(default_factory=dict)
    default_loss_increase: float = 1.0  # ~2.7x increased loss when not associated

    def get_effect(self, family_id: str, context: Optional[Dict] = None) -> ConstraintEffect:
        """
        Get effect based on host context.

        Args:
            family_id: Gene family
            context: Must contain 'host' key
        """
        if context is None or "host" not in context:
            return ConstraintEffect(family_id=family_id)

        host = context["host"]

        if family_id not in self.associations:
            return ConstraintEffect(family_id=family_id)

        associated_hosts = self.associations[family_id]

        if host in associated_hosts:
            # Gene is in its preferred host - reduced loss
            effect = self.host_effects.get(host, -1.0)
            return ConstraintEffect(gain_effect=0.0, loss_effect=effect, family_id=family_id)
        else:
            # Gene is NOT in preferred host - increased loss
            return ConstraintEffect(
                gain_effect=0.0, loss_effect=self.default_loss_increase, family_id=family_id
            )

    def get_parameters(self) -> Dict[str, float]:
        """Get host effect parameters."""
        params = {"default_loss_increase": self.default_loss_increase}
        for host, effect in self.host_effects.items():
            params[f"host_{host}"] = effect
        return params

    def set_parameters(self, params: Dict[str, float]):
        """Set host effect parameters."""
        if "default_loss_increase" in params:
            self.default_loss_increase = params["default_loss_increase"]

        for key, val in params.items():
            if key.startswith("host_"):
                host = key[5:]
                self.host_effects[host] = val

    def n_parameters(self) -> int:
        """1 + number of hosts."""
        return 1 + len(self.host_effects)


@dataclass
class GenomeReductionConstraint(GeneContentConstraint):
    """
    Genome reduction bias: lineage-specific loss acceleration.

    Models the idea that certain lineages are undergoing genome reduction
    (e.g., endosymbionts, obligate parasites).

    Attributes:
        reduction_lineages: Set of lineage labels under reduction
        reduction_strength: Log-increase in loss rate (positive = more loss)
    """

    reduction_lineages: Set[str] = field(default_factory=set)
    reduction_strength: float = 1.0  # ~2.7x increased loss rate

    def get_effect(self, family_id: str, context: Optional[Dict] = None) -> ConstraintEffect:
        """Increased loss rate in reduction lineages."""
        if context is None or "lineage" not in context:
            return ConstraintEffect(family_id=family_id)

        lineage = context["lineage"]

        if lineage in self.reduction_lineages:
            return ConstraintEffect(
                gain_effect=0.0, loss_effect=self.reduction_strength, family_id=family_id
            )
        return ConstraintEffect(family_id=family_id)

    def get_parameters(self) -> Dict[str, float]:
        """Single parameter: reduction strength."""
        return {"reduction_strength": self.reduction_strength}

    def set_parameters(self, params: Dict[str, float]):
        """Set reduction strength."""
        if "reduction_strength" in params:
            self.reduction_strength = params["reduction_strength"]

    def n_parameters(self) -> int:
        """One parameter."""
        return 1
