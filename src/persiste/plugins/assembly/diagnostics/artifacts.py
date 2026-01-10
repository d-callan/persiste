"""
Diagnostic inputs and artifacts for assembly inference.

Separates inference artifacts (θ̂, LL, cache) from diagnostic artifacts
(plots, distributions, sensitivity results).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import persiste_rust


@dataclass
class InferenceArtifacts:
    """Artifacts from inference, input to diagnostics."""

    theta_hat: dict[str, float]
    """Fitted feature weights."""

    log_likelihood: float
    """Log-likelihood at θ̂."""

    cache_id: str
    """Reference to cached path stats."""

    baseline_config: dict
    """Baseline configuration used."""

    graph_config: dict
    """Graph configuration used."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "theta_hat": self.theta_hat,
            "log_likelihood": self.log_likelihood,
            "cache_id": self.cache_id,
            "baseline_config": self.baseline_config,
            "graph_config": self.graph_config,
        }

    def to_json(self, path: Path) -> None:
        """Serialize to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "InferenceArtifacts":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class CachedPathData:
    """Cached trajectories with sufficient stats for diagnostics."""

    feature_counts: list[dict[str, int]]
    """Feature counts from each trajectory."""

    final_state_ids: list[int]
    """Final state IDs from each trajectory."""

    theta_ref: dict[str, float]
    """Reference θ at which trajectories were simulated."""

    def __len__(self) -> int:
        return len(self.feature_counts)

    def reweight_to(self, theta: dict[str, float]) -> tuple[list[float], float]:
        """
        Reweight cached paths to new θ without resimulation.

        Returns:
            Tuple of (normalized_weights, effective_sample_size)
        """
        weights, ess = persiste_rust.compute_importance_weights(
            path_feature_counts=self.feature_counts,
            theta=theta,
            theta_ref=self.theta_ref,
        )
        return list(weights), ess

    def weighted_state_distribution(
        self, theta: dict[str, float]
    ) -> dict[int, float]:
        """Get state distribution at new θ using importance weights."""
        weights, _ = self.reweight_to(theta)

        state_probs: dict[int, float] = {}
        for state_id, weight in zip(self.final_state_ids, weights):
            state_probs[state_id] = state_probs.get(state_id, 0.0) + weight

        return state_probs

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "feature_counts": self.feature_counts,
            "final_state_ids": self.final_state_ids,
            "theta_ref": self.theta_ref,
        }

    def to_json(self, path: Path) -> None:
        """Serialize to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "CachedPathData":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class DiagnosticArtifacts:
    """Output from diagnostic suite, separate from inference artifacts."""

    null_distribution: dict | None = None
    """Results from null resampling."""

    profile_likelihoods: dict[str, dict] = field(default_factory=dict)
    """Profile likelihoods for each feature."""

    baseline_sensitivity: dict | None = None
    """Results from baseline sensitivity analysis."""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "null_distribution": self.null_distribution,
            "profile_likelihoods": self.profile_likelihoods,
            "baseline_sensitivity": self.baseline_sensitivity,
        }

    def to_json(self, path: Path) -> None:
        """Serialize for CI pipelines / Datamonkey-style UIs."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "DiagnosticArtifacts":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def summary(self) -> str:
        """Generate text summary of diagnostics."""
        lines = ["Diagnostic Summary", "=" * 40]

        if self.null_distribution:
            nd = self.null_distribution
            lines.append(f"Null resampling: p-value={nd.get('p_value', 'N/A'):.4f}")

        for feature, profile in self.profile_likelihoods.items():
            mle = profile.get("mle", "N/A")
            ci = profile.get("confidence_interval", ["N/A", "N/A"])
            lines.append(f"Profile({feature}): MLE={mle}, 95% CI=[{ci[0]}, {ci[1]}]")

        if self.baseline_sensitivity:
            bs = self.baseline_sensitivity
            lines.append(f"Baseline sensitivity: stable={bs.get('stable', 'N/A')}")

        return "\n".join(lines)
