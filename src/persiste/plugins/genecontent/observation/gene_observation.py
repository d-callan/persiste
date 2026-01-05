from __future__ import annotations

"""
Observation model for gene content evolution.

Observed data: Gene presence/absence at tree tips.
    Y[taxon, gene] ∈ {0,1}

Assumptions (v1):
- Fully observed presence/absence
- No uncertainty in gene calls (yet)

Later extensions:
- Missingness
- Copy number
- Assembly bias / detection probability
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from persiste.core.observation_models import ObservationModel
from persiste.core.pruning import (
    ArrayTipConditionalProvider,
    FelsensteinPruning,
    SimpleBinaryTransitionProvider,
)

try:
    from persiste.core.pruning_rust import compute_likelihoods_batch, check_rust_available

    RUST_AVAILABLE = check_rust_available()
except ImportError:
    compute_likelihoods_batch = None
    RUST_AVAILABLE = False

from ..baselines.gene_baseline import GeneContentBaseline
from ..states.gene_state import GeneFamilyVector


@dataclass
class TipObservations:
    """
    Observed gene presence/absence at tree tips.

    This is the primary data structure for gene content analysis.

    Attributes:
        observations: Dict mapping taxon_id -> GeneFamilyVector
        family_ids: List of all gene family IDs (column order)
        taxon_ids: List of all taxon IDs (row order)
    """

    observations: dict[str, GeneFamilyVector] = field(default_factory=dict)
    family_ids: list[str] = field(default_factory=list)
    taxon_ids: list[str] = field(default_factory=list)

    @property
    def n_taxa(self) -> int:
        """Number of taxa."""
        return len(self.taxon_ids)

    @property
    def n_families(self) -> int:
        """Number of gene families."""
        return len(self.family_ids)

    def get_observation(self, taxon_id: str) -> GeneFamilyVector:
        """Get observation for a taxon."""
        return self.observations[taxon_id]

    def get_family_presence(self, family_id: str) -> Dict[str, bool]:
        """Get presence/absence of a family across all taxa."""
        return {taxon: obs.is_present(family_id) for taxon, obs in self.observations.items()}

    def get_family_count(self, family_id: str) -> int:
        """Count how many taxa have a family present."""
        return sum(1 for obs in self.observations.values() if obs.is_present(family_id))

    def to_matrix(self) -> np.ndarray:
        """
        Convert to numpy matrix.

        Returns:
            2D array of shape (n_taxa, n_families) with 0/1 values
        """
        matrix = np.zeros((self.n_taxa, self.n_families), dtype=np.int8)

        for i, taxon in enumerate(self.taxon_ids):
            obs = self.observations[taxon]
            for j, fam in enumerate(self.family_ids):
                if obs.is_present(fam):
                    matrix[i, j] = 1

        return matrix

    @classmethod
    def from_matrix(
        cls, matrix: np.ndarray, taxon_ids: list[str], family_ids: list[str]
    ) -> TipObservations:
        """
        Create from numpy matrix.

        Args:
            matrix: 2D array of shape (n_taxa, n_families) with 0/1 values
            taxon_ids: List of taxon IDs (row order)
            family_ids: List of family IDs (column order)

        Returns:
            TipObservations instance
        """
        observations = {}
        for i, taxon in enumerate(taxon_ids):
            presence = {fam: bool(matrix[i, j]) for j, fam in enumerate(family_ids)}
            observations[taxon] = GeneFamilyVector(presence=presence, taxon_id=taxon)

        return cls(
            observations=observations, family_ids=list(family_ids), taxon_ids=list(taxon_ids)
        )

    def summary(self) -> Dict:
        """
        Compute summary statistics.

        Returns:
            Dict with summary stats
        """
        matrix = self.to_matrix()

        # Per-family stats
        family_counts = matrix.sum(axis=0)

        # Per-taxon stats
        taxon_counts = matrix.sum(axis=1)

        # Core/accessory classification
        n_taxa = self.n_taxa
        core_threshold = 0.95 * n_taxa
        rare_threshold = 0.15 * n_taxa

        n_core = np.sum(family_counts >= core_threshold)
        n_rare = np.sum(family_counts <= rare_threshold)
        n_accessory = self.n_families - n_core - n_rare

        return {
            "n_taxa": self.n_taxa,
            "n_families": self.n_families,
            "n_core": int(n_core),
            "n_accessory": int(n_accessory),
            "n_rare": int(n_rare),
            "mean_genes_per_taxon": float(taxon_counts.mean()),
            "std_genes_per_taxon": float(taxon_counts.std()),
            "mean_taxa_per_gene": float(family_counts.mean()),
            "std_taxa_per_gene": float(family_counts.std()),
        }

    def __repr__(self) -> str:
        return f"TipObservations({self.n_taxa} taxa, {self.n_families} families)"


class GeneContentObservationModel(ObservationModel):
    """
    ObservationModel adapter that scores gene content presence/absence on a tree.
    
    This bridges the plugin's tree-based likelihood code with the core ConstraintInference
    pipeline by exposing the standard log_likelihood(data, baseline, graph) signature.
    """

    def __init__(
        self,
        graph: Optional[Any],
        tree: Any,
        tip_observations: TipObservations,
        use_rust: bool = True,
        use_jax: bool = False,
    ):
        self.graph = graph
        self.tree = tree
        self.tip_observations = tip_observations
        self.use_rust = use_rust and RUST_AVAILABLE and compute_likelihoods_batch is not None
        self.use_jax = use_jax
        self._pruning: Optional[FelsensteinPruning] = None
        self._presence_matrix: Optional[np.ndarray] = None

    def rate(self, i: int, j: int) -> float:
        """
        Gene content likelihoods are computed across tree branches rather than
        per-graph transitions, so this interface is unused.
        """
        return 0.0

    def _ensure_pruning(self) -> FelsensteinPruning:
        if self._pruning is None:
            self._pruning = FelsensteinPruning(
                self.tree,
                n_states=2,
                use_jax=self.use_jax,
            )
        return self._pruning

    def _presence(self) -> np.ndarray:
        if self._presence_matrix is None:
            self._presence_matrix = self.tip_observations.to_matrix()
        return self._presence_matrix

    def log_likelihood(
        self,
        data: Optional[Any],
        baseline: GeneContentBaseline,
        graph: Optional[Any],
    ) -> float:
        """
        Compute log-likelihood of the observed gene presence/absence matrix.
        
        Args:
            data: Optional override containing presence_matrix/taxon_names/family_names.
            baseline: GeneContentBaseline supplying gain/loss rates per family.
            graph: TransitionGraph (unused for tree-based models).
        """
        if not isinstance(baseline, GeneContentBaseline):
            raise TypeError("GeneContentObservationModel requires a GeneContentBaseline")

        tip_obs = self.tip_observations
        presence_matrix = getattr(data, "presence_matrix", None)
        family_names = getattr(data, "family_names", None)
        taxon_names = getattr(data, "taxon_names", None)

        if presence_matrix is None:
            presence_matrix = self._presence()
        if family_names is None:
            family_names = tip_obs.family_ids
        if taxon_names is None:
            taxon_names = tip_obs.taxon_ids

        total_ll = 0.0
        if self.use_rust:
            rates = baseline.get_all_rates(family_names)
            gain_rates = np.array([rates[fam].gain_rate for fam in family_names])
            loss_rates = np.array([rates[fam].loss_rate for fam in family_names])

            log_liks = compute_likelihoods_batch(
                self.tree,
                presence_matrix,
                gain_rates,
                loss_rates,
                taxon_names,
                use_rust=True,
            )
            total_ll = float(np.sum(log_liks))
        else:
            pruning = self._ensure_pruning()
            for idx, fam in enumerate(family_names):
                rates = baseline.get_rates(fam)
                transition_provider = SimpleBinaryTransitionProvider(
                    gain_rate=rates.gain_rate,
                    loss_rate=rates.loss_rate,
                )
                tip_provider = ArrayTipConditionalProvider(
                    data=presence_matrix[:, idx : idx + 1],
                    taxon_names=taxon_names,
                    n_states=2,
                )
                result = pruning.compute_likelihood(
                    transition_provider=transition_provider,
                    tip_provider=tip_provider,
                    n_sites=1,
                )
                total_ll += float(result.log_likelihood)

        return total_ll
