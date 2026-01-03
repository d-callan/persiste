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
from typing import Dict, List, Optional

import numpy as np

from persiste.core.tip_observations import OneHotTipObservation

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

    observations: Dict[str, GeneFamilyVector] = field(default_factory=dict)
    family_ids: List[str] = field(default_factory=list)
    taxon_ids: List[str] = field(default_factory=list)

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
        cls, matrix: np.ndarray, taxon_ids: List[str], family_ids: List[str]
    ) -> "TipObservations":
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


class GeneContentObservation:
    """
    Observation model for gene content likelihood computation.

    For v1, this is simple: observed states at tips are exact.
    The likelihood is computed via Felsenstein pruning on the tree.

    Later extensions:
    - Detection probability (some genes may be missed)
    - Missingness model (some taxa may have incomplete data)
    """

    def __init__(
        self,
        tip_observations: TipObservations,
        detection_prob: float = 1.0,
    ):
        """
        Initialize observation model.

        Args:
            tip_observations: Observed gene presence/absence at tips
            detection_prob: Probability of detecting a present gene (v1: always 1.0)
        """
        self.tip_observations = tip_observations
        self.detection_prob = detection_prob
        self._tip_model = OneHotTipObservation(2)

    @property
    def family_ids(self) -> List[str]:
        """Gene family IDs."""
        return self.tip_observations.family_ids

    @property
    def taxon_ids(self) -> List[str]:
        """Taxon IDs."""
        return self.tip_observations.taxon_ids

    def get_tip_state(self, taxon_id: str, family_id: str) -> int:
        """
        Get observed state at a tip.

        Args:
            taxon_id: Taxon identifier
            family_id: Gene family identifier

        Returns:
            1 if present, 0 if absent
        """
        obs = self.tip_observations.get_observation(taxon_id)
        return 1 if obs.is_present(family_id) else 0

    def get_tip_likelihood(self, taxon_id: str, family_id: str, state: int) -> float:
        """
        Compute likelihood of latent state given observation.

        For v1 (perfect observation):
            P(obs | state) = 1 if obs == state, 0 otherwise

        Args:
            taxon_id: Taxon identifier
            family_id: Gene family identifier
            state: Latent state (0 or 1)

        Returns:
            Likelihood P(observation | latent state)
        """
        observed = self.get_tip_state(taxon_id, family_id)

        if self.detection_prob >= 1.0:
            return float(self._tip_model.get_tip_likelihood(observed)[state])
        else:
            # Imperfect detection (future extension)
            if state == 1:  # Gene is present
                if observed == 1:
                    return self.detection_prob
                else:
                    return 1.0 - self.detection_prob
            else:  # Gene is absent
                if observed == 0:
                    return 1.0  # Can't observe what's not there
                else:
                    return 0.0  # False positive (not modeled in v1)

    def get_tip_conditional(self, taxon_id: str, family_id: str) -> np.ndarray:
        """
        Get conditional likelihood vector at a tip.

        Returns:
            Array of length 2: [P(obs | state=0), P(obs | state=1)]
        """
        observed = self.get_tip_state(taxon_id, family_id)

        if self.detection_prob >= 1.0:
            return self._tip_model.get_tip_likelihood(observed)

        return np.array(
            [
                self.get_tip_likelihood(taxon_id, family_id, 0),
                self.get_tip_likelihood(taxon_id, family_id, 1),
            ]
        )
