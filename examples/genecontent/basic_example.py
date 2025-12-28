"""
GeneContent Plugin: Basic Example

Demonstrates the core components of the GeneContent plugin:
1. Loading gene presence/absence data
2. Setting up baseline and constraint models
3. Computing transition probabilities

This is an exploratory example - not production-ready inference.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np

from persiste.plugins.genecontent.states.gene_state import (
    GenePresenceState,
    GeneFamilyVector,
    enumerate_transitions,
    transition_type,
)
from persiste.plugins.genecontent.baselines.gene_baseline import (
    HierarchicalRates,
    FixedRates,
    GlobalRates,
    RateParameters,
)
from persiste.plugins.genecontent.constraints.gene_constraint import (
    NullConstraint,
    RetentionBiasConstraint,
    PerFamilyConstraint,
)
from persiste.plugins.genecontent.observation.gene_observation import (
    TipObservations,
    GeneContentObservation,
)


def demo_state_model():
    """Demonstrate the state model."""
    print("=" * 60)
    print("1. STATE MODEL")
    print("=" * 60)
    
    # Single gene family state
    present = GenePresenceState.present_state("OG0001")
    absent = GenePresenceState.absent_state("OG0001")
    
    print(f"\nSingle gene states:")
    print(f"  Present: {present} (value={present.value})")
    print(f"  Absent:  {absent} (value={absent.value})")
    
    # Full genome content
    genome = GeneFamilyVector(
        presence={
            'OG0001': True,
            'OG0002': False,
            'OG0003': True,
            'OG0004': True,
            'OG0005': False,
        },
        taxon_id='SpeciesA'
    )
    
    print(f"\nFull genome: {genome}")
    print(f"  Present families: {genome.present_families}")
    print(f"  Absent families: {genome.absent_families}")
    
    # Transitions
    print(f"\nTransitions for OG0001:")
    for from_state, to_state in enumerate_transitions("OG0001"):
        ttype = transition_type(from_state, to_state)
        print(f"  {from_state} → {to_state} ({ttype})")


def demo_baseline_model():
    """Demonstrate baseline models."""
    print("\n" + "=" * 60)
    print("2. BASELINE MODEL")
    print("=" * 60)
    
    # Hierarchical baseline (recommended)
    print("\nHierarchical baseline (recommended):")
    hierarchical = HierarchicalRates(
        mu_gain=-2.0,    # Low gain rate
        sigma_gain=1.0,
        mu_loss=-1.0,    # Moderate loss rate
        sigma_loss=1.0,
    )
    
    # Sample rates for some families
    family_ids = ['OG0001', 'OG0002', 'OG0003']
    rates = hierarchical.get_all_rates(family_ids)
    
    for fam, r in rates.items():
        print(f"  {fam}: gain={r.gain_rate:.4f}, loss={r.loss_rate:.4f}")
    
    # Transition probability over branch length
    print("\nTransition probabilities for OG0001 (branch length = 0.1):")
    r = rates['OG0001']
    P = r.transition_probability(0.1)
    print(f"  P(absent→absent) = {P[0,0]:.4f}")
    print(f"  P(absent→present) = {P[0,1]:.4f}")
    print(f"  P(present→absent) = {P[1,0]:.4f}")
    print(f"  P(present→present) = {P[1,1]:.4f}")
    
    # Global baseline (simple)
    print("\nGlobal baseline (simple, for testing):")
    global_bl = GlobalRates(gain_rate=0.1, loss_rate=0.2)
    r = global_bl.get_rates('OG0001')
    print(f"  All families: gain={r.gain_rate:.4f}, loss={r.loss_rate:.4f}")


def demo_constraint_model():
    """Demonstrate constraint models."""
    print("\n" + "=" * 60)
    print("3. CONSTRAINT MODEL")
    print("=" * 60)
    
    # Null constraint
    print("\nNull constraint (no effect):")
    null = NullConstraint()
    effect = null.get_effect('OG0001')
    print(f"  OG0001: gain_mult={effect.gain_multiplier:.2f}, loss_mult={effect.loss_multiplier:.2f}")
    
    # Retention bias
    print("\nRetention bias constraint:")
    retention = RetentionBiasConstraint(
        retained_families={'OG0001', 'OG0003'},
        retention_strength=-1.0,  # ~2.7x reduction in loss
    )
    
    for fam in ['OG0001', 'OG0002', 'OG0003']:
        effect = retention.get_effect(fam)
        print(f"  {fam}: loss_mult={effect.loss_multiplier:.2f} {'(retained)' if fam in retention.retained_families else ''}")
    
    # Per-family constraint
    print("\nPer-family constraint:")
    per_fam = PerFamilyConstraint(
        effects={
            'OG0001': (-0.5, -1.0),  # Reduced gain and loss
            'OG0002': (0.5, 0.0),    # Increased gain, normal loss
        },
        regularization=0.1,
    )
    
    for fam in ['OG0001', 'OG0002', 'OG0003']:
        effect = per_fam.get_effect(fam)
        print(f"  {fam}: gain_mult={effect.gain_multiplier:.2f}, loss_mult={effect.loss_multiplier:.2f}")


def demo_observation_model():
    """Demonstrate observation model."""
    print("\n" + "=" * 60)
    print("4. OBSERVATION MODEL")
    print("=" * 60)
    
    # Create synthetic tip observations
    matrix = np.array([
        [1, 0, 1, 1, 0],  # Species A
        [1, 1, 1, 0, 0],  # Species B
        [0, 0, 1, 1, 1],  # Species C
        [1, 1, 0, 1, 0],  # Species D
    ], dtype=np.int8)
    
    taxon_ids = ['SpeciesA', 'SpeciesB', 'SpeciesC', 'SpeciesD']
    family_ids = ['OG0001', 'OG0002', 'OG0003', 'OG0004', 'OG0005']
    
    tips = TipObservations.from_matrix(matrix, taxon_ids, family_ids)
    
    print(f"\nTip observations: {tips}")
    
    # Summary statistics
    summary = tips.summary()
    print(f"\nSummary:")
    print(f"  Taxa: {summary['n_taxa']}")
    print(f"  Families: {summary['n_families']}")
    print(f"  Core genes: {summary['n_core']}")
    print(f"  Accessory genes: {summary['n_accessory']}")
    print(f"  Rare genes: {summary['n_rare']}")
    print(f"  Mean genes/taxon: {summary['mean_genes_per_taxon']:.1f}")
    
    # Per-family counts
    print(f"\nPer-family presence counts:")
    for fam in family_ids:
        count = tips.get_family_count(fam)
        print(f"  {fam}: {count}/{tips.n_taxa} taxa")
    
    # Observation model
    obs_model = GeneContentObservation(tips)
    
    print(f"\nTip conditional likelihoods (SpeciesA, OG0001):")
    cond = obs_model.get_tip_conditional('SpeciesA', 'OG0001')
    print(f"  P(obs=1 | state=0) = {cond[0]:.2f}")
    print(f"  P(obs=1 | state=1) = {cond[1]:.2f}")


def demo_effective_rates():
    """Demonstrate how baseline + constraint = effective rates."""
    print("\n" + "=" * 60)
    print("5. EFFECTIVE RATES (Baseline × Constraint)")
    print("=" * 60)
    
    # Setup baseline
    baseline = GlobalRates(gain_rate=0.1, loss_rate=0.2)
    
    # Setup constraint (retention bias)
    constraint = RetentionBiasConstraint(
        retained_families={'OG0001'},
        retention_strength=-1.0,
    )
    
    print("\nBaseline rates:")
    for fam in ['OG0001', 'OG0002']:
        r = baseline.get_rates(fam)
        print(f"  {fam}: gain={r.gain_rate:.3f}, loss={r.loss_rate:.3f}")
    
    print("\nConstraint effects:")
    for fam in ['OG0001', 'OG0002']:
        effect = constraint.get_effect(fam)
        print(f"  {fam}: gain_mult={effect.gain_multiplier:.3f}, loss_mult={effect.loss_multiplier:.3f}")
    
    print("\nEffective rates (baseline × constraint):")
    for fam in ['OG0001', 'OG0002']:
        r = baseline.get_rates(fam)
        effect = constraint.get_effect(fam)
        
        eff_gain = r.gain_rate * effect.gain_multiplier
        eff_loss = r.loss_rate * effect.loss_multiplier
        
        print(f"  {fam}: gain={eff_gain:.3f}, loss={eff_loss:.3f}")


if __name__ == '__main__':
    print("=" * 60)
    print("GeneContent Plugin: Basic Example")
    print("=" * 60)
    print("\nThis demonstrates the core components of the plugin.")
    print("Full inference (likelihood, MLE) coming in next iteration.")
    
    demo_state_model()
    demo_baseline_model()
    demo_constraint_model()
    demo_observation_model()
    demo_effective_rates()
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Implement Felsenstein pruning for likelihood")
    print("  - Add MLE inference")
    print("  - Test with real pan-genome data")
