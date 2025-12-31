"""
Cross-plugin analysis recipes for PersiSTE.

This directory contains recipes that ORCHESTRATE multiple plugins for
integrated analyses. These are NOT joint likelihood models - they run
plugins sequentially and integrate results biologically.

Philosophy:
- Single-plugin recipes: self-contained, plugin-specific questions
- Cross-plugin recipes: orchestration, biological integration
- No joint likelihoods: separate analysis + integration is more powerful

Structure:
- recipes/                          (this directory - cross-plugin orchestration)
  - gene_dosage_effect.py           (GeneContent + CopyNumberDynamics)
  - strain_heterogeneity.py         (future: multi-plugin strain analysis)
  - host_conditioned_dosage.py      (future: environment-dependent dosage)
  
- src/persiste/plugins/*/recipes/   (single-plugin recipes)
  - copynumber/recipes/             (CN-specific recipes 0-3)
  - genecontent/recipes/            (presence/absence recipes)
  - assembly/recipes/               (assembly-specific recipes)

Design Rationale:
Cross-plugin recipes are orchestration protocols, not models:
1. Run Plugin A (e.g., GeneContent for retention)
2. Run Plugin B conditional on A (e.g., CopyNumberDynamics on present genes)
3. Integrate results biologically (pattern classification)

Why not joint likelihoods?
- Presence/absence events are rare but decisive
- CN changes are frequent but subtle
- Forcing them into one CTMC dilutes both signals
- Separate analysis + biological integration captures more

Example:
    from persiste.recipes import gene_dosage_effect
    
    report = gene_dosage_effect(
        cn_matrix="data/cn_matrix.tsv",
        family_names=families,
        taxon_names=taxa,
        tree="data/tree.nwk",
    )
    
    # Identify essential dosage-sensitive genes
    essential = [
        fam for fam, pattern in report.integrated_patterns.items()
        if pattern == "essential_dosage_sensitive"
    ]
"""

from .gene_dosage_effect import gene_dosage_effect, GeneDosageEffectReport

__all__ = [
    'gene_dosage_effect',
    'GeneDosageEffectReport',
]
