"""
Recipe 4: Gene Dosage Effect (Cross-Plugin Orchestration)

Question: "How does gene retention interact with copy number dynamics?"

NOTE: This is a CROSS-PLUGIN recipe that orchestrates GeneContent + CopyNumberDynamics.
The actual implementation lives in: persiste/recipes/gene_dosage_effect.py

This stub provides a pointer and explains the design philosophy.
"""

from typing import Union, List, Optional
from pathlib import Path

from persiste.core.trees import Tree


def gene_dosage_effect(
    cn_matrix: Union[str, Path],
    family_names: List[str],
    taxon_names: List[str],
    tree: Union[Tree, str, Path],
    output_dir: Optional[str] = None,
    verbose: bool = True,
):
    """
    Cross-Plugin Recipe: Gene Dosage Effect
    
    This is an ORCHESTRATION recipe - it runs two plugins sequentially and
    integrates their results. No joint likelihood required.
    
    Design Philosophy:
    - Presence/absence events are rare but decisive
    - CN changes are frequent but subtle
    - Forcing them into one CTMC dilutes both signals
    - Separate analysis + biological integration is more powerful
    
    Workflow:
    1. Gene retention analysis (GeneContent)
       - Classify: core-like, accessory-like, lineage-restricted
       - Outputs: retention regime, π₁, λ, μ
    
    2. Conditional copy-number analysis (CopyNumberDynamics)
       - Analyze CN dynamics only where gene is present
       - Test: dosage stability, amplification bias
    
    3. Integrative interpretation (the "dosage effect")
       - High retention + strong buffering → Essential dosage-sensitive
       - High retention + amplification bias → Adaptive dosage modulation
       - Low retention + high amplification → Mobile elements / selfish genes
       - Host-specific retention + CN volatility → Environment-dependent dosage
    
    Args:
        cn_matrix: Copy number matrix (n_families, n_taxa)
        family_names: List of gene family names
        taxon_names: List of taxon names
        tree: Phylogenetic tree
        output_dir: Optional directory for output files
        verbose: Print progress and interpretation
    
    Returns:
        GeneDosageEffectReport with integrated analysis
    
    Example:
        >>> from persiste.recipes import gene_dosage_effect
        >>> 
        >>> report = gene_dosage_effect(
        ...     cn_matrix="data/cn_matrix.tsv",
        ...     family_names=families,
        ...     taxon_names=taxa,
        ...     tree="data/tree.nwk",
        ... )
        >>> 
        >>> # Identify essential dosage-sensitive genes
        >>> essential = [
        ...     fam for fam, pattern in report.integrated_patterns.items()
        ...     if pattern == "essential_dosage_sensitive"
        ... ]
    
    Note:
        The actual implementation lives in: persiste/recipes/gene_dosage_effect.py
        This stub provides a convenient import path from the CopyNumber plugin.
    """
    # Import from cross-plugin recipes
    from persiste.recipes.gene_dosage_effect import gene_dosage_effect as _gene_dosage_effect
    
    return _gene_dosage_effect(
        cn_matrix=cn_matrix,
        family_names=family_names,
        taxon_names=taxon_names,
        tree=tree,
        output_dir=output_dir,
        verbose=verbose,
    )


# Future implementation notes for developers:
"""
Implementation Strategy for Recipe 4 (v2+):

1. Model Structure:
   - State space: (presence, CN) joint states
   - Transitions: 
     * absent → present (gain)
     * present → absent (loss)
     * present + CN changes (conditional on presence)
   
2. Key Questions:
   a) Retained but dosage-unstable:
      - Low loss rate (stable presence)
      - High amplify/contract rates (volatile CN)
      - Example: stress response genes
   
   b) Stable presence AND dosage:
      - Low loss rate
      - Low CN transition rates
      - Example: housekeeping genes
   
   c) Amplification bias in core vs accessory:
      - Core (high presence): low amplification bias
      - Accessory (low presence): high amplification bias
      - Tests interaction: presence × amplification
   
   d) Recently gained genes:
      - Condition on recent gain events
      - Test if amplification rate is elevated
      - Suggests relaxed constraint on new genes

3. Technical Implementation:
   - Extend state space: 4 CN states × 2 presence states = 8 states
   - Or: condition CN model on presence state
   - Requires joint parameter estimation
   - May need hierarchical model for family heterogeneity

4. Recipe Location:
   - Should live in: src/persiste/recipes/ (multi-plugin)
   - Not in: src/persiste/plugins/copynumber/recipes/
   - Requires both GeneContent and CopyNumberDynamics

5. Interpretation:
   - "Core genes show stable presence but volatile dosage"
   - "Accessory genes show amplification bias when present"
   - "Recently gained genes are more likely to amplify"
   
   NOT: "Presence causes dosage instability"
   (Descriptive, not causal)
"""
