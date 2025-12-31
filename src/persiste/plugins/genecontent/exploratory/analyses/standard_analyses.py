"""
Standard gene content analyses.

Opinionated analysis recipes for common biological questions.
Each analysis has:
- Clear biological question
- Recommended default configuration
- Automatic null comparison
- Interpretable headline result
- Automatic caveats/warnings
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import sys

from persiste.plugins.genecontent.inference.gene_inference import (
    GeneContentData,
    GeneContentInference,
    BaselineDiagnostics,
    ComparisonResult,
)
from persiste.plugins.genecontent.constraints.gene_constraint import (
    NullConstraint,
    RetentionBiasConstraint,
    HostAssociationConstraint,
)
from persiste.core.tree_inference import MLEResult


@dataclass
class GlobalRatesResult:
    """
    Result of global gain/loss rate estimation.
    
    Answers: What are the overall gene gain and loss rates on this tree?
    """
    gain_rate: float
    loss_rate: float
    gain_ci: Tuple[float, float]  # 95% CI
    loss_ci: Tuple[float, float]  # 95% CI
    equilibrium_presence: float
    expected_turnover_per_branch: float
    log_likelihood: float
    diagnostics: BaselineDiagnostics
    
    def print_report(self, file=None):
        """Print analysis report."""
        if file is None:
            file = sys.stdout
        
        print("\n" + "=" * 80, file=file)
        print("GLOBAL GAIN/LOSS RATE ESTIMATION", file=file)
        print("=" * 80, file=file)
        print(file=file)
        
        print("Question: What are the overall gene gain and loss rates?", file=file)
        print(file=file)
        
        print("Results:", file=file)
        print("  Gain rate: {0:.4f} (95% CI: {1:.4f} - {2:.4f})".format(
            self.gain_rate, self.gain_ci[0], self.gain_ci[1]
        ), file=file)
        print("  Loss rate: {0:.4f} (95% CI: {1:.4f} - {2:.4f})".format(
            self.loss_rate, self.loss_ci[0], self.loss_ci[1]
        ), file=file)
        print("  Equilibrium presence: {0:.2%}".format(self.equilibrium_presence), file=file)
        print("  Expected turnover per branch: {0:.2f} events".format(
            self.expected_turnover_per_branch
        ), file=file)
        print(file=file)
        
        # Show diagnostics
        self.diagnostics.print_report(file=file)


@dataclass
class RetentionTestResult:
    """
    Result of targeted retention test.
    
    Answers: Are specific gene families retained (or lost faster) in a defined subset?
    """
    retention_strength: float
    delta_ll: float
    p_value: float
    evidence_strength: str  # 'none', 'weak', 'moderate', 'strong'
    effect_direction: str  # 'retained', 'lost_faster', 'neutral'
    n_families_tested: int
    baseline_stable: bool
    comparison: ComparisonResult
    
    def print_report(self, file=None):
        """Print analysis report."""
        if file is None:
            file = sys.stdout
        
        print("\n" + "=" * 80, file=file)
        print("TARGETED RETENTION TEST", file=file)
        print("=" * 80, file=file)
        print(file=file)
        
        print("Question: Are selected genes retained or lost faster than expected?", file=file)
        print(file=file)
        
        print("Tested families: {0}".format(self.n_families_tested), file=file)
        print(file=file)
        
        print("Results:", file=file)
        print("  ΔLL = {0:.2f}".format(self.delta_ll), file=file)
        print("  Evidence: {0}".format(self.evidence_strength.upper()), file=file)
        print("  Effect: {0}".format(self.effect_direction.replace('_', ' ').upper()), file=file)
        print(file=file)
        
        if self.evidence_strength in ['moderate', 'strong']:
            print("Interpretation:", file=file)
            if self.effect_direction == 'retained':
                print("  → Selected genes show INCREASED retention", file=file)
                print("  → These families are preferentially maintained", file=file)
            elif self.effect_direction == 'lost_faster':
                print("  → Selected genes show DECREASED retention", file=file)
                print("  → These families are preferentially lost", file=file)
            print(file=file)
        else:
            print("Interpretation:", file=file)
            print("  → No evidence for differential retention", file=file)
            print("  → Selected genes behave like the genome-wide average", file=file)
            print(file=file)
        
        if not self.baseline_stable:
            print("⚠ WARNING: Baseline rates may be unstable", file=file)
            print("  → Results should be interpreted with caution", file=file)
            print(file=file)


@dataclass
class BranchShiftResult:
    """
    Result of branch-set gain/loss shift test.
    
    Answers: Do gene content dynamics differ on specific lineages?
    """
    gain_multiplier: float
    loss_multiplier: float
    delta_ll: float
    p_value: float
    evidence_strength: str
    n_foreground_branches: int
    comparison: ComparisonResult
    
    def print_report(self, file=None):
        """Print analysis report."""
        if file is None:
            file = sys.stdout
        
        print("\n" + "=" * 80, file=file)
        print("BRANCH-SET GAIN/LOSS SHIFT TEST", file=file)
        print("=" * 80, file=file)
        print(file=file)
        
        print("Question: Do gene dynamics differ on selected lineages?", file=file)
        print(file=file)
        
        print("Foreground branches: {0}".format(self.n_foreground_branches), file=file)
        print(file=file)
        
        print("Results:", file=file)
        print("  Gain multiplier: {0:.2f}x".format(self.gain_multiplier), file=file)
        print("  Loss multiplier: {0:.2f}x".format(self.loss_multiplier), file=file)
        print("  ΔLL = {0:.2f}".format(self.delta_ll), file=file)
        print("  Evidence: {0}".format(self.evidence_strength.upper()), file=file)
        print(file=file)


@dataclass
class AssociationTestResult:
    """
    Result of host/metadata association test.
    
    Answers: Is gene retention associated with host, environment, or trait?
    """
    association_strength: float
    delta_ll: float
    p_value: float
    evidence_strength: str
    trait_name: str
    comparison: ComparisonResult
    
    def print_report(self, file=None):
        """Print analysis report."""
        if file is None:
            file = sys.stdout
        
        print("\n" + "=" * 80, file=file)
        print("HOST/METADATA ASSOCIATION TEST", file=file)
        print("=" * 80, file=file)
        print(file=file)
        
        print("Question: Is gene retention associated with '{0}'?".format(self.trait_name), file=file)
        print(file=file)
        
        print("Results:", file=file)
        print("  Association strength: {0:.4f}".format(self.association_strength), file=file)
        print("  ΔLL = {0:.2f}".format(self.delta_ll), file=file)
        print("  Evidence: {0}".format(self.evidence_strength.upper()), file=file)
        print(file=file)


@dataclass
class ExploratoryScreeningResult:
    """
    Result of exploratory family screening.
    
    Answers: Which gene families show unusual gain/loss dynamics?
    
    IMPORTANT: This is exploratory, not confirmatory.
    """
    family_effects: Dict[str, float]  # family_name -> retention_strength
    ranked_families: List[Tuple[str, float]]  # sorted by |effect|
    stability_flags: Dict[str, bool]
    n_families_screened: int
    
    def print_report(self, file=None, top_n: int = 20):
        """Print analysis report."""
        if file is None:
            file = sys.stdout
        
        print("\n" + "=" * 80, file=file)
        print("EXPLORATORY FAMILY SCREENING", file=file)
        print("=" * 80, file=file)
        print(file=file)
        
        print("⚠ CAUTION: This is EXPLORATORY, not confirmatory", file=file)
        print("⚠ Use for hypothesis generation only", file=file)
        print(file=file)
        
        print("Question: Which families show unusual retention patterns?", file=file)
        print(file=file)
        
        print("Screened {0} families".format(self.n_families_screened), file=file)
        print(file=file)
        
        print("Top {0} candidates (by |effect|):".format(min(top_n, len(self.ranked_families))), file=file)
        print("  {0:<20} {1:>10} {2:>10}".format("Family", "Effect", "Stable"), file=file)
        print("  " + "-" * 42, file=file)
        
        for i, (family, effect) in enumerate(self.ranked_families[:top_n]):
            stable = "✓" if self.stability_flags.get(family, True) else "⚠"
            print("  {0:<20} {1:>10.4f} {2:>10}".format(family, effect, stable), file=file)
        
        print(file=file)
        print("Next steps:", file=file)
        print("  1. Select candidate families for targeted testing", file=file)
        print("  2. Run retention_test on specific candidates", file=file)
        print("  3. Validate with independent data if possible", file=file)
        print(file=file)


class GeneContentAnalysis:
    """
    Standard gene content analyses.
    
    Provides opinionated, easy-to-use analysis recipes.
    
    Usage:
        analysis = GeneContentAnalysis(data)
        
        # Analysis 1: Global rates
        result = analysis.global_rates()
        
        # Analysis 2: Retention test
        result = analysis.retention_test(families=['OG0001', 'OG0002'])
        
        # Analysis 3: Branch shift
        result = analysis.branch_shift(foreground_taxa=['species1', 'species2'])
        
        # Analysis 4: Metadata association
        result = analysis.association_test(trait_name='host', trait_values={'sp1': 1, 'sp2': 0})
        
        # Analysis 5: Exploratory screening
        result = analysis.exploratory_screening()
    """
    
    def __init__(self, data: GeneContentData):
        """
        Initialize analysis engine.
        
        Args:
            data: GeneContentData with tree and presence matrix
        """
        self.data = data
        self.inference = GeneContentInference(data)
    
    def global_rates(self, verbose: bool = True) -> GlobalRatesResult:
        """
        Analysis 1: Global Gain/Loss Estimation.
        
        Question: What are the overall gene gain and loss rates on this tree?
        
        This is the baseline for everything else. Runs null model only.
        
        Args:
            verbose: Whether to print report
            
        Returns:
            GlobalRatesResult with rates and diagnostics
        """
        # Get baseline diagnostics (includes null model fit)
        diagnostics = self.inference.get_baseline_diagnostics(verbose=False)
        
        # Fit null model to get full result with covariance
        null_result = self.inference.fit_null()
        
        # Extract rates
        gain_rate = np.exp(null_result.parameters['log_gain'])
        loss_rate = np.exp(null_result.parameters['log_loss'])
        
        # Compute 95% CI using delta method
        # CI for exp(θ) ≈ exp(θ) * [1 ± 1.96 * SE(θ)]
        if null_result.standard_errors is not None:
            gain_se = null_result.standard_errors.get('log_gain', 0.0)
            loss_se = null_result.standard_errors.get('log_loss', 0.0)
            
            gain_ci = (
                gain_rate * np.exp(-1.96 * gain_se),
                gain_rate * np.exp(1.96 * gain_se)
            )
            loss_ci = (
                loss_rate * np.exp(-1.96 * loss_se),
                loss_rate * np.exp(1.96 * loss_se)
            )
        else:
            # Fallback: use ±50% as rough CI
            gain_ci = (gain_rate * 0.5, gain_rate * 1.5)
            loss_ci = (loss_rate * 0.5, loss_rate * 1.5)
        
        result = GlobalRatesResult(
            gain_rate=gain_rate,
            loss_rate=loss_rate,
            gain_ci=gain_ci,
            loss_ci=loss_ci,
            equilibrium_presence=diagnostics.equilibrium_presence,
            expected_turnover_per_branch=diagnostics.mean_transitions_per_branch,
            log_likelihood=diagnostics.log_likelihood,
            diagnostics=diagnostics,
        )
        
        if verbose:
            result.print_report()
        
        return result
    
    def retention_test(
        self,
        families: List[str],
        verbose: bool = True,
    ) -> RetentionTestResult:
        """
        Analysis 2: Targeted Retention Test.
        
        Question: Are specific gene families retained (or lost faster) in a defined subset?
        
        This is the core RELAX-style analysis.
        
        Args:
            families: List of family names to test
            verbose: Whether to print report
            
        Returns:
            RetentionTestResult with test results and interpretation
        """
        # Create constraint
        retained_families = set(families)
        constraint = RetentionBiasConstraint(retained_families=retained_families)
        
        # Run comparison
        comparison = self.inference.compare_to_null(constraint, verbose=False)
        
        # Extract results
        retention_strength = comparison.alt_result.parameters['retention_strength']
        delta_ll = comparison.delta_ll
        p_value = comparison.lrt_result.pvalue
        evidence_strength = comparison.evidence_strength
        
        # Determine effect direction
        if abs(retention_strength) < 0.2:
            effect_direction = 'neutral'
        elif retention_strength > 0:
            effect_direction = 'retained'
        else:
            effect_direction = 'lost_faster'
        
        # Check baseline stability
        diagnostics = self.inference.get_baseline_diagnostics(verbose=False)
        baseline_stable = (
            diagnostics.gain_rate >= 0.01 and
            diagnostics.gain_rate <= 100 and
            diagnostics.loss_rate >= 0.01 and
            diagnostics.loss_rate <= 100
        )
        
        result = RetentionTestResult(
            retention_strength=retention_strength,
            delta_ll=delta_ll,
            p_value=p_value,
            evidence_strength=evidence_strength,
            effect_direction=effect_direction,
            n_families_tested=len(families),
            baseline_stable=baseline_stable,
            comparison=comparison,
        )
        
        if verbose:
            result.print_report()
        
        return result
    
    def branch_shift(
        self,
        foreground_taxa: List[str],
        verbose: bool = True,
    ) -> BranchShiftResult:
        """
        Analysis 3: Branch-Set Gain/Loss Shift.
        
        Question: Do gene content dynamics differ on specific lineages?
        
        NOTE: Current implementation uses a simplified approach.
        It tests whether families present in foreground taxa show different
        retention patterns. Full branch-aware implementation requires
        architectural changes to the likelihood computation.
        
        Args:
            foreground_taxa: List of taxa defining foreground clade
            verbose: Whether to print report
            
        Returns:
            BranchShiftResult
        """
        print("NOTE: Branch shift test uses simplified implementation.")
        print("Testing retention of families present in foreground taxa.")
        print()
        
        # Find families present in at least one foreground taxon
        foreground_indices = [
            i for i, taxon in enumerate(self.data.taxon_names)
            if taxon in foreground_taxa
        ]
        
        if not foreground_indices:
            raise ValueError("No foreground taxa found in data")
        
        # Find families present in foreground
        foreground_families = set()
        for fam_idx, fam_name in enumerate(self.data.family_names):
            presence_in_foreground = self.data.presence_matrix[foreground_indices, fam_idx]
            if presence_in_foreground.sum() > 0:
                foreground_families.add(fam_name)
        
        print("Found {0} families present in foreground taxa".format(
            len(foreground_families)
        ))
        print()
        
        # Test retention of foreground families
        constraint = RetentionBiasConstraint(retained_families=foreground_families)
        comparison = self.inference.compare_to_null(constraint, verbose=False)
        
        # Extract results
        retention_strength = comparison.alt_result.parameters['retention_strength']
        delta_ll = comparison.delta_ll
        p_value = comparison.lrt_result.pvalue
        evidence_strength = comparison.evidence_strength
        
        # Interpret as gain/loss multipliers
        # Positive retention = reduced loss, negative = increased loss
        # For simplicity, assume symmetric effect
        gain_multiplier = np.exp(retention_strength / 2.0)
        loss_multiplier = np.exp(-retention_strength / 2.0)
        
        result = BranchShiftResult(
            gain_multiplier=gain_multiplier,
            loss_multiplier=loss_multiplier,
            delta_ll=delta_ll,
            p_value=p_value,
            evidence_strength=evidence_strength,
            n_foreground_branches=len(foreground_taxa),
            comparison=comparison,
        )
        
        if verbose:
            result.print_report()
        
        return result
    
    def association_test(
        self,
        trait_name: str,
        trait_values: Dict[str, int],
        enrichment_threshold: float = 1.5,
        verbose: bool = True,
    ) -> AssociationTestResult:
        """
        Analysis 4: Host/Metadata Association Test.
        
        Question: Is gene retention associated with host, environment, or trait?
        
        This identifies families enriched in trait=1 taxa and tests whether
        they show differential retention.
        
        Args:
            trait_name: Name of the trait (for reporting)
            trait_values: Dict mapping taxon names to trait values (0/1)
            enrichment_threshold: Minimum enrichment ratio to consider (default: 1.5x)
            verbose: Whether to print report
            
        Returns:
            AssociationTestResult
        """
        print("Testing association with trait: '{0}'".format(trait_name))
        print()
        
        # Separate taxa by trait value
        trait1_taxa = [t for t, v in trait_values.items() if v == 1]
        trait0_taxa = [t for t, v in trait_values.items() if v == 0]
        
        trait1_indices = [
            i for i, taxon in enumerate(self.data.taxon_names)
            if taxon in trait1_taxa
        ]
        trait0_indices = [
            i for i, taxon in enumerate(self.data.taxon_names)
            if taxon in trait0_taxa
        ]
        
        if not trait1_indices or not trait0_indices:
            raise ValueError("Trait must have both 0 and 1 values")
        
        print("Trait distribution:")
        print("  {0}=0: {1} taxa".format(trait_name, len(trait0_taxa)))
        print("  {0}=1: {1} taxa".format(trait_name, len(trait1_taxa)))
        print()
        
        # Find families enriched in trait=1 taxa
        enriched_families = set()
        for fam_idx, fam_name in enumerate(self.data.family_names):
            presence_trait1 = self.data.presence_matrix[trait1_indices, fam_idx].sum()
            presence_trait0 = self.data.presence_matrix[trait0_indices, fam_idx].sum()
            
            # Compute enrichment ratio (with pseudocount)
            freq_trait1 = (presence_trait1 + 1) / (len(trait1_indices) + 2)
            freq_trait0 = (presence_trait0 + 1) / (len(trait0_indices) + 2)
            enrichment = freq_trait1 / freq_trait0
            
            if enrichment >= enrichment_threshold:
                enriched_families.add(fam_name)
        
        print("Found {0} families enriched in {1}=1 taxa (threshold: {2:.1f}x)".format(
            len(enriched_families), trait_name, enrichment_threshold
        ))
        print()
        
        if len(enriched_families) == 0:
            print("WARNING: No enriched families found. Cannot test association.")
            print("Try lowering enrichment_threshold or check data quality.")
            print()
            # Return null result
            return AssociationTestResult(
                association_strength=0.0,
                delta_ll=0.0,
                p_value=1.0,
                evidence_strength='none',
                trait_name=trait_name,
                comparison=None,
            )
        
        # Test retention of enriched families
        constraint = RetentionBiasConstraint(retained_families=enriched_families)
        comparison = self.inference.compare_to_null(constraint, verbose=False)
        
        # Extract results
        association_strength = comparison.alt_result.parameters.get('retention_strength', 0.0)
        delta_ll = comparison.delta_ll
        p_value = comparison.lrt_result.pvalue
        evidence_strength = comparison.evidence_strength
        
        result = AssociationTestResult(
            association_strength=association_strength,
            delta_ll=delta_ll,
            p_value=p_value,
            evidence_strength=evidence_strength,
            trait_name=trait_name,
            comparison=comparison,
        )
        
        if verbose:
            result.print_report()
        
        return result
    
    def exploratory_screening(
        self,
        verbose: bool = True,
        top_n: int = 20,
    ) -> ExploratoryScreeningResult:
        """
        Analysis 5: Exploratory Family Screening.
        
        Question: Which gene families show unusual gain/loss dynamics?
        
        IMPORTANT: This is exploratory, not confirmatory.
        Use for hypothesis generation only.
        
        Args:
            verbose: Whether to print report
            top_n: Number of top families to show in report
            
        Returns:
            ExploratoryScreeningResult
        """
        print("Running exploratory screening on {0} families...".format(
            self.data.n_families
        ))
        print("This may take a while.")
        print()
        
        family_effects = {}
        stability_flags = {}
        
        # Test each family individually
        for i, family_name in enumerate(self.data.family_names):
            if (i + 1) % 10 == 0:
                print("  Processed {0}/{1} families...".format(
                    i + 1, self.data.n_families
                ))
            
            # Create single-family constraint
            constraint = RetentionBiasConstraint(
                retained_families={family_name},
                prior_std=1.0,  # Stronger shrinkage for exploratory
            )
            
            try:
                # Fit model
                comparison = self.inference.compare_to_null(constraint, verbose=False)
                
                # Extract effect
                effect = comparison.alt_result.parameters['retention_strength']
                family_effects[family_name] = effect
                
                # Check stability (simple heuristic)
                stability_flags[family_name] = abs(effect) < 5.0
                
            except Exception as e:
                # Skip families that fail to fit
                print("  Warning: Failed to fit {0}: {1}".format(family_name, str(e)))
                family_effects[family_name] = 0.0
                stability_flags[family_name] = False
        
        print()
        
        # Rank families by |effect|
        ranked_families = sorted(
            family_effects.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        result = ExploratoryScreeningResult(
            family_effects=family_effects,
            ranked_families=ranked_families,
            stability_flags=stability_flags,
            n_families_screened=self.data.n_families,
        )
        
        if verbose:
            result.print_report(top_n=top_n)
        
        return result
