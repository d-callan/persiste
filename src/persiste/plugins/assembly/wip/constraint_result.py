"""
Constraint inference result with built-in diagnostics.

This makes it impossible to ignore robustness checks.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ProfileDiagnostics:
    """Profile likelihood diagnostics for a single parameter."""
    parameter: str
    peak_value: float
    ll_range: float
    curvature: float
    identifiable: bool
    evidence: str  # "none", "weak", "moderate", "strong"
    
    def __str__(self) -> str:
        return f"{self.parameter}: {self.evidence} ({self.ll_range:.1f})"


@dataclass
class ConstraintResult:
    """
    Result of constraint inference with automatic diagnostics.
    
    This is the safe-by-default API - users can't accidentally
    ignore robustness checks.
    
    Attributes:
        estimate: Fitted constraint parameters θ
        baseline_params: Fitted baseline parameters φ (if joint inference)
        ll_constrained: Log-likelihood of constrained model
        ll_null: Log-likelihood of null model (θ=0)
        delta_ll: Improvement over null (ll_constrained - ll_null)
        evidence: Evidence class ("none", "weak", "moderate", "strong")
        identifiable: Whether parameters are identifiable
        profile_diagnostics: Profile likelihood diagnostics per parameter
        warnings: List of warning messages
        robustness_score: Overall robustness score (0-1)
        baseline_sensitivity: Stability across baseline variations (if tested)
        cv_score: Cross-validation score (if tested)
    """
    
    estimate: Dict[str, float]
    baseline_params: Optional[Dict[str, float]]
    ll_constrained: float
    ll_null: float
    delta_ll: float
    evidence: str
    identifiable: bool
    profile_diagnostics: Dict[str, ProfileDiagnostics]
    warnings: List[str]
    robustness_score: float
    baseline_sensitivity: Optional[Dict[str, float]] = None
    cv_score: Optional[float] = None
    
    def __str__(self) -> str:
        """Human-readable summary."""
        lines = []
        lines.append("=" * 60)
        lines.append("Constraint Inference Result")
        lines.append("=" * 60)
        
        # Evidence
        lines.append(f"\nEvidence: {self.evidence.upper()}")
        lines.append(f"  Δ LL: {self.delta_ll:.2f}")
        lines.append(f"  Identifiable: {self.identifiable}")
        lines.append(f"  Robustness: {self.robustness_score:.2f}")
        
        # Parameters
        lines.append(f"\nConstraint Parameters:")
        for param, value in self.estimate.items():
            diag = self.profile_diagnostics.get(param)
            if diag:
                lines.append(f"  {param}: {value:.3f} ({diag.evidence})")
            else:
                lines.append(f"  {param}: {value:.3f}")
        
        # Baseline (if joint)
        if self.baseline_params:
            lines.append(f"\nBaseline Parameters:")
            for param, value in self.baseline_params.items():
                lines.append(f"  {param}: {value:.3f}")
        
        # Warnings
        if self.warnings:
            lines.append(f"\n⚠ Warnings:")
            for warning in self.warnings:
                lines.append(f"  • {warning}")
        
        # Cross-validation
        if self.cv_score is not None:
            lines.append(f"\nCross-validation score: {self.cv_score:.3f}")
        
        # Baseline sensitivity
        if self.baseline_sensitivity:
            lines.append(f"\nBaseline sensitivity:")
            for param, std in self.baseline_sensitivity.items():
                lines.append(f"  {param}: ±{std:.3f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def is_significant(self, threshold: float = 10.0) -> bool:
        """
        Check if constraints are significant.
        
        Args:
            threshold: Δ LL threshold (default: 10 for conservative)
        
        Returns:
            True if Δ LL > threshold
        """
        return self.delta_ll > threshold
    
    def get_recommendation(self) -> str:
        """
        Get recommendation for interpreting results.
        
        Returns:
            Human-readable recommendation
        """
        if self.evidence == "none":
            return "No evidence for constraints. Use null model."
        elif self.evidence == "weak":
            return "Weak evidence. Interpret with caution. Consider more data."
        elif self.evidence == "moderate":
            if self.warnings:
                return "Moderate evidence, but warnings present. Validate carefully."
            else:
                return "Moderate evidence. Constraints likely real but validate."
        else:  # strong
            if self.warnings:
                return "Strong evidence, but warnings present. Check robustness."
            else:
                return "Strong evidence. Constraints are well-supported."


def classify_evidence(delta_ll: float, identifiable: bool, warnings: List[str]) -> str:
    """
    Classify evidence strength.
    
    Args:
        delta_ll: Improvement over null
        identifiable: Whether parameters are identifiable
        warnings: List of warnings
    
    Returns:
        Evidence class: "none", "weak", "moderate", "strong"
    """
    if delta_ll < 2.0:
        return "none"
    elif delta_ll < 5.0:
        return "weak"
    elif delta_ll < 10.0:
        return "moderate"
    else:
        # Strong evidence, but check for issues
        if not identifiable or len(warnings) > 2:
            return "moderate"  # Downgrade if problems
        else:
            return "strong"


def compute_robustness_score(
    identifiable: bool,
    delta_ll: float,
    warnings: List[str],
    baseline_sensitivity: Optional[Dict[str, float]] = None,
    cv_score: Optional[float] = None,
) -> float:
    """
    Compute overall robustness score (0-1).
    
    Higher is better. Combines multiple diagnostics.
    
    Args:
        identifiable: Whether parameters are identifiable
        delta_ll: Improvement over null
        warnings: List of warnings
        baseline_sensitivity: Parameter stability across baselines
        cv_score: Cross-validation score
    
    Returns:
        Robustness score between 0 and 1
    """
    score = 0.0
    
    # Identifiability (30%)
    if identifiable:
        score += 0.3
    
    # Evidence strength (30%)
    if delta_ll > 20:
        score += 0.3
    elif delta_ll > 10:
        score += 0.2
    elif delta_ll > 5:
        score += 0.1
    
    # Warnings (20%)
    if len(warnings) == 0:
        score += 0.2
    elif len(warnings) == 1:
        score += 0.1
    
    # Baseline sensitivity (10%)
    if baseline_sensitivity is not None:
        max_std = max(baseline_sensitivity.values())
        if max_std < 0.2:
            score += 0.1
        elif max_std < 0.5:
            score += 0.05
    else:
        score += 0.05  # Neutral if not tested
    
    # Cross-validation (10%)
    if cv_score is not None:
        if cv_score > 0.8:
            score += 0.1
        elif cv_score > 0.5:
            score += 0.05
    else:
        score += 0.05  # Neutral if not tested
    
    return score
