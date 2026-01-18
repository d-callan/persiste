"""Decoupled diagnostics suite for assembly inference."""

from persiste.plugins.assembly.diagnostics.artifacts import (
    CachedPathData,
    DiagnosticArtifacts,
    InferenceArtifacts,
)
from persiste.plugins.assembly.diagnostics.suite import (
    BaselineSensitivityResult,
    NullResamplingResult,
    ProfileLikelihoodResult,
    baseline_sensitivity,
    null_resampling,
    profile_likelihood,
)

__all__ = [
    "DiagnosticArtifacts",
    "InferenceArtifacts",
    "CachedPathData",
    "null_resampling",
    "profile_likelihood",
    "baseline_sensitivity",
    "NullResamplingResult",
    "ProfileLikelihoodResult",
    "BaselineSensitivityResult",
]
