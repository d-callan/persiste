"""
Base class for diagnostic recipe reports.

Ensures consistent interface across all recipes.
"""

from abc import ABC, abstractmethod
from typing import Literal


class DiagnosticReport(ABC):
    """
    Base class for all diagnostic recipe reports.

    All recipes must implement:
    - severity: 'ok', 'warning', 'fail'
    - recommendation: Actionable next step
    - print_summary(): Human-readable output
    """

    severity: Literal["ok", "warning", "fail"]
    recommendation: str

    @abstractmethod
    def print_summary(self) -> None:
        """Print human-readable summary."""
        pass

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}
