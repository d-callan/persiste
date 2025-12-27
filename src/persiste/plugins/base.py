"""Base plugin class."""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class PluginBase(ABC):
    """
    Base class for PERSISTE plugins.
    
    Plugins provide domain-specific:
    - State space definitions
    - Baseline process specifications
    - Data loaders
    - Pre-built analysis workflows
    - Visualization templates
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name (e.g., 'phylo', 'assembly')."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @property
    def state_spaces(self) -> Dict[str, type]:
        """Dictionary of state space factories."""
        return {}
    
    @property
    def baselines(self) -> Dict[str, type]:
        """Dictionary of baseline process factories."""
        return {}
    
    @property
    def analyses(self) -> Dict[str, type]:
        """Dictionary of pre-built analysis workflows."""
        return {}
    
    @property
    def loaders(self) -> Dict[str, type]:
        """Dictionary of data loaders."""
        return {}
    
    @property
    def priors(self) -> Dict[str, Any]:
        """Dictionary of domain-specific priors."""
        return {}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, version={self.version})"
