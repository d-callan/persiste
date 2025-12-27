"""Plugin discovery and loading."""

from typing import Dict, List, Optional
import importlib.metadata
from persiste.plugins.base import PluginBase


class PluginRegistry:
    """
    Registry for discovering and loading PERSISTE plugins.
    
    Plugins are discovered via entry points in the 'persiste.plugins' group.
    """
    
    def __init__(self):
        self._plugins: Dict[str, PluginBase] = {}
        self._discovered = False
    
    def discover(self) -> None:
        """Discover available plugins via entry points."""
        if self._discovered:
            return
        
        try:
            entry_points = importlib.metadata.entry_points()
            if hasattr(entry_points, 'select'):
                # Python 3.10+
                plugin_eps = entry_points.select(group='persiste.plugins')
            else:
                # Python 3.9
                plugin_eps = entry_points.get('persiste.plugins', [])
            
            for ep in plugin_eps:
                try:
                    plugin_class = ep.load()
                    plugin = plugin_class()
                    self._plugins[plugin.name] = plugin
                except Exception as e:
                    print(f"Warning: Failed to load plugin {ep.name}: {e}")
        
        except Exception as e:
            print(f"Warning: Plugin discovery failed: {e}")
        
        self._discovered = True
    
    def list(self) -> List[str]:
        """
        List available plugin names.
        
        Returns:
            List of plugin names
        """
        self.discover()
        return list(self._plugins.keys())
    
    def load(self, name: str) -> PluginBase:
        """
        Load a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance
            
        Raises:
            KeyError: If plugin not found
        """
        self.discover()
        
        if name not in self._plugins:
            available = ", ".join(self.list())
            raise KeyError(
                f"Plugin '{name}' not found. Available plugins: {available}"
            )
        
        return self._plugins[name]
    
    def __repr__(self) -> str:
        self.discover()
        return f"PluginRegistry(plugins={list(self._plugins.keys())})"


# Global plugin registry
plugins = PluginRegistry()
