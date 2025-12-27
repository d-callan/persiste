"""Tests for plugin system."""

from persiste.plugins import plugins


def test_plugin_registry_list():
    """Test plugin listing (should be empty initially)."""
    plugin_list = plugins.list()
    assert isinstance(plugin_list, list)


def test_plugin_registry_load_nonexistent():
    """Test loading non-existent plugin raises KeyError."""
    import pytest
    
    with pytest.raises(KeyError):
        plugins.load('nonexistent_plugin')
