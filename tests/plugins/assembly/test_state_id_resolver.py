import pytest
import numpy as np
from persiste.plugins.assembly.states.resolver import StateIDResolver
from persiste.plugins.assembly.states.assembly_state import AssemblyState

def test_state_id_resolver_basic():
    """Test that StateIDResolver can resolve simple primitive names."""
    primitives = ["A", "B", "C"]
    resolver = StateIDResolver(primitives)
    
    # Resolve a primitive
    id_a = resolver.resolve_string("A")
    assert isinstance(id_a, int)
    assert id_a != 0
    
    # Resolve same primitive again (should be cached)
    id_a_again = resolver.resolve_string("A")
    assert id_a == id_a_again

def test_state_id_resolver_complex_string():
    """Test resolution of complex state strings like 'State(d=1: A, B)'."""
    primitives = ["A", "B"]
    resolver = StateIDResolver(primitives)
    
    # Resolve a state string
    state_str = "State(d=1: A, B)"
    id_ab = resolver.resolve_string(state_str)
    assert isinstance(id_ab, int)
    
    # Resolve another state string with same composition but different order (if it were possible to get such a string)
    # The resolver parses "A, B" into ["A", "B"]
    state_str_alt = "State(d=1: B, A)"
    id_ba = resolver.resolve_string(state_str_alt)
    assert id_ab == id_ba

def test_state_id_resolver_multiplicity():
    """Test resolution of strings with multiplicity like 'A×2'."""
    primitives = ["A", "B"]
    resolver = StateIDResolver(primitives)
    
    state_str = "State(d=2: A×2, B)"
    id_aab = resolver.resolve_string(state_str)
    assert isinstance(id_aab, int)
    
    # Manually constructed state should have same ID
    state_obj = AssemblyState.from_parts(["A", "A", "B"], depth=2)
    assert resolver.resolve(state_obj) == id_aab

def test_state_id_resolver_consistency_with_assembly_state():
    """Test that StateIDResolver and AssemblyState.stable_id agree."""
    primitives = ["A", "B"]
    resolver = StateIDResolver(primitives)
    
    state = AssemblyState.from_parts(["A", "B"], depth=1)
    id_from_obj = state.stable_id
    id_from_resolver = resolver.resolve(state)
    id_from_str = resolver.resolve_string(str(state))
    
    assert id_from_obj == id_from_resolver
    assert id_from_obj == id_from_str

def test_state_id_resolver_invalid_input():
    """Test that resolver handles invalid strings gracefully."""
    resolver = StateIDResolver(["A", "B"])
    
    with pytest.raises(ValueError, match="Failed to resolve state string"):
        # Use a string that is definitely not in primitives and not a valid state string
        resolver.resolve_string("DefinitelyNotAPrimitive")
    
    with pytest.raises(TypeError):
        resolver.resolve_string(None)

def test_state_id_resolver_numeric_input():
    """Test that resolver returns numeric IDs if passed directly."""
    resolver = StateIDResolver(["A"])
    expected_id = 12345
    assert resolver.resolve_string(expected_id) == expected_id
    assert resolver.resolve_string(np.int64(expected_id)) == expected_id
