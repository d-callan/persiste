"""Tests for AssemblyState."""

import pytest
from persiste.plugins.assembly.states.assembly_state import AssemblyState


def test_create_simple_state():
    """Test creating a simple assembly state."""
    state = AssemblyState.from_parts(['A', 'B', 'C'], depth=2)
    
    assert state.assembly_depth == 2
    assert state.total_parts() == 3
    assert state.contains_part('A')
    assert state.contains_part('B')
    assert state.contains_part('C')
    assert not state.contains_part('D')


def test_create_multiset_state():
    """Test creating state with repeated parts."""
    state = AssemblyState.from_parts(['A', 'A', 'B'], depth=3)
    
    assert state.assembly_depth == 3
    assert state.total_parts() == 3
    parts_dict = state.get_parts_dict()
    assert parts_dict['A'] == 2
    assert parts_dict['B'] == 1


def test_state_with_motifs():
    """Test creating state with motif labels."""
    state = AssemblyState.from_parts(
        ['peptide'] * 5,
        depth=4,
        motifs={'helix', 'stable'}
    )
    
    assert state.assembly_depth == 4
    assert state.contains_motif('helix')
    assert state.contains_motif('stable')
    assert not state.contains_motif('sheet')


def test_empty_state():
    """Test creating empty state."""
    state = AssemblyState.empty()
    
    assert state.assembly_depth == 0
    assert state.total_parts() == 0
    assert len(state.motifs) == 0


def test_state_is_hashable():
    """Test that states can be used as dict keys and in sets."""
    s1 = AssemblyState.from_parts(['A', 'B'], depth=1)
    s2 = AssemblyState.from_parts(['A', 'B'], depth=1)
    s3 = AssemblyState.from_parts(['A', 'C'], depth=1)
    
    # Same state should hash to same value
    assert hash(s1) == hash(s2)
    
    # Can use in set
    state_set = {s1, s2, s3}
    assert len(state_set) == 2  # s1 and s2 are same
    
    # Can use as dict key
    state_dict = {s1: 'first', s3: 'third'}
    assert state_dict[s2] == 'first'  # s2 same as s1


def test_state_is_immutable():
    """Test that states are frozen (immutable)."""
    state = AssemblyState.from_parts(['A', 'B'], depth=1)
    
    with pytest.raises(AttributeError):
        state.assembly_depth = 2


def test_subassembly_check():
    """Test checking if one state is subassembly of another."""
    s1 = AssemblyState.from_parts(['A', 'B'], depth=1)
    s2 = AssemblyState.from_parts(['A', 'B', 'C'], depth=2)
    s3 = AssemblyState.from_parts(['A', 'A', 'B'], depth=2)
    
    assert s1.is_subassembly_of(s2)  # {A,B} ⊆ {A,B,C}
    assert not s2.is_subassembly_of(s1)  # {A,B,C} ⊄ {A,B}
    assert not s1.is_subassembly_of(s3)  # {A,B} ⊄ {A,A,B} (need 2 A's)


def test_state_ordering():
    """Test that states can be ordered for search algorithms."""
    s1 = AssemblyState.from_parts(['A'], depth=1)
    s2 = AssemblyState.from_parts(['A', 'B'], depth=2)
    s3 = AssemblyState.from_parts(['A', 'B'], depth=1)
    
    # Order by depth first
    assert s1 < s2
    assert s3 < s2
    
    # Then by total parts
    assert s1 < s3
    
    # Can sort
    states = [s2, s1, s3]
    sorted_states = sorted(states)
    assert sorted_states == [s1, s3, s2]


def test_state_string_representation():
    """Test human-readable string representation."""
    s1 = AssemblyState.from_parts(['A', 'B'], depth=1)
    s2 = AssemblyState.from_parts(['A', 'A', 'B'], depth=2)
    s3 = AssemblyState.from_parts(['A'], depth=1, motifs={'stable'})
    
    assert 'A' in str(s1)
    assert 'B' in str(s1)
    assert 'd=1' in str(s1)
    
    assert 'A×2' in str(s2)
    assert 'd=2' in str(s2)
    
    assert 'stable' in str(s3)


def test_invalid_state():
    """Test that invalid states raise errors."""
    # Negative depth
    with pytest.raises(ValueError):
        AssemblyState.from_parts(['A'], depth=-1)
    
    # Empty state with non-zero depth
    with pytest.raises(ValueError):
        AssemblyState.from_parts([], depth=1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
