import pytest
import persiste_rust
from persiste.plugins.assembly.states.assembly_state import AssemblyState

def test_rust_simulation_returns_state_ids():
    """Test that the Rust simulation returns paths with state IDs and a populated discovered_states dict."""
    primitives = ["A", "B"]
    initial_parts = ["A"]
    theta = {"reuse_count": 1.0}
    
    results = persiste_rust.simulate_assembly_trajectories(
        primitives=primitives,
        initial_parts=initial_parts,
        theta=theta,
        n_samples=10,
        t_max=10.0,
        burn_in=0.0,
        max_depth=5,
        seed=42
    )
    
    assert "paths" in results
    assert "discovered_states" in results
    
    # Check paths
    for path in results["paths"]:
        assert "final_state_id" in path
        assert isinstance(path["final_state_id"], int)
        assert path["final_state_id"] != 0
        
        # Verify the final state is in discovered_states
        final_id = path["final_state_id"]
        assert final_id in results["discovered_states"]

def test_discovered_states_format():
    """Test that discovered_states from Rust has the expected structure and contents."""
    primitives = ["A", "B"]
    results = persiste_rust.simulate_assembly_trajectories(
        primitives=primitives,
        initial_parts=["A"],
        theta={},
        n_samples=5,
        t_max=5.0,
        burn_in=0.0,
        max_depth=3,
        seed=1
    )
    
    discovered = results["discovered_states"]
    assert len(discovered) > 0
    
    for state_id, info in discovered.items():
        assert isinstance(state_id, int)
        assert "parts" in info
        assert "depth" in info
        assert "motifs" in info
        
        # Parts should be a dict of part_name -> count
        assert isinstance(info["parts"], dict)
        for part, count in info["parts"].items():
            assert part in primitives
            assert isinstance(count, int)
            assert count > 0

def test_state_id_stability_across_calls():
    """Test that the same state gets the same ID across different simulation calls."""
    primitives = ["A", "B"]
    
    # Call 1
    res1 = persiste_rust.simulate_assembly_trajectories(
        primitives=primitives, initial_parts=["A", "B"], theta={}, 
        n_samples=1, t_max=0.0, burn_in=0.0, max_depth=2, seed=1
    )
    id1 = res1["paths"][0]["final_state_id"]
    
    # Call 2 (different seed, same initial state)
    res2 = persiste_rust.simulate_assembly_trajectories(
        primitives=primitives, initial_parts=["A", "B"], theta={}, 
        n_samples=1, t_max=0.0, burn_in=0.0, max_depth=2, seed=2
    )
    id2 = res2["paths"][0]["final_state_id"]
    
    assert id1 == id2
    
    # Verify it matches AssemblyState.stable_id
    state = AssemblyState.from_parts(["A", "B"], depth=1)
    assert state.stable_id == id1
