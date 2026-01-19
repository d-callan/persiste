"""
Centralized state ID resolution for bridging Python and Rust.
"""

import numpy as np
import persiste_rust
from persiste.plugins.assembly.states.assembly_state import AssemblyState

class StateIDResolver:
    """
    Resolves human-readable AssemblyState objects to the 64-bit IDs used by Rust.

    This works by running a zero-duration simulation in Rust for a given composition.
    Since the ID is a deterministic hash of the multiset in the backend, 
    this provides a reliable way to compute the same IDs in Python.
    """

    def __init__(self, primitives: list[str]):
        self.primitives = primitives
        self._cache: dict[str, int] = {}

    def resolve(self, state: AssemblyState) -> int:
        """Get 64-bit integer ID for an AssemblyState."""
        return state.stable_id

    def resolve_string(self, state_str: str | int) -> int:
        """
        Attempt to resolve a string representation to a Rust ID.
        Expects format: 'State(d=...: A, B, ...)' or a primitive name.
        """
        if not isinstance(state_str, str):
            # If it's already an int, assume it's a Rust ID and return it
            if isinstance(state_str, (int, np.integer)):
                return int(state_str)
            raise TypeError(f"expected string or bytes-like object, got {type(state_str)}")

        if state_str in self._cache:
            return self._cache[state_str]

        try:
            # 1. Try to parse as AssemblyState string
            import re
            match = re.search(r"State\(d=(\d+): (.*)\)", state_str)
            if match:
                depth = int(match.group(1))
                parts_raw = match.group(2)
                # Remove motifs if present: State(d=1: A [motif])
                if " [" in parts_raw:
                    parts_raw = parts_raw.split(" [")[0]
                
                parts_list = parts_raw.split(", ")
                parts = []
                for p in parts_list:
                    if "×" in p:
                        name, count = p.split("×")
                        parts.extend([name] * int(count))
                    else:
                        parts.append(p)
            elif state_str in self.primitives:
                # 2. Assume it's a primitive name
                parts = [state_str]
                depth = 0
            else:
                # Neither a valid state string nor a known primitive
                raise ValueError(f"String '{state_str}' is neither a valid state representation nor a known primitive.")
            
            # Use Rust's simulation to compute the ID deterministically
            # This is the "source of truth" strategy.
            results = persiste_rust.simulate_assembly_trajectories(
                primitives=self.primitives if self.primitives else parts,
                initial_parts=parts,
                theta={}, # theta doesn't matter for ID
                n_samples=1,  # 1 sample
                t_max=0.0, # 0 duration
                burn_in=0.0, # 0 burn-in
                max_depth=depth + 1, # max_depth
                seed=1,  # seed
                kappa=1.0, # kappa
                join_exponent=0.0, # join_exponent
                split_exponent=0.0, # split_exponent
                decay_rate=0.0, # decay_rate
            )
            rust_id = results["paths"][0]["final_state_id"]
            self._cache[state_str] = rust_id
            return rust_id
        except Exception as e:
            raise ValueError(f"Failed to resolve state string {state_str}: {e}")
