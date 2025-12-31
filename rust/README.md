# Persiste Rust Acceleration

Rust-based parallel pruning for 5-10x speedup in gene content analysis.

## Building

### Prerequisites

1. Install Rust (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

2. Install maturin:
```bash
pip install maturin
```

### Build and Install

Development build (with debug symbols):
```bash
cd rust
maturin develop
```

Release build (optimized):
```bash
cd rust
maturin develop --release
```

This will compile the Rust code and install the `persiste_rust` Python module.

## Usage

```python
import numpy as np
from persiste_rust import compute_likelihoods_parallel

# Your data
tree_newick = "((A:1,B:1):1,(C:1,D:1):1);"
presence_matrix = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=np.int8)
gain_rates = np.array([1.5, 1.5, 1.5])
loss_rates = np.array([2.0, 2.0, 2.0])
taxon_names = ["A", "B", "C", "D"]

# Compute likelihoods in parallel
log_likelihoods = compute_likelihoods_parallel(
    tree_newick,
    presence_matrix,
    gain_rates,
    loss_rates,
    taxon_names
)

print(f"Log-likelihoods: {log_likelihoods}")
```

## Testing

Run Rust tests:
```bash
cd rust
cargo test
```

## Architecture

- `src/lib.rs` - Python bindings (PyO3)
- `src/tree.rs` - Tree structure and parsing
- `src/pruning.rs` - Felsenstein pruning algorithm
- Parallelization via Rayon (automatic thread pool)
