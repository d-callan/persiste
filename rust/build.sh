#!/bin/bash
# Build script for Rust extension

set -e

echo "=========================================="
echo "Building Persiste Rust Extension"
echo "=========================================="
echo ""

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust is not installed"
    echo "Install with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

echo "Building Rust extension (release mode)..."
maturin develop --release

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "Test with:"
echo "  python -c 'import persiste_rust; print(\"Rust extension loaded successfully!\")'"
echo ""
