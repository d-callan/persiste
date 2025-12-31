# Development History

This directory contains historical development documentation and implementation notes.

## Contents

### Implementation Milestones
- **`PHASE_1_2_COMPLETE.md`** - Phase 1 & 2 completion summary
- **`RUST_IMPLEMENTATION_GUIDE.md`** - Guide for Rust integration
- **`RUST_INTEGRATION_COMPLETE.md`** - Rust integration completion notes

### Analysis Development
- **`SAMPLING_ANALYSIS.md`** - Initial sampling bias analysis
- **`SAMPLING_BIAS_DOCUMENTATION.md`** - Old gene-frequency-focused approach (superseded)

## Evolution of the Framework

### Phase 1: Core Implementation
- Basic GeneContent inference
- Tree inference methods
- PAM interface

### Phase 2: Performance Optimization
- Rust acceleration for distance matrices
- Parallelization strategies
- Profiling and optimization

### Phase 3: Strain Heterogeneity Framework
- Discovery of strain-level heterogeneity phenomenon
- Development of two-recipe framework
- Validation on E. coli dataset

## Current Status

**Superseded by:** `/STRAIN_HETEROGENEITY_FRAMEWORK.md`

The old sampling bias documentation focused on gene frequency (singletons, rare genes) rather than strain heterogeneity. The current framework recognizes that different strains can have fundamentally different gene dynamics, which is more biologically motivated and statistically principled.

## For Developers

These documents provide historical context for design decisions and implementation details. For current development:

1. See `/src/persiste/plugins/genecontent/README.md` for plugin architecture
2. See `/rust/README.md` for Rust implementation details
3. See `/STRAIN_HETEROGENEITY_FRAMEWORK.md` for current analytical framework
