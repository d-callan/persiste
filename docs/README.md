# Documentation

This directory contains development documentation, benchmark results, and historical records.

## Structure

### `benchmarks/`
Performance comparisons and tool benchmarks:
- `ECOLI_FULL_ANALYSIS.md` - Full E. coli dataset analysis
- `GLOOME_COMPARISON.md` - Comparison with GLOOME tool
- `TOOL_COMPARISON_FINAL.md` - Final tool comparison results
- `compare_*.py` - Comparison scripts
- `test_gloome_*.py` - GLOOME scaling tests

### `development/`
Development history and implementation notes:
- `PHASE_1_2_COMPLETE.md` - Phase 1 & 2 completion summary
- `RUST_IMPLEMENTATION_GUIDE.md` - Rust integration guide
- `RUST_INTEGRATION_COMPLETE.md` - Rust integration completion notes
- `SAMPLING_ANALYSIS.md` - Sampling bias analysis notes
- `SAMPLING_BIAS_DOCUMENTATION.md` - Old sampling bias docs (superseded)

### Root-level docs/
General repository documentation:
- `BENCHMARK_RESULTS.md` - Overall benchmark results
- `PERFORMANCE_OPTIMIZATIONS.md` - Performance optimization notes
- `RUST_IMPLEMENTATION_PLAN.md` - Rust implementation planning

## Current Framework Documentation

For current, production-ready documentation, see:

**Main framework:**
- `/STRAIN_HETEROGENEITY_FRAMEWORK.md` - **Primary reference** for strain heterogeneity analysis
- `/README.md` - Repository overview

**Plugin documentation:**
- `/src/persiste/plugins/genecontent/README.md` - GeneContent plugin guide
- `/src/persiste/plugins/genecontent/exploratory/README.md` - Exploratory scripts

**Rust acceleration:**
- `/rust/README.md` - Rust implementation details

## Navigation

**For users:** Start with `/STRAIN_HETEROGENEITY_FRAMEWORK.md` and `/README.md`

**For developers:** See plugin READMEs and this docs/ directory for historical context

**For benchmarking:** See `benchmarks/` subdirectory
