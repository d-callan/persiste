# Repository Structure

Clean, organized structure separating production code from development artifacts.

## Root Level

```
persiste/
├── README.md                              # Main repository overview
├── STRAIN_HETEROGENEITY_FRAMEWORK.md      # Primary framework documentation
├── REPOSITORY_STRUCTURE.md                # This file - structure guide
├── src/                                   # Source code
├── docs/                                  # Documentation archive
├── rust/                                  # Rust acceleration
├── tests/                                 # Test suite
├── data/                                  # Data directory (gitignored)
├── pyproject.toml                         # Package configuration
└── requirements.txt                       # Python dependencies
```

## Production Code

### GeneContent Plugin (Production-Ready)
```
src/persiste/plugins/genecontent/
├── pam_interface.py                        # Main interface - START HERE
├── strain_diagnostics.py                   # Heterogeneity diagnostics
├── strain_recipes.py                        # Two-recipe framework
├── constraint.py                            # Parameter constraints
├── inference.py                             # Core inference engine
├── data.py                                  # Data structures
├── README.md                                # Plugin documentation
├── examples/                                # Example scripts & workflows
│   ├── example_strain_heterogeneity_workflow.py  # ⭐ RECOMMENDED TEMPLATE
│   ├── analyze_ecoli_real.py                # Real data example
│   ├── basic_example.py                     # Basic API examples
│   ├── pam_only_example.py
│   └── README.md
└── exploratory/                             # Development artifacts (archived)
    ├── analyze_strain_heterogeneity.py
    ├── test_*.py
    ├── benchmark_*.py
    ├── validation/
    ├── analyses/
    └── README.md
```

### Core Framework (Exploratory)
```
src/persiste/core/
├── inference.py               # Core inference
├── tree_inference.py          # Phylogenetic tree inference
├── simulation.py              # Binary trait simulation
├── pruning.py                 # Felsenstein pruning
└── ...
```

### Other Plugins (Exploratory)
```
src/persiste/plugins/
├── assembly/                  # Assembly Theory plugin
│   └── examples/              # Assembly examples
├── phylo/                     # Phylogenetics plugin (proof-of-concept)
└── ...
```

## Documentation

```
docs/
├── README.md                  # Documentation index
├── benchmarks/                # Performance & comparisons
│   ├── ECOLI_FULL_ANALYSIS.md
│   ├── GLOOME_COMPARISON.md
│   ├── TOOL_COMPARISON_FINAL.md
│   ├── results/               # Benchmark output files
│   ├── tool_comparison_output/
│   ├── compare_*.py
│   └── test_gloome_*.py
├── development/               # Development history
│   ├── PHASE_1_2_COMPLETE.md
│   ├── RUST_IMPLEMENTATION_GUIDE.md
│   ├── SAMPLING_BIAS_DOCUMENTATION.md (superseded)
│   └── README.md
├── BENCHMARK_RESULTS.md
├── PERFORMANCE_OPTIMIZATIONS.md
└── RUST_IMPLEMENTATION_PLAN.md
```

## Rust Acceleration

```
rust/
├── src/
│   └── lib.rs                 # Distance matrix computation (5-6x speedup)
├── Cargo.toml
└── README.md
```

## Key Files for Different Users

### For Users (Analyzing Pangenomes)
1. `/README.md` - Start here
2. `/STRAIN_HETEROGENEITY_FRAMEWORK.md` - **Primary reference**
3. `/src/persiste/plugins/genecontent/examples/example_strain_heterogeneity_workflow.py` - **Template**
4. `/src/persiste/plugins/genecontent/README.md` - Plugin guide

### For Developers (Maintaining Code)
1. `/src/persiste/plugins/genecontent/` - Production code
2. `/src/persiste/plugins/genecontent/README.md` - Architecture
3. `/rust/README.md` - Rust implementation
4. `/docs/development/` - Historical context

### For Benchmarking
1. `/docs/benchmarks/` - All benchmark results and scripts
2. `/docs/benchmarks/README.md` - Benchmark guide

## What's Where

### Production Code
- **GeneContent plugin:** `src/persiste/plugins/genecontent/` (main modules only)
- **Rust acceleration:** `rust/`
- **Example workflows:** `src/persiste/plugins/genecontent/examples/`
- **Assembly examples:** `src/persiste/plugins/assembly/examples/`

### Archived/Exploratory
- **Development scripts:** `src/persiste/plugins/genecontent/exploratory/`
- **Benchmark scripts:** `docs/benchmarks/`
- **Old validation:** `src/persiste/plugins/genecontent/exploratory/validation/`
- **Old analyses:** `src/persiste/plugins/genecontent/exploratory/analyses/`

### Documentation
- **Framework guide:** `STRAIN_HETEROGENEITY_FRAMEWORK.md` (root)
- **Repository structure:** `REPOSITORY_STRUCTURE.md` (root)
- **Plugin guide:** `src/persiste/plugins/genecontent/README.md`
- **Benchmark results:** `docs/benchmarks/`
- **Development history:** `docs/development/`

## Clean Separation

**Production code** (what developers maintain):
- Main plugin modules
- Core framework
- Example scripts
- Primary documentation

**Archived artifacts** (preserved for reference):
- Exploratory scripts
- Validation experiments
- Benchmark comparisons
- Development notes
- Old approaches (superseded)

All exploratory work is preserved but organized so production code is immediately accessible.

## Navigation Tips

1. **New to the framework?** Start with `/README.md` → `/STRAIN_HETEROGENEITY_FRAMEWORK.md`
2. **Want to analyze data?** Use `/scripts/example_strain_heterogeneity_workflow.py` as template
3. **Developing the plugin?** See `src/persiste/plugins/genecontent/` and plugin README
4. **Looking for benchmarks?** See `docs/benchmarks/`
5. **Understanding history?** See `docs/development/` and `exploratory/` subdirectories
