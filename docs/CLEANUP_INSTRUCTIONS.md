# Robustness Experiments: Cleanup Instructions

**Date:** December 28, 2024

---

## Quick Summary

We attempted to build a robust inference framework with built-in diagnostics. The work didn't pan out due to performance issues and a fundamental caching flaw. All experimental code has been moved to `wip/robustness_experiments/` for future reference.

**Bottom line:** The original simple validation approach works perfectly (3s runtime, validated results). Use that.

---

## What to Do

### Option 1: Clean up now (recommended)

```bash
chmod +x cleanup_robustness_experiments.sh
./cleanup_robustness_experiments.sh
```

This will:
- Remove all experimental source files
- Remove experimental example scripts  
- Remove experimental documentation
- Keep the summary documents and WIP directory

### Option 2: Review first, clean later

1. Read `docs/ROBUSTNESS_WORK_SUMMARY.md` (comprehensive summary)
2. Review `wip/robustness_experiments/` (all experimental code preserved there)
3. Run cleanup script when ready

---

## What Gets Removed

### Source Files (experimental, not working)
- `src/persiste/plugins/assembly/baselines/baseline_family.py`
- `src/persiste/plugins/assembly/inference/constraint_result.py`
- `src/persiste/plugins/assembly/inference/robust_inference.py`
- `src/persiste/plugins/assembly/inference/robust_inference_v2.py`
- `src/persiste/plugins/assembly/inference/state_cache.py`

### Example Scripts (for non-working code)
- `examples/assembly_robust_example.py`
- `examples/test_fast_inference.py`
- `examples/validate_v2_quick.py`
- `examples/validate_v2_simple.py`
- `examples/validate_v2_vs_v1.py`
- `examples/profile_inference_bottleneck.py`

### Documentation (experimental/interim)
- `docs/ASSEMBLY_ARCHITECTURE_V2.md`
- `docs/ASSEMBLY_V2_SUMMARY.md`
- `docs/ASSEMBLY_V2_VALIDATION.md`
- `docs/ASSEMBLY_README.md`
- `docs/ASSEMBLY_PHASE1_*.md`
- `docs/ASSEMBLY_PLUGIN_SUMMARY.md`
- `docs/ASSEMBLY_IDENTIFIABILITY_SOLVED.md`

---

## What Gets Kept

### Working Code (validated baseline)
- ✓ `examples/assembly_validation_fixed.py` - **Use this!**
- ✓ `examples/assembly_scaling_curves.py`
- ✓ `examples/assembly_robustness_tests.py`
- ✓ All core assembly plugin code (states, baselines, constraints, etc.)

### Documentation (useful reference)
- ✓ `docs/ROBUSTNESS_WORK_SUMMARY.md` - **Read this first!**
- ✓ `docs/ASSEMBLY_USER_GUIDE.md` - Good conceptual overview
- ✓ `docs/ASSEMBLY_TECHNICAL.md` - Implementation details
- ✓ `docs/ASSEMBLY_ROBUSTNESS.md` - Robustness analysis results
- ✓ `docs/ASSEMBLY_SCALING_CURVES.md` - Validated scaling results

### WIP Directory (for future attempts)
- ✓ `wip/robustness_experiments/` - All experimental code preserved
- ✓ `wip/robustness_experiments/README.md` - What was tried and why it failed

---

## Key Learnings (TL;DR)

1. **Original validation works perfectly**
   - 3 seconds per evaluation
   - Δ LL = 9.38 (strong signal)
   - No false positives
   - Simple and maintainable

2. **V1 (robust framework) was too slow**
   - 5-10 minutes per inference
   - Bottleneck: repeated simulation in optimization loop
   - Good API design, but unusable performance

3. **V2 (caching) had fundamental flaw**
   - Can't cache states at θ=0 and optimize away from it
   - Needs importance sampling to work properly
   - Without caching, just as slow as V1

4. **Don't add complexity without clear benefit**
   - Simple approach works
   - Complex framework added problems
   - Lesson learned

---

## For Future Attempts

If we want to revisit robustness:

1. **Implement importance sampling first**
   - Reweight cached states: `w(s) = p(s|θ) / p(s|θ_ref)`
   - Theoretically sound
   - More complex but would actually work

2. **Or use deterministic approximation**
   - Mean-field ODE for screening
   - Stochastic simulation for validation
   - Two-stage approach

3. **Or accept the 3-second baseline**
   - Current approach works
   - Fast enough for most use cases
   - Simple and maintainable

---

## Git Status After Cleanup

After running the cleanup script, `git status` should show:

**Untracked files to keep:**
```
docs/ROBUSTNESS_WORK_SUMMARY.md
docs/ASSEMBLY_USER_GUIDE.md
docs/ASSEMBLY_TECHNICAL.md
docs/ASSEMBLY_ROBUSTNESS.md
docs/ASSEMBLY_SCALING_CURVES.md
wip/robustness_experiments/
CLEANUP_INSTRUCTIONS.md
cleanup_robustness_experiments.sh
```

**Everything else:** Should be clean (experimental files removed)

---

## Questions?

See `docs/ROBUSTNESS_WORK_SUMMARY.md` for comprehensive details on:
- What we tried and why
- Performance results
- Why caching failed
- What would fix it
- Recommendations

---

## Ready to Clean Up?

```bash
# Review the summary first
cat docs/ROBUSTNESS_WORK_SUMMARY.md

# When ready, run cleanup
chmod +x cleanup_robustness_experiments.sh
./cleanup_robustness_experiments.sh

# Verify
git status
```

All experimental code is preserved in `wip/robustness_experiments/` if needed later.
