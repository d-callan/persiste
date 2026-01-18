"""Debug script to trace likelihood evaluation with enriched observations."""

import json
from pathlib import Path
from persiste.plugins.assembly.cli import fit_assembly_constraints, InferenceMode

path = Path('src/persiste/plugins/assembly/validation/results/depth_gated/depth_gated_pr8_depth7_traj200_signal.json')
data = json.load(path.open())

print(f"Dataset loaded:")
print(f"  Observed compounds: {len(data.get('observed_compounds', []))}")
print(f"  Observation records: {len(data.get('observation_records', []))}")
print(f"  Has duration_summary: {bool(data.get('duration_summary'))}")

if data.get('observation_records'):
    rec = data['observation_records'][0]
    print(f"\nSample observation record:")
    print(f"  compound_id: {rec.get('compound_id')}")
    print(f"  mean_reuse_count: {rec.get('mean_reuse_count')}")
    print(f"  mean_max_depth: {rec.get('mean_max_depth')}")
    print(f"  frequency: {rec.get('frequency')}")

if data.get('duration_summary'):
    print(f"\nDuration summary:")
    print(f"  mean_reuse_count: {data['duration_summary'].get('mean_reuse_count')}")
    print(f"  variance_reuse_count: {data['duration_summary'].get('variance_reuse_count')}")

print("\n" + "="*60)
print("Running inference with screen_budget=10...")
print("="*60)

res = fit_assembly_constraints(
    observed_compounds=set(data['observed_compounds']),
    primitives=data['primitives'],
    mode=InferenceMode.SCREEN_AND_REFINE,
    feature_names=['reuse_count','depth_change'],
    seed=42,
    skip_safety_checks=True,
    observation_records=data.get('observation_records'),
    observation_summary=data.get('duration_summary'),
    max_depth=data.get('config', {}).get('max_depth'),
    screen_budget=10,
    **data.get('inference_baseline', {})
)

print(f"\nResults:")
print(f"  Screening results: {len(res.get('screening_results', []))}")
if res.get('screening_results'):
    print(f"\nTop 5 screening candidates:")
    for i, r in enumerate(res['screening_results'][:5]):
        print(f"    {i}: theta={r['theta']}, delta_ll={r['delta_ll']:.6f}, norm_delta_ll={r['normalized_delta_ll']:.6f}")
else:
    print("  No screening results!")

print(f"\n  Best theta: {res.get('theta_hat')}")
