"""
Plot scaling curves from assembly_scaling_results.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('assembly_scaling_results.json', 'r') as f:
    data = json.load(f)

results = data['results']
minimal = data['minimal_requirements']

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Scaling Curves: Minimal Data Requirements for Inference', fontsize=14, fontweight='bold')

# ============================================================================
# Plot 1: Number of Primitives
# ============================================================================
ax = axes[0, 0]
prim_data = results['primitives']
x = [r['n_primitives'] for r in prim_data]
y_range = [r['ll_range'] for r in prim_data]
y_runtime = [r['runtime'] for r in prim_data]

ax.plot(x, y_range, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='LL Range')
ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Identifiable Threshold')
ax.set_xlabel('Number of Primitives', fontsize=11)
ax.set_ylabel('Log-Likelihood Range', fontsize=11, color='#2E86AB')
ax.tick_params(axis='y', labelcolor='#2E86AB')
ax.grid(True, alpha=0.3)
ax.set_title('(A) Number of Primitives', fontsize=12, fontweight='bold')

# Add runtime on secondary axis
ax2 = ax.twinx()
ax2.plot(x, y_runtime, 's--', linewidth=1.5, markersize=6, color='#A23B72', alpha=0.7, label='Runtime')
ax2.set_ylabel('Runtime (s)', fontsize=11, color='#A23B72')
ax2.tick_params(axis='y', labelcolor='#A23B72')

# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

# ============================================================================
# Plot 2: Max Depth
# ============================================================================
ax = axes[0, 1]
depth_data = results['depth']
x = [r['max_depth'] for r in depth_data]
y_range = [r['ll_range'] for r in depth_data]
y_runtime = [r['runtime'] for r in depth_data]

ax.plot(x, y_range, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='LL Range')
ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Identifiable Threshold')
ax.set_xlabel('Max Depth', fontsize=11)
ax.set_ylabel('Log-Likelihood Range', fontsize=11, color='#2E86AB')
ax.tick_params(axis='y', labelcolor='#2E86AB')
ax.grid(True, alpha=0.3)
ax.set_title('(B) Max Depth', fontsize=12, fontweight='bold')

ax2 = ax.twinx()
ax2.plot(x, y_runtime, 's--', linewidth=1.5, markersize=6, color='#A23B72', alpha=0.7, label='Runtime')
ax2.set_ylabel('Runtime (s)', fontsize=11, color='#A23B72')
ax2.tick_params(axis='y', labelcolor='#A23B72')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

# ============================================================================
# Plot 3: Sample Size
# ============================================================================
ax = axes[1, 0]
sample_data = results['samples']
x = [r['n_samples'] for r in sample_data]
y_range = [r['ll_range'] for r in sample_data]
y_runtime = [r['runtime'] for r in sample_data]

ax.plot(x, y_range, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='LL Range')
ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Identifiable Threshold')
ax.set_xlabel('Sample Size', fontsize=11)
ax.set_ylabel('Log-Likelihood Range', fontsize=11, color='#2E86AB')
ax.tick_params(axis='y', labelcolor='#2E86AB')
ax.grid(True, alpha=0.3)
ax.set_title('(C) Sample Size', fontsize=12, fontweight='bold')

ax2 = ax.twinx()
ax2.plot(x, y_runtime, 's--', linewidth=1.5, markersize=6, color='#A23B72', alpha=0.7, label='Runtime')
ax2.set_ylabel('Runtime (s)', fontsize=11, color='#A23B72')
ax2.tick_params(axis='y', labelcolor='#A23B72')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

# ============================================================================
# Plot 4: Simulation Time
# ============================================================================
ax = axes[1, 1]
time_data = results['time']
x = [r['t_max'] for r in time_data]
y_range = [r['ll_range'] for r in time_data]
y_runtime = [r['runtime'] for r in time_data]

ax.plot(x, y_range, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='LL Range')
ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Identifiable Threshold')
ax.set_xlabel('Simulation Time', fontsize=11)
ax.set_ylabel('Log-Likelihood Range', fontsize=11, color='#2E86AB')
ax.tick_params(axis='y', labelcolor='#2E86AB')
ax.grid(True, alpha=0.3)
ax.set_title('(D) Simulation Time', fontsize=12, fontweight='bold')

ax2 = ax.twinx()
ax2.plot(x, y_runtime, 's--', linewidth=1.5, markersize=6, color='#A23B72', alpha=0.7, label='Runtime')
ax2.set_ylabel('Runtime (s)', fontsize=11, color='#A23B72')
ax2.tick_params(axis='y', labelcolor='#A23B72')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

# ============================================================================
# Layout and save
# ============================================================================
plt.tight_layout()
plt.savefig('assembly_scaling_curves.png', dpi=300, bbox_inches='tight')
print("Saved: assembly_scaling_curves.png")

# Also create a summary text box figure
fig2, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

summary_text = f"""
MINIMAL DATA REQUIREMENTS FOR INFERENCE

Recommended Configuration:
  • Primitives:  ≥ {minimal['min_primitives']}
  • Max Depth:   ≥ {minimal['min_depth']}
  • Samples:     ≥ {minimal['min_samples']}
  • Sim Time:    ≥ {minimal['min_time']:.0f}

Key Findings:
  1. Sample size has strongest effect on identifiability
     (range: 29 → 167 as samples increase 20 → 200)
  
  2. Depth increases identifiability but is computationally expensive
     (range: 38 → 119 as depth increases 3 → 7)
  
  3. More primitives help moderately
     (range: 49 → 88 as primitives increase 3 → 7)
  
  4. Simulation time shows diminishing returns after t=20
     (range peaks at 111 for t=20, then plateaus)

Practical Recommendations:
  1. Start with 5 primitives, depth 5, 80 samples, t=50
  2. If not identifiable, increase samples first (linear cost)
  3. Then increase primitives (moderate cost)
  4. Depth is expensive (exponential state space)

Computational Cost:
  • Baseline (5 prim, depth 5, 80 samples): ~7 seconds
  • Doubling samples (160): ~8 seconds (+14%)
  • Increasing depth to 7: ~8.5 seconds (+21%)
  • Increasing primitives to 7: ~9 seconds (+29%)
"""

ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, 
        fontsize=11, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('assembly_scaling_summary.png', dpi=300, bbox_inches='tight')
print("Saved: assembly_scaling_summary.png")

print("\nPlots created successfully!")
