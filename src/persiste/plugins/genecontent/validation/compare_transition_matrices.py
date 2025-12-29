#!/usr/bin/env python
"""Compare transition matrix computation methods."""

import numpy as np
from scipy.linalg import expm

gain_rate = 2.0
loss_rate = 3.0
t = 1.0

print("Comparing transition matrix computation:")
print("=" * 80)
print(f"Gain rate: {gain_rate}")
print(f"Loss rate: {loss_rate}")
print(f"Branch length: {t}")
print()

# Method 1: Matrix exponentiation (used in simulation)
Q = np.array([
    [-gain_rate, gain_rate],
    [loss_rate, -loss_rate]
])
P_expm = expm(Q * t)

print("Method 1: Matrix exponentiation")
print("-" * 80)
print("Rate matrix Q:")
print(Q)
print()
print("P(t) = exp(Q*t):")
print(P_expm)
print()

# Method 2: Closed-form (used in SimpleBinaryTransitionProvider)
λ = gain_rate
μ = loss_rate
total = λ + μ
exp_term = np.exp(-total * t)

p00 = (μ + λ * exp_term) / total
p01 = (λ - λ * exp_term) / total
p10 = (μ - μ * exp_term) / total
p11 = (λ + μ * exp_term) / total

P_closed = np.array([
    [p00, p01],
    [p10, p11]
])

print("Method 2: Closed-form solution")
print("-" * 80)
print("P(t) = closed-form:")
print(P_closed)
print()

# Compare
print("Comparison:")
print("-" * 80)
print("Difference:")
print(P_expm - P_closed)
print()
print(f"Max absolute difference: {np.abs(P_expm - P_closed).max():.10f}")
print()

if np.allclose(P_expm, P_closed, atol=1e-10):
    print("✓ Methods agree - no mismatch here!")
else:
    print("✗ Methods disagree - this is the bug!")
