#!/usr/bin/env python3
"""
Inspect one-step sensitivity weights used by the CLEAN first SRC experiment.
No GPU/checkpoint required.
"""

import numpy as np

T = 64
rho = np.linspace(1.0, 0.0, T + 1)

rows = []
s2 = np.zeros(T + 1, dtype=np.float64)
for t in range(2, T + 1):
    s = t - 1
    rs, rt = rho[s], rho[t]
    S = (rs*rs - rt*rt) / (rs * (1.0 - rt*rt))
    s2[t] = S*S
    rows.append((t, s, S, S*S))

mean_s2 = s2[1:].mean()
scale = 1.0 / mean_s2

print("Primary matched sampler: 64 NFE, one-step t->t-1.")
print("Final 1->0 hard projection is excluded from SRC (weight 0).")
print(f"E_uniform_t[S^2] = {mean_s2:.10f}")
print(f"global normalization scale = {scale:.6f}")
print()
print("Selected weights:")
for t in [64, 48, 32, 16, 8, 4, 3, 2, 1]:
    if t == 1:
        print("  1->0: SRC excluded (hard final)")
    else:
        S = np.sqrt(s2[t])
        w = s2[t] * scale
        print(f"  {t:>2}->{t-1:<2}: S={S:.6f}, raw S^2={s2[t]:.8f}, normalized w={w:.4f}")
