"""Dot-product (adjoint) test of Project 1's Python forward_project / back_project pair.

Run from prior_projects/Filtered-Back-Projection/py/ :
    python ../../../docs/evidence/legacy_p1_adjoint_check.py
Recorded run: 2026-09-29, macOS, numpy 2.3.5, scipy 1.16.3, rng seed 0.
"""
import sys, numpy as np
sys.path.insert(0, "phantom"); sys.path.insert(0, "sinogram"); sys.path.insert(0, "reconstruction")
import matplotlib; matplotlib.use("Agg")
from forward_projection import forward_project
from back_projection import back_project
rng = np.random.default_rng(0)
for N, K in [(64, 30), (64, 90), (128, 60)]:
    ang = np.linspace(0, 180, K, endpoint=False)
    for t in range(3):
        x = rng.standard_normal((N, N)); y = rng.standard_normal((K, N))
        lhs = np.vdot(forward_project(x, ang), y)
        # back_project divides by K; undo that so we test A^T, not A^T/K
        rhs = np.vdot(x, back_project(y, ang, N) * K)
        print(f"N={N} K={K} trial={t}  <Ax,y>={lhs:+.4e}  <x,A^Ty>={rhs:+.4e}  rel.err={abs(lhs-rhs)/max(abs(lhs),abs(rhs)):.3e}")
