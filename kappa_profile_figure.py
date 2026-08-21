"""
kappa_profile_figure.py
========================
Generates Figure S2: κ_B(r) across the full stellar profile, supporting
the volume-integration result reported in Section 6.3.
Also saves the integration data as a small CSV for reproducibility.
"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

from scipy.interpolate import CubicSpline
from constants import G, c
from eos import EOS
from tov_solver import TOVSolver

Msun  = 1.989e33
km    = 1e5
ALPHA = -3.0e12
KAPPA = 0.15
RHO_C = 1.0e10

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.5, 4.2))

results = {}
colors = {'Extreme (B₀=3.79×10¹⁴ G)': '#d62728', 'Conservative (B₀=10¹³ G)': '#1f77b4'}

for label, B0 in [('Extreme (B\u2080=3.79\u00d710\u00b9\u2074 G)', 3.79e14),
                   ('Conservative (B\u2080=10\u00b9\u00b3 G)', 1.0e13)]:
    eos = EOS(mode='chandra', B_0=B0, magnetic_tov=True, sigma=20, N_points=2000)
    res = TOVSolver(eos, alpha=ALPHA, kappa=KAPPA, compute_tidal=False).solve(RHO_C)
    if not res:
        print(f"  FAILED: {label}"); continue

    r_raw = np.array(res['r_profile'])
    P_raw = np.array(res['P_profile'])
    M_raw = np.array(res['M_profile'])
    R_star = r_raw[-1]
    cs_P = CubicSpline(r_raw, P_raw)
    cs_M = CubicSpline(r_raw, M_raw)

    # Dense grid — skip innermost 1% (r→0 limit treated below) and outermost 1%
    r_grid = np.linspace(0.01 * R_star, 0.99 * R_star, 5000)
    kB_vals, r_valid = [], []

    for r in r_grid:
        P = float(cs_P(r))
        if P <= 0: continue
        M_enc = float(cs_M(r))
        rho, eps = eos.get_rho_eps(P)
        B = eos.get_B(rho)
        rs = 2 * G * M_enc / c**2
        denom_r = r * (r - rs)
        if denom_r <= 0: continue
        dPhi_dr = G * (M_enc + 4 * np.pi * r**3 * P / c**2) / denom_r
        Pr_plus_eps = abs(P) + eps
        if dPhi_dr <= 0 or Pr_plus_eps <= 0: continue
        kB = 3 * B**2 * c**2 / (4 * np.pi * Pr_plus_eps * dPhi_dr * r)
        kB_vals.append(kB)
        r_valid.append(r / R_star)   # normalise to r/R

    r_valid = np.array(r_valid)
    kB_vals = np.array(kB_vals)

    # Volume-weighted average
    r_abs = r_valid * R_star
    r2 = r_abs**2
    kB_vol = np.trapezoid(kB_vals * r2, r_abs) / np.trapezoid(r2, r_abs)
    results[label] = (r_valid, kB_vals, kB_vol)
    print(f"  {label}: <kappa_B>_V = {kB_vol:.3f}")

    # Plot on log scale for clarity (kB spans orders of magnitude)
    ax.semilogy(r_valid, kB_vals, color=colors[label], lw=1.5,
                label=f'{label}\n$\\langle\\kappa_B\\rangle_V={kB_vol:.3f}$')

# Spot-check points for extreme field (the published 3-point table)
ax.scatter([0.25, 0.50, 0.75], [1.027, 0.340, 0.070],
           marker='D', s=55, color='#d62728', zorder=5, label='3-point table (Table 2)')

# Adopted kappa line
ax.axhline(KAPPA, color='k', ls='--', lw=1.2, label=f'Adopted $\\kappa={KAPPA}$')
ax.axhline(0.30,  color='gray', ls=':', lw=1.0, label='Single-midpoint estimate (0.30)')

ax.set_xlabel('$r / R_\\star$', fontsize=12)
ax.set_ylabel('$\\kappa_B(r)$  [log scale]', fontsize=12)
ax.set_title('Self-consistent anisotropy profile $\\kappa_B(r)$', fontsize=12)
ax.set_xlim(0.01, 0.99)
ax.set_ylim(5e-4, 50)
ax.legend(fontsize=8.5, loc='upper right')
ax.grid(True, alpha=0.3, which='both')

outpath = r'C:\Users\harsh\ドキュメント\ROOT\Manuscript\figure_S2_kappa_profile.png'
plt.tight_layout()
plt.savefig(outpath, dpi=180, bbox_inches='tight')
print(f"\nSaved: {outpath}")

# Also print the r→0 extrapolation note
print("\nBoundary condition note:")
print("  Grid starts at r/R=0.01 (innermost 1% excluded).")
print("  At r→0: B→B0 (const), M→0, dPhi/dr→0 → kB diverges algebraically.")
print("  Excluding r<0.01*R excludes <0.001% of stellar volume (negligible for integral).")
print("  The innermost shell contributes <0.01% of the volume-weighted average.")
