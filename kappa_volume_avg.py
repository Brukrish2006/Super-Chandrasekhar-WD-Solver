"""
kappa_volume_avg.py
===================
Computes the volume-averaged self-consistent kappa_B for both
the conservative companion (B0=1e13 G) and extreme-field (B0=3.79e14 G)
configurations, using the same formula as reproduce_new_results.py
but integrating over all radial shells instead of spot-checking 3 radii.

kappa_B(r) = 3 B^2(r) c^2 / (4 pi (P + eps) dPhi/dr r)

Volume-weighted average:
<kappa_B>_V = integral_0^R kappa_B(r) 4pi r^2 dr  /  (4pi R^3/3)
            = 3/R^3 * integral_0^R kappa_B(r) r^2 dr

Mass-weighted average also computed for comparison.
"""

import sys, numpy as np
from scipy.interpolate import CubicSpline
sys.stdout.reconfigure(encoding='utf-8')

from constants import G, c
from eos import EOS
from tov_solver import TOVSolver

Msun = 1.989e33
km   = 1e5
ALPHA = -3.0e12
KAPPA_FID = 0.15
RHO_C = 1.0e10

def compute_kappa_avg(B0, label):
    print(f"\n{'='*60}")
    print(f"Config: {label}  (B0={B0:.2e} G)")
    print('='*60)

    eos = EOS(mode='chandra', B_0=B0, magnetic_tov=True, sigma=20, N_points=2000)
    solver = TOVSolver(eos, alpha=ALPHA, kappa=KAPPA_FID, compute_tidal=False)
    res = solver.solve(RHO_C)

    if not res:
        print("  FAILED"); return

    r_raw = np.array(res['r_profile'])
    P_raw = np.array(res['P_profile'])
    M_raw = np.array(res['M_profile'])
    R_star = r_raw[-1]

    cs_P = CubicSpline(r_raw, P_raw)
    cs_M = CubicSpline(r_raw, M_raw)

    M_star = M_raw[-1]
    print(f"  M = {M_star/Msun:.4f} Msun   R = {R_star/km:.1f} km")

    # Dense radial grid (skip innermost 0.5% to avoid dPhi/dr singularity)
    r_grid = np.linspace(0.005 * R_star, 0.995 * R_star, 5000)
    kB_vals   = []
    r_valid   = []
    rho_vals  = []

    for r in r_grid:
        P = float(cs_P(r))
        if P <= 0:
            continue
        M_enc = float(cs_M(r))
        rho, eps = eos.get_rho_eps(P)
        B = eos.get_B(rho)
        rs = 2 * G * M_enc / c**2
        denom_r = r * (r - rs)
        if denom_r <= 0:
            continue
        dPhi_dr = G * (M_enc + 4 * np.pi * r**3 * P / c**2) / denom_r
        Pr_plus_eps = abs(P) + eps
        if dPhi_dr <= 0 or Pr_plus_eps <= 0:
            continue
        kB = 3 * B**2 * c**2 / (4 * np.pi * Pr_plus_eps * dPhi_dr * r)
        kB_vals.append(kB)
        r_valid.append(r)
        rho_vals.append(rho)

    r_valid  = np.array(r_valid)
    kB_vals  = np.array(kB_vals)
    rho_vals = np.array(rho_vals)

    # Volume-weighted average: integral(kB * r^2 dr) / integral(r^2 dr)
    r2 = r_valid**2
    kB_vol_avg = np.trapezoid(kB_vals * r2, r_valid) / np.trapezoid(r2, r_valid)

    # Mass-weighted average: integral(kB * rho * r^2 dr) / integral(rho * r^2 dr)
    kB_mass_avg = np.trapezoid(kB_vals * rho_vals * r2, r_valid) / np.trapezoid(rho_vals * r2, r_valid)

    # Simple arithmetic mean
    kB_mean = np.mean(kB_vals)

    # Report profile at spot-check fractions
    print(f"\n  Spot-check (reproducing 3-point table):")
    for frac in [0.25, 0.50, 0.75]:
        r = frac * R_star
        P = float(cs_P(r))
        M_enc = float(cs_M(r))
        rho, eps = eos.get_rho_eps(P)
        B = eos.get_B(rho)
        rs = 2 * G * M_enc / c**2
        dPhi_dr = G * (M_enc + 4 * np.pi * r**3 * P / c**2) / (r * (r - rs))
        Pr_plus_eps = abs(P) + eps
        kB = 3 * B**2 * c**2 / (4 * np.pi * Pr_plus_eps * dPhi_dr * r)
        print(f"    r/R={frac:.2f}:  kappa_B = {kB:.3f}")

    print(f"\n  Volume-weighted average  <kappa_B>_V  = {kB_vol_avg:.3f}")
    print(f"  Mass-weighted average    <kappa_B>_M  = {kB_mass_avg:.3f}")
    print(f"  Simple arithmetic mean   <kappa_B>_arith = {kB_mean:.3f}")
    print(f"\n  Adopted kappa = {KAPPA_FID}")
    diff_pct = 100*(kB_vol_avg - KAPPA_FID)/KAPPA_FID
    print(f"  Volume-avg vs adopted: {kB_vol_avg:.3f} vs {KAPPA_FID}  ({diff_pct:+.1f}%)")

    # Fractional volume breakdown
    print(f"\n  Volume fractions and mean kB by shell:")
    shells = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0)]
    total_vol = R_star**3 / 3
    for lo, hi in shells:
        mask = (r_valid >= lo*R_star) & (r_valid < hi*R_star)
        if mask.sum() == 0: continue
        shell_vol = (hi**3 - lo**3) * R_star**3 / 3
        vol_frac = shell_vol / total_vol
        kB_shell = np.mean(kB_vals[mask])
        contrib = vol_frac * kB_shell
        print(f"    r/R [{lo:.2f},{hi:.2f}]: vol={vol_frac:.3f}  mean kB={kB_shell:.3f}  contrib={contrib:.3f}")

    return kB_vol_avg

print("Full radial kappa_B(r) volume-integration")
print("Replaces the 3-point spot-check with a proper integral")

k_con = compute_kappa_avg(1.0e13, "Conservative companion (B0=1e13 G)")
k_ext = compute_kappa_avg(3.79e14, "Extreme field (B0=3.79e14 G)")

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"  Conservative companion:  <kappa_B>_V = {k_con:.3f}  (adopted kappa = {KAPPA_FID})")
print(f"  Extreme field:           <kappa_B>_V = {k_ext:.3f}  (adopted kappa = {KAPPA_FID})")
print(f"\nConclusion: volume-averaged self-consistent kappa for paper update.")
