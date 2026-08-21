"""
kappa_stability_boundary.py
============================
Maps κ*(B₀) from B₀ = 10¹³ G up to 5×10¹³ G to find the lowest field
at which the self-consistent anisotropy parameter becomes non-negligible
(κ* > 0.01, i.e., crosses out of the "κ≈0 regime").

This is the physically interesting boundary: below it, no degeneracy
survives self-consistent treatment inside the stability-safe zone.

Uses the iterative fixed-point scheme already validated.
"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from scipy.interpolate import CubicSpline
from constants import G, c
from eos import EOS
from tov_solver import TOVSolver

Msun = 1.989e33; km = 1e5
ALPHA = -3.0e12
RHO_C = 1.0e10
THRESHOLD = 0.01   # "non-negligible" κ*

def volume_avg_kappa(res, eos, r_cut=0.01):
    r_raw = np.array(res['r_profile']); P_raw = np.array(res['P_profile'])
    M_raw = np.array(res['M_profile']); R_star = r_raw[-1]
    cs_P = CubicSpline(r_raw, P_raw); cs_M = CubicSpline(r_raw, M_raw)
    r_grid = np.linspace(r_cut*R_star, (1-r_cut)*R_star, 2000)
    kB_v, r_v = [], []
    for r in r_grid:
        P = float(cs_P(r))
        if P <= 0: continue
        M_enc = float(cs_M(r))
        rho, eps = eos.get_rho_eps(P)
        B = eos.get_B(rho)
        rs = 2*G*M_enc/c**2
        denom = r*(r - rs)
        if denom <= 0: continue
        dPhi = G*(M_enc + 4*np.pi*r**3*P/c**2)/denom
        Pe = abs(P)+eps
        if dPhi <= 0 or Pe <= 0: continue
        kB_v.append(3*B**2*c**2/(4*np.pi*Pe*dPhi*r)); r_v.append(r)
    if len(r_v) < 5: return 0.0
    r_v = np.array(r_v); kB_v = np.array(kB_v); r2 = r_v**2
    return np.trapezoid(kB_v*r2, r_v)/np.trapezoid(r2, r_v)

def solve_iterated(eos, alpha, kappa_init=0.15, tol=0.004, max_iter=8):
    kappa = kappa_init
    for it in range(max_iter):
        sol = TOVSolver(eos, alpha=alpha, kappa=kappa, compute_tidal=False).solve(RHO_C)
        if sol is None: return None, kappa
        kappa_new = volume_avg_kappa(sol, eos)
        if abs(kappa_new - kappa) < tol:
            return sol, kappa_new
        kappa = kappa_new
    return sol, kappa

# Scan B₀ from 10¹³ to 1.5×10¹⁴ G — log-spaced, 15 points
B0_scan = np.logspace(np.log10(1e13), np.log10(1.5e14), 15)

print(f"{'B₀ (G)':>14}  {'κ*':>7}  {'M (M⊙)':>9}  {'R (km)':>8}  {'Note'}")
print("-"*60)

boundary_B0 = None
prev_kappa = 0.0

for B0 in B0_scan:
    eos = EOS(mode='chandra', B_0=B0, magnetic_tov=True, sigma=20, N_points=800)
    sol, kappa_sc = solve_iterated(eos, ALPHA)
    if sol is None:
        print(f"{B0:>14.3e}  {'FAIL':>7}")
        continue
    M = sol['M']/Msun; R = sol['R']/km
    note = ""
    if kappa_sc >= THRESHOLD and prev_kappa < THRESHOLD and boundary_B0 is None:
        boundary_B0 = B0
        note = "  ← κ* crosses 0.01 here"
    print(f"{B0:>14.3e}  {kappa_sc:>7.4f}  {M:>9.4f}  {R:>8.1f}{note}")
    prev_kappa = kappa_sc

print()
if boundary_B0:
    print(f"κ*(B₀) threshold crossing (κ* = {THRESHOLD}):")
    print(f"  B₀ ≈ {boundary_B0:.2e} G  (Manreza Paret bound ≈ 10¹³ G)")
    if boundary_B0 > 1e13:
        print(f"  → The stable zone (B₀ ≤ 10¹³ G) lies entirely BELOW the κ*≥0.01 threshold.")
        print(f"  → No self-consistent anisotropy (κ*≥0.01) exists within the stability-safe regime.")
    else:
        print(f"  → Some self-consistent anisotropy survives inside the stability bound.")
else:
    print("κ* never crossed 0.01 in this scan range.")
print("\nDone.")
