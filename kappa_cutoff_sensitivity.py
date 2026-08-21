"""
kappa_cutoff_sensitivity.py
============================
Tests the κ_B volume-integration result against different
inner-boundary cutoffs: 0.5%, 1% (current), 2%, 5% of R_star.
Also tests outer boundary cuts.

If <kappa_B>_V is stable across cuts, the "negligible" claim is verified.
"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from scipy.interpolate import CubicSpline
from constants import G, c
from eos import EOS
from tov_solver import TOVSolver

Msun = 1.989e33; km = 1e5
ALPHA = -3.0e12; KAPPA = 0.15; RHO_C = 1.0e10

eos = EOS(mode='chandra', B_0=3.79e14, magnetic_tov=True, sigma=20, N_points=2000)
res = TOVSolver(eos, alpha=ALPHA, kappa=KAPPA, compute_tidal=False).solve(RHO_C)
r_raw = np.array(res['r_profile']); P_raw = np.array(res['P_profile'])
M_raw = np.array(res['M_profile']); R_star = r_raw[-1]
cs_P = CubicSpline(r_raw, P_raw); cs_M = CubicSpline(r_raw, M_raw)

# Build full grid from 0.001*R to 0.999*R
r_full = np.linspace(0.001*R_star, 0.999*R_star, 10000)
kB_full, r_valid = [], []
for r in r_full:
    P = float(cs_P(r))
    if P <= 0: continue
    M_enc = float(cs_M(r))
    rho, eps = eos.get_rho_eps(P)
    B = eos.get_B(rho)
    rs = 2*G*M_enc/c**2
    denom_r = r*(r - rs)
    if denom_r <= 0: continue
    dPhi_dr = G*(M_enc + 4*np.pi*r**3*P/c**2)/denom_r
    Pr_plus_eps = abs(P) + eps
    if dPhi_dr <= 0 or Pr_plus_eps <= 0: continue
    kB = 3*B**2*c**2/(4*np.pi*Pr_plus_eps*dPhi_dr*r)
    kB_full.append(kB); r_valid.append(r)

r_valid = np.array(r_valid); kB_full = np.array(kB_full)

print("κ_B volume-integration cutoff sensitivity test")
print("Extreme-field config (B0=3.79e14 G, alpha=-3e12, kappa=0.15)")
print(f"R_star = {R_star/km:.1f} km")
print()
print(f"{'Inner cut':>12}  {'Outer cut':>10}  {'<kB>_V':>8}  {'vol excluded':>14}  {'note'}")
print("-"*70)

for r_lo_frac, r_hi_frac in [
    (0.005, 0.995),  # 0.5% / 0.5%
    (0.010, 0.990),  # 1%   / 1%   (published)
    (0.020, 0.980),  # 2%   / 2%
    (0.050, 0.950),  # 5%   / 5%
    (0.001, 0.999),  # tightest (reference)
    (0.010, 0.999),  # inner 1%, outer tight
    (0.001, 0.990),  # inner tight, outer 1%
]:
    mask = (r_valid >= r_lo_frac*R_star) & (r_valid <= r_hi_frac*R_star)
    r_s = r_valid[mask]; kB_s = kB_full[mask]
    if len(r_s) < 10: continue
    r2 = r_s**2
    kB_vol = np.trapezoid(kB_s*r2, r_s) / np.trapezoid(r2, r_s)
    vol_excl = (r_lo_frac**3 + (1-r_hi_frac)**3) * 100
    note = "(published)" if (r_lo_frac==0.01 and r_hi_frac==0.99) else ""
    print(f"  {r_lo_frac*100:>5.1f}% / {r_hi_frac*100:>5.1f}%  "
          f"  {kB_vol:>8.4f}  {vol_excl:>12.4f}%  {note}")

print()
print("Conclusion: if <kB>_V is stable across cutoffs, the 'negligible' claim holds.")
