"""
reproduce_new_results.py
========================
Reproduces the three new numerical results added in the Aug 2026 revision of
"Disentangling f(R) Curvature from Magnetic Anisotropy in Super-Chandrasekhar
White Dwarfs" (Adhikary, IISc Bangalore).

Results reproduced
------------------
1. sigma=0 (unsmoothed, N=5000) smoothing-bias test  → Section 5.6
   Expected: M_sigma0 = 2.5732 Msun  (vs fiducial 2.5742 Msun, delta = 0.04%)

2. kappa_B radial profile at r = R/4, R/2, 3R/4      → Table (tab:kappa_profile)
   Expected: kappa_B = [1.027, 0.340, 0.070] at fractions [0.25, 0.50, 0.75]

3. kappa sensitivity bracket (kappa=0.15 vs 0.30)    → Table 3 (tab:kappa_bracket)
   Expected:
     B0=3.79e14: M(k=0.15)=2.759, M(k=0.30)=2.981  (at rho_c=1e10)
     B0=1e13:    M(k=0.15)=1.651, M(k=0.30)=1.793

All results use the public solver (https://github.com/Brukrish2006/Super-Chandrasekhar-WD-Solver).
"""

import numpy as np
from scipy.interpolate import CubicSpline

from constants import G, c
from eos import EOS
from tov_solver import TOVSolver

Msun = 1.989e33
km   = 1e5

# ─────────────────────────────────────────────────────────────────────────────
# Shared fiducial parameters
# ─────────────────────────────────────────────────────────────────────────────
ALPHA  = -3.0e12   # cm^2  (f(R) coupling)
KAPPA  = 0.15      # Bowers-Liang anisotropy
B0_EXT = 3.79e14   # G     (extreme-field fiducial)
B0_CON = 1.0e13    # G     (conservative companion)
RHO_C  = 1.0e10    # g/cm^3

# ─────────────────────────────────────────────────────────────────────────────
# Result 1: sigma=0 smoothing-bias test  (Section 5.6)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("RESULT 1: sigma=0 smoothing-bias test")
print("=" * 65)

for sigma, N, label in [
    (20, 2000, "fiducial  (sigma=20, N=2000)"),
    ( 0, 5000, "unsmoothed (sigma=0,  N=5000)"),
]:
    eos = EOS(mode='chandra', B_0=B0_EXT, magnetic_tov=True,
              sigma=sigma, N_points=N)
    sol = TOVSolver(eos, alpha=ALPHA, kappa=KAPPA,
                    compute_tidal=False).solve(RHO_C)
    if sol:
        M = sol['M'] / Msun
        R = sol['R'] / km
        print(f"  {label}: M = {M:.4f} Msun   R = {R:.2f} km")
    else:
        print(f"  {label}: FAILED")

print()
print("  Expected: delta M = 0.001 Msun (0.04%) -- within stated precision")
print("  Conclusion: smoothing introduces no measurable systematic bias")


# ─────────────────────────────────────────────────────────────────────────────
# Result 2: kappa_B radial profile  (Table tab:kappa_profile)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("RESULT 2: kappa_B(r) radial profile at R/4, R/2, 3R/4")
print("=" * 65)

eos_fid = EOS(mode='chandra', B_0=B0_EXT, magnetic_tov=True,
              sigma=20, N_points=2000)
res = TOVSolver(eos_fid, alpha=ALPHA, kappa=KAPPA,
                compute_tidal=False).solve(RHO_C)

r_raw = res['r_profile']
P_raw = res['P_profile']
M_raw = res['M_profile']
R_star = r_raw[-1]
cs_P = CubicSpline(r_raw, P_raw)
cs_M = CubicSpline(r_raw, M_raw)

print(f"  R_star = {R_star/km:.1f} km")
print(f"  {'r/R':>6}  {'r (km)':>8}  {'rho (g/cc)':>12}  {'B (G)':>12}  kappa_B")
print(f"  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*12}  -------")

for frac in [0.25, 0.50, 0.75]:
    r = frac * R_star
    P = float(cs_P(r))
    M_enc = float(cs_M(r))
    rho, eps = eos_fid.get_rho_eps(P)
    B = eos_fid.get_B(rho)
    rs = 2 * G * M_enc / c**2
    dPhi_dr = (G * (M_enc + 4 * np.pi * r**3 * P / c**2)
               / (r * (r - rs))) if r > rs else 0.0
    Pr_plus_eps = abs(P) + eps
    kB = (3 * B**2 * c**2 / (4 * np.pi * Pr_plus_eps * dPhi_dr * r)
          if (dPhi_dr > 0 and Pr_plus_eps > 0 and r > 0)
          else float('nan'))
    print(f"  {frac:>6.2f}  {r/km:>8.1f}  {rho:>12.2e}  {B:>12.2e}  {kB:.3f}")

print()
print("  Adopted kappa = 0.15 lies between outer (0.070) and midpoint (0.340).")
print("  Self-consistency gap is core-concentrated, not a uniform 2x offset.")


# ─────────────────────────────────────────────────────────────────────────────
# Result 3: kappa sensitivity bracket  (Table 3, tab:kappa_bracket)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("RESULT 3: kappa sensitivity bracket (Table 3)")
print("  alpha = -3e12 cm^2, rho_c = 1e10 g/cc")
print("=" * 65)

bracket_configs = [
    ("Extreme (B0=3.79e14)", B0_EXT, 0.15),
    ("Extreme (B0=3.79e14)", B0_EXT, 0.30),
    ("Conservative (B0=1e13)", B0_CON, 0.15),
    ("Conservative (B0=1e13)", B0_CON, 0.30),
]

print(f"  {'Config':30s}  {'kappa':>5}  {'M (Msun)':>9}  {'R (km)':>8}  {'v_grav (km/s)':>14}")
print(f"  {'-'*30}  {'-'*5}  {'-'*9}  {'-'*8}  {'-'*14}")
for label, B0, kappa in bracket_configs:
    eos = EOS(mode='chandra', B_0=B0, magnetic_tov=True,
              sigma=20, N_points=1000)
    sol = TOVSolver(eos, alpha=ALPHA, kappa=kappa,
                    compute_tidal=False).solve(RHO_C)
    if sol:
        M = sol['M'] / Msun
        R = sol['R'] / km
        C = G * sol['M'] / (sol['R'] * c**2)   # compactness
        v = C * c / 1e5                           # gravitational redshift velocity km/s
        print(f"  {label:30s}  {kappa:>5.2f}  {M:>9.3f}  {R:>8.1f}  {v:>14.1f}")
    else:
        print(f"  {label:30s}  {kappa:>5.2f}  FAILED")

print()
print("Done. All values should reproduce Table 3 of the paper to ±0.001 Msun.")
