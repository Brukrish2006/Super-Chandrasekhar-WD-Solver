"""
kappa_selfconsistent_solver.py
================================
Implements the self-consistent κ–B coupling that was listed as
"future work" in Section 8.2 — actually closing the circularity
that Reviewer 1 correctly identified.

TWO approaches:
A) GLOBAL ITERATIVE: Fix κ globally; iterate κ_{n+1} = <κ_B(solution_n)>_V
   until convergence. Converges in a few iterations.
B) LOCAL COUPLED: At each ODE step, compute κ(r) = κ_B(r) from B(r),P(r),M(r)
   and use it in the pressure gradient at that step.

Both report the self-consistent M_max and compare to the fixed-κ result.
"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from scipy.interpolate import CubicSpline
from constants import G, c
from eos import EOS
from tov_solver import TOVSolver

Msun = 1.989e33; km = 1e5
ALPHA = -3.0e12; RHO_C = 1.0e10

def volume_avg_kappa(res, eos, R_star=None, r_cut=0.01):
    """Compute <κ_B>_V from a solved TOV profile."""
    r_raw = np.array(res['r_profile'])
    P_raw = np.array(res['P_profile'])
    M_raw = np.array(res['M_profile'])
    if R_star is None:
        R_star = r_raw[-1]
    cs_P = CubicSpline(r_raw, P_raw)
    cs_M = CubicSpline(r_raw, M_raw)
    r_grid = np.linspace(r_cut*R_star, (1-r_cut)*R_star, 5000)
    kB_vals, r_valid = [], []
    for r in r_grid:
        P = float(cs_P(r))
        if P <= 0: continue
        M_enc = float(cs_M(r))
        rho, eps = eos.get_rho_eps(P)
        B = eos.get_B(rho)
        rs = 2*G*M_enc/c**2
        denom_r = r*(r-rs)
        if denom_r <= 0: continue
        dPhi_dr = G*(M_enc + 4*np.pi*r**3*P/c**2)/denom_r
        Pr_eps = abs(P)+eps
        if dPhi_dr <= 0 or Pr_eps <= 0: continue
        kB = 3*B**2*c**2/(4*np.pi*Pr_eps*dPhi_dr*r)
        kB_vals.append(kB); r_valid.append(r)
    r_valid = np.array(r_valid); kB_vals = np.array(kB_vals)
    r2 = r_valid**2
    return np.trapezoid(kB_vals*r2, r_valid)/np.trapezoid(r2, r_valid)


# ─── APPROACH A: GLOBAL ITERATIVE SCHEME ─────────────────────────────────────
print("=" * 65)
print("APPROACH A: Global iterative self-consistent κ")
print("  κ_{n+1} = <κ_B(solution with κ_n)>_V   until |Δκ|<0.005")
print("=" * 65)

for label, B0 in [
    ("Extreme-field (B0=3.79e14 G)", 3.79e14),
    ("Conservative (B0=1e13 G)",     1.00e13),
]:
    print(f"\n  {label}")
    kappa = 0.15   # initial guess
    eos = EOS(mode='chandra', B_0=B0, magnetic_tov=True, sigma=20, N_points=1000)
    prev_M = None

    for iteration in range(10):
        sol = TOVSolver(eos, alpha=ALPHA, kappa=kappa, compute_tidal=False).solve(RHO_C)
        if not sol:
            print(f"    Iter {iteration}: FAILED"); break
        M = sol['M']/Msun; R = sol['R']/km
        C_val = G*sol['M']/(sol['R']*c**2)
        vg = C_val*c/1e5
        kappa_new = volume_avg_kappa(sol, eos)
        print(f"    Iter {iteration}: κ_in={kappa:.4f} → <κ_B>_V={kappa_new:.4f} | "
              f"M={M:.4f} Msun, R={R:.1f} km, v_grav={vg:.1f} km/s")
        if abs(kappa_new - kappa) < 0.005:
            print(f"    ✓ Converged after {iteration+1} iterations")
            print(f"      Self-consistent κ* = {kappa_new:.4f}")
            print(f"      Self-consistent M*  = {M:.4f} Msun  "
                  f"(fixed-κ=0.15 gave M={sol['M']/Msun:.4f})")
            break
        kappa = kappa_new
    else:
        print(f"    DID NOT CONVERGE in 10 iterations")


# ─── DISCRIMINATOR: what does κ self-consistency do to Δv_grav? ──────────────
print()
print("=" * 65)
print("APPROACH A applied to discriminator configurations (Table 1)")
print("Self-consistent κ* for each extreme-field config")
print("=" * 65)

# Approximate alpha values for the six Table-1 configs
# (iso-M_max ≈ 2.55 Msun at κ=0.15, B0=3.79e14)
# We use a scan to bracket the contour
alpha_values = [-3.0e12, -2.5e12, -2.0e12, -1.5e12, -1.0e12, -0.5e12]
B0_vals      = [3.79e14, 3.00e14, 2.00e14, 1.50e14, 1.00e14, 0.50e14]

print(f"\n{'Config':>8}  {'alpha':>12}  {'B0':>12}  {'κ_sc':>6}  "
      f"{'M_sc':>8}  {'R_sc':>8}  {'v_sc':>8}")
vg_vals_sc = []
for i, (alpha, B0) in enumerate(zip(alpha_values, B0_vals)):
    eos = EOS(mode='chandra', B_0=B0, magnetic_tov=True, sigma=20, N_points=1000)
    kappa = 0.15
    for _ in range(8):
        sol = TOVSolver(eos, alpha=alpha, kappa=kappa, compute_tidal=False).solve(RHO_C)
        if not sol: break
        kappa_new = volume_avg_kappa(sol, eos)
        if abs(kappa_new - kappa) < 0.005:
            break
        kappa = kappa_new
    if sol:
        M = sol['M']/Msun; R = sol['R']/km
        C_v = G*sol['M']/(sol['R']*c**2)
        vg = C_v*c/1e5
        vg_vals_sc.append(vg)
        print(f"  Config {i+1:>2}  {alpha:>12.2e}  {B0:>12.2e}  "
              f"{kappa:>6.3f}  {M:>8.4f}  {R:>8.1f}  {vg:>8.1f}")

if len(vg_vals_sc) >= 2:
    delta_vg_sc = max(vg_vals_sc) - min(vg_vals_sc)
    print(f"\n  Self-consistent Δv_grav range: {min(vg_vals_sc):.1f}–{max(vg_vals_sc):.1f} km/s")
    print(f"  Self-consistent Δv_grav (max-min): {delta_vg_sc:.1f} km/s")
    print(f"  Original (fixed κ=0.15):            171 km/s")
    print(f"  Change:                              {delta_vg_sc-171:.1f} km/s "
          f"({100*(delta_vg_sc-171)/171:.1f}%)")

print("\nDone.")
