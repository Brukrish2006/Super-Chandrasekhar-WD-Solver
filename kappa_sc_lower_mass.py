"""
kappa_sc_lower_mass.py
=======================
Runs the self-consistent κ iso-mass contour at LOWER target masses
(1.6, 1.7, 1.8, 1.9 M⊙) to find where a genuine multi-configuration
self-consistent family exists.

At M=2.55 M⊙, only B₀=3.79e14 G could reach it self-consistently.
At lower masses, more B₀ values should be accessible.
Scans target masses to find the one with the MOST configs on contour.
"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
from constants import G, c
from eos import EOS
from tov_solver import TOVSolver

Msun = 1.989e33; km = 1e5

RHO_C = 1.0e10

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

def solve_iterated(eos, alpha, kappa_init=0.15, tol=0.004, max_iter=10):
    kappa = kappa_init
    for it in range(max_iter):
        sol = TOVSolver(eos, alpha=alpha, kappa=kappa, compute_tidal=False).solve(RHO_C)
        if sol is None: return None, kappa, it
        kappa_new = volume_avg_kappa(sol, eos)
        if abs(kappa_new - kappa) < tol:
            return sol, kappa_new, it+1
        kappa = kappa_new
    return sol, kappa, max_iter

def mass_residual(alpha, eos, M_tgt, kappa_init):
    sol, _, _ = solve_iterated(eos, alpha, kappa_init=kappa_init)
    if sol is None: return -99.0
    return (sol['M'] - M_tgt) / Msun

B0_list = [3.79e14, 3.00e14, 2.00e14, 1.50e14, 1.00e14, 5.00e13]
ALPHA_LO = -1.2e13
ALPHA_HI_vals = [0.0, -1e10, -1e11, -2e11]  # try progressively from 0 downward

target_masses = [1.60, 1.70, 1.80, 1.90]

best_result = {'n': 0, 'M_tgt': None, 'configs': []}

for M_tgt_sol in target_masses:
    M_tgt = M_tgt_sol * Msun
    print(f"\n{'='*65}")
    print(f"TARGET M = {M_tgt_sol:.2f} M⊙")
    print(f"{'='*65}")
    configs = []

    for B0 in B0_list:
        eos = EOS(mode='chandra', B_0=B0, magnetic_tov=True, sigma=20, N_points=800)
        # Get initial κ estimate
        sol0 = TOVSolver(eos, alpha=-3e12, kappa=0.15, compute_tidal=False).solve(RHO_C)
        kappa_init = volume_avg_kappa(sol0, eos) if sol0 else 0.10

        # Try to find bracket — extend ALPHA_HI toward 0 if needed
        m_lo = mass_residual(ALPHA_LO, eos, M_tgt, kappa_init)
        found_hi = None
        for a_hi in ALPHA_HI_vals:
            m_hi = mass_residual(a_hi if a_hi != 0 else -1e8, eos, M_tgt, kappa_init)
            if m_lo * m_hi < 0:
                found_hi = (a_hi if a_hi != 0 else -1e8, m_hi)
                break

        if found_hi is None:
            print(f"  B₀={B0:.2e}  κ_init={kappa_init:.3f}  [lo={m_lo:+.3f}]  → NO BRACKET")
            continue

        a_hi_val, m_hi = found_hi
        try:
            alpha_star = brentq(mass_residual, ALPHA_LO, a_hi_val,
                                args=(eos, M_tgt, kappa_init),
                                xtol=1e11, rtol=5e-4, maxiter=30)
        except ValueError as e:
            print(f"  B₀={B0:.2e}  brentq failed: {e}"); continue

        sol_sc, kappa_sc, n_it = solve_iterated(eos, alpha_star, kappa_init=kappa_init, tol=0.004)
        if sol_sc is None:
            print(f"  B₀={B0:.2e}  final solve failed"); continue

        M_sc = sol_sc['M']/Msun; R_sc = sol_sc['R']/km
        vg = G*sol_sc['M']/(sol_sc['R']*c**2)*c/1e5
        print(f"  B₀={B0:.2e}  α*={alpha_star:.3e}  κ*={kappa_sc:.4f}  "
              f"M={M_sc:.4f}  R={R_sc:.1f} km  vg={vg:.1f} km/s  ({n_it} iter)")
        configs.append({'B0': B0, 'alpha': alpha_star, 'kappa': kappa_sc,
                        'M': M_sc, 'R': R_sc, 'vg': vg})

    if len(configs) >= 2:
        vg_vals = [r['vg'] for r in configs]; R_vals = [r['R'] for r in configs]
        dvg = max(vg_vals)-min(vg_vals); dR_pct = 100*(max(R_vals)-min(R_vals))/min(R_vals)
        print(f"\n  ✓ {len(configs)} configs on contour | "
              f"Δv_grav = {dvg:.1f} km/s | ΔR/R = {dR_pct:.1f}%")
        if len(configs) > best_result['n']:
            best_result = {'n': len(configs), 'M_tgt': M_tgt_sol,
                           'dvg': dvg, 'dR': dR_pct, 'configs': configs}
    else:
        print(f"  ✗ Only {len(configs)} config(s) — no discriminator at this mass")

print(f"\n{'='*65}")
print("BEST SELF-CONSISTENT DISCRIMINATOR FOUND:")
if best_result['n'] >= 2:
    br = best_result
    print(f"  Target mass:  {br['M_tgt']:.2f} M⊙")
    print(f"  Configs:      {br['n']}")
    print(f"  Δv_grav:      {br['dvg']:.1f} km/s")
    print(f"  ΔR/R:         {br['dR']:.1f}%")
    print(f"  v_grav range: {min(r['vg'] for r in br['configs']):.1f}–"
          f"{max(r['vg'] for r in br['configs']):.1f} km/s")
    print(f"  R range:      {min(r['R'] for r in br['configs']):.1f}–"
          f"{max(r['R'] for r in br['configs']):.1f} km")
    for r in br['configs']:
        print(f"    B₀={r['B0']:.2e}  α*={r['alpha']:.3e}  κ*={r['kappa']:.4f}  "
              f"M={r['M']:.4f}  R={r['R']:.1f}  vg={r['vg']:.1f}")
else:
    print("  No target mass yielded ≥2 self-consistent configs.")
