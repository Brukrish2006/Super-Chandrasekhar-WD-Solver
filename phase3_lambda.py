"""
phase3_lambda.py  —  Decoupled two-pass tidal deformability
===========================================================
Pass 1: solve background (P, M) with compute_tidal=False  → fast
Pass 2: integrate y(r) Riccati ODE using stored profile   → fast (decoupled)

Key unit-conversion notes (cgs ↔ geometrized G=c=1):
  ν'(r) = G*(m+4πr³p/c²) / (c²*r²*(1-β))   [cm⁻¹]   (dΦ_dim/dr / c²)
  Q_cgs = 4πG/c⁴ × e²λ × (5ε+9p+h_eff) - 6e²λ/r² - ν'²  [cm⁻²]
  F_cgs = (e²λ × (1 + 4πG/c⁴ × r²×(p-ε)) - 1)/r - r×ν'²  [cm⁻¹]
  h_eff = (ε+p) × (dε/dp)   [erg/cm³]   (relativistic enthalpy factor)

Previous bug: used G/c² instead of G/c⁴ → Q was c²=9×10²⁰ × too large.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from eos import EOS
from tov_solver import TOVSolver

G    = 6.67430e-8      # cm³ g⁻¹ s⁻²
c    = 2.99792458e10   # cm s⁻¹
Msun = 1.989e33
km   = 1e5

CONTOUR_POINTS = [
    (-1.00e13, 0.00, "A"),
    (-8.20e12, 0.00, "B"),
    (-6.40e12, 0.08, "C"),
    (-4.60e12, 0.08, "D"),
    (-2.80e12, 0.16, "E"),
    (-1.00e12, 0.16, "F"),
]
FIDUCIAL = [(-3.00e12, 0.15, "Fid")]
ALL_POINTS = CONTOUR_POINTS + FIDUCIAL

RHO_COARSE = np.logspace(8.5, 10.0, 20)
eos = EOS(mode='chandra', B_0=3.79e14, magnetic_tov=True)


def find_peak(eos, alpha, kappa):
    solver = TOVSolver(eos, alpha=alpha, kappa=kappa, compute_tidal=False)
    best_m, best_rho = 0.0, RHO_COARSE[5]
    for rc in RHO_COARSE:
        res = solver.solve(rc)
        if res and res['M']/Msun > best_m:
            best_m, best_rho = res['M']/Msun, rc
    fine = np.logspace(np.log10(best_rho)-0.4, np.log10(best_rho)+0.4, 20)
    for rc in fine:
        res = solver.solve(rc)
        if res and res['M']/Msun > best_m:
            best_m, best_rho = res['M']/Msun, rc
    return best_rho


def get_profile(eos, alpha, kappa, rho_c):
    solver = TOVSolver(eos, alpha=alpha, kappa=kappa, compute_tidal=False)
    return solver.solve(rho_c)


def compute_love(res, eos):
    """
    Two-pass tidal Love number computation.

    Uses the standard Hinderer (2008) Riccati ODE:
        r*y' = -(y² + y*F(r) + r²*Q(r))
    with F and Q evaluated from the pre-computed background profile.

    Correct cgs formulae (derived from G=c=1 by careful dimensional analysis):
        nu'  = G*(m + 4πr³p/c²) / (c²*r²*(1-β))           [cm⁻¹]
        F(r) = (e²λ*(1 + 4πG/c⁴*r²*(p-ε)) - 1)/r - r*nu'²  [cm⁻¹]
        Q(r) = 4πG/c⁴*e²λ*(5ε+9p+h) - 6e²λ/r² - nu'²       [cm⁻²]
        h    = (ε+p)*deps_dp                                  [erg/cm³]
    where deps_dp = dε/dp (dimensionless) from EOS spline.
    """
    r_arr = res['r_profile']
    P_arr = res['P_profile']
    M_arr = res['M_profile']
    R_s   = res['R']
    M_s   = res['M']

    # ── Sample on a clean, dense radial grid (avoid endpoints) ───────────────
    # Use 1% to 99% of the profile range; outside this range splines extrapolate
    r_min = 0.01 * R_s
    r_max = 0.99 * R_s

    # Build raw splines from the stored ODE solution (may be unevenly spaced)
    # Only use unique, sorted r points
    idx_sort = np.argsort(r_arr)
    r_raw = r_arr[idx_sort]
    P_raw = P_arr[idx_sort]
    M_raw = M_arr[idx_sort]

    # Remove duplicates
    _, uniq = np.unique(r_raw, return_index=True)
    r_raw = r_raw[uniq];  P_raw = P_raw[uniq];  M_raw = M_raw[uniq]

    # Filter to physical interior
    mask = (r_raw >= r_min) & (r_raw <= r_max) & (P_raw > 1e18)
    if mask.sum() < 10:
        print("    WARNING: fewer than 10 interior profile points — skipping")
        return None, None, None, None

    r  = r_raw[mask]
    P  = P_raw[mask]
    Mv = M_raw[mask]

    # ── EOS quantities at each radial point ──────────────────────────────────
    eps_arr  = np.empty_like(r)
    deps_arr = np.empty_like(r)   # dε/dP  (dimensionless)
    for i, p in enumerate(P):
        p_s = max(p, 1e21)
        _, eps_i        = eos.get_rho_eps(p_s)
        _, _, _, dep_i  = eos.get_R0_derivs(p_s)
        eps_arr[i]  = eps_i
        deps_arr[i] = dep_i if dep_i > 0 else 3.0  # safe fallback

    # ── Derived metric / gravitational quantities ─────────────────────────────
    beta  = 2.0*G*Mv / (c**2 * r)                      # Schwarzschild compactness β(r)
    beta  = np.clip(beta, 0.0, 0.99)
    e2lam = 1.0 / (1.0 - beta)                         # e^{2λ}

    # ν'(r) = G*(m + 4πr³p/c²) / (c²*r²*(1-β))         [cm⁻¹]
    nu_prime = G * (Mv + 4.0*np.pi*r**3*P/c**2) / (c**2 * r**2 * (1.0 - beta))

    # ── Relativistic enthalpy: h = (ε+p)*deps_dp        [erg/cm³] ───────────
    h_arr = (eps_arr + P) * deps_arr

    # ── Build CubicSpline interpolants ───────────────────────────────────────
    eps_sp  = CubicSpline(r, eps_arr,  extrapolate=False)
    P_sp    = CubicSpline(r, P,        extrapolate=False)
    e2l_sp  = CubicSpline(r, e2lam,   extrapolate=False)
    nu_sp   = CubicSpline(r, nu_prime, extrapolate=False)
    h_sp    = CubicSpline(r, h_arr,    extrapolate=False)

    def safe_eval(sp, rv, fallback=1.0):
        try:
            v = float(sp(rv))
            return v if np.isfinite(v) and v > -1e60 else fallback
        except Exception:
            return fallback

    # ── Hinderer (2008) F(r) and Q(r) in cgs ─────────────────────────────────
    def F_r(rv):
        e2l = safe_eval(e2l_sp, rv, 1.0)
        ep  = safe_eval(eps_sp,  rv, 1e29)
        p   = max(safe_eval(P_sp, rv, 1e29), 0.0)
        nu  = safe_eval(nu_sp, rv, 0.0)
        # F [cm⁻¹] — note G/c⁴ not G/c²
        return (e2l * (1.0 + 4.0*np.pi*G/c**4 * rv**2 * (p - ep)) - 1.0) / rv \
               - rv * nu**2

    def Q_r(rv):
        e2l = safe_eval(e2l_sp, rv, 1.0)
        ep  = max(safe_eval(eps_sp, rv, 1e29), 0.0)
        p   = max(safe_eval(P_sp,   rv, 1e29), 0.0)
        nu  = safe_eval(nu_sp, rv, 0.0)
        h   = max(safe_eval(h_sp,   rv, ep), 0.0)
        # Q [cm⁻²] — note G/c⁴, not G/c²
        return 4.0*np.pi*G/c**4 * e2l * (5.0*ep + 9.0*p + h) \
               - 6.0*e2l / rv**2 \
               - nu**2

    # ── Riccati ODE: dy/dr = -(y² + y*F + r²*Q) / r ─────────────────────────
    def riccati(rv, y_vec):
        y = float(y_vec[0])
        F = F_r(rv)
        Q = Q_r(rv)
        rhs = -(y**2 + y*F + rv**2 * Q) / rv
        if not np.isfinite(rhs):
            return [0.0]
        return [rhs]

    # Integrate from r_min to r_max
    r_start = float(r[0])
    r_end   = float(r[-1])

    # IC: y(r_start) = 2 + small correction from the leading-order ODE
    # Near center: y ≈ 2 + (2/r)*Δr since y'~2/r (regular solution dominates)
    # At r_start this approximation is fine (r_start = 1% R_star)
    y_ic = 2.0

    sol = solve_ivp(
        riccati,
        (r_start, r_end),
        [y_ic],
        method='DOP853',
        rtol=1e-7,
        atol=1e-4,
        max_step=(r_end - r_start) / 500,
        dense_output=False,
    )

    if not sol.success:
        # Try Radau as fallback (better for stiff/noisy problems)
        sol = solve_ivp(
            riccati,
            (r_start, r_end),
            [y_ic],
            method='Radau',
            rtol=1e-5,
            atol=1e-3,
            dense_output=False,
        )

    if not sol.success:
        print(f"    WARNING: Riccati ODE failed — status={sol.status} msg={sol.message}")
        return None, None, None, None

    y_R  = float(sol.y[0, -1])
    C_s  = G * M_s / (c**2 * R_s)

    # Sanity check on y_R: physically, y_R should be between 0 and 10 for WDs
    if not (0.0 < y_R < 20.0):
        print(f"    WARNING: y_R={y_R:.3f} is unphysical — Λ unreliable")

    # ── Love number k2 (Hinderer 2008 Eq. 23) ────────────────────────────────
    def love_k2(yR, C):
        two_C = 2.0 * C
        A = 2.0*C*(6.0 - 3.0*yR + 3.0*C*(5.0*yR - 8.0))
        B = 4.0*C**3*(13.0 - 11.0*yR + C*(3.0*yR - 2.0) + 2.0*C**2*(1.0+yR))
        D = 3.0*(1.0-two_C)**2*(2.0 - yR + 2.0*C*(yR-1.0))*np.log(1.0-two_C)
        num = (8.0*C**5/5.0)*(1.0-two_C)**2*(2.0 + 2.0*C*(yR-1.0) - yR)
        den = A + B + D
        if abs(den) < 1e-30:
            return 0.0
        k2 = num / den
        return max(k2, 0.0)  # k2 ≥ 0 physically

    k2  = love_k2(y_R, C_s)
    Lam = (2.0/3.0) * k2 / C_s**5 if C_s > 0 else 0.0

    return y_R, k2, Lam, C_s


# ─── Main ─────────────────────────────────────────────────────────────────────
print("Phase 3 LAMBDA (decoupled, corrected units)")
print("=" * 60)
print()

results = []
for alpha, kappa, label in ALL_POINTS:
    print(f"[{label}]  alpha={alpha:.2e}  kappa={kappa:.2f}")
    rho_peak = find_peak(eos, alpha, kappa)
    res      = get_profile(eos, alpha, kappa, rho_peak)
    if res is None:
        print("    FAILED: background solve returned None"); continue

    M_sol = res['M']/Msun; R_km = res['R']/km
    print(f"    Background: M={M_sol:.4f} Msun  R={R_km:.3f} km  rho_c={rho_peak:.3e}")

    y_R, k2, Lam, C_s = compute_love(res, eos)
    if y_R is None:
        print("    FAILED: tidal ODE failed"); continue

    print(f"    Tidal:      C={C_s:.6f}  y_R={y_R:.4f}  k2={k2:.5f}  Lambda={Lam:.4e}")
    results.append({'label':label,'alpha':alpha,'kappa':kappa,
                    'M':M_sol,'R':R_km,'C':C_s,'y_R':y_R,'k2':k2,'Lambda':Lam})
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
contour = [r for r in results if r['label'] != 'Fid']
fid     = next((r for r in results if r['label'] == 'Fid'), None)

if len(contour) >= 2:
    M_a   = np.array([r['M']      for r in contour])
    R_a   = np.array([r['R']      for r in contour])
    L_a   = np.array([r['Lambda'] for r in contour])
    C_a   = np.array([r['C']      for r in contour])
    k2_a  = np.array([r['k2']     for r in contour])

    dM  = 100*(M_a.max()-M_a.min())/M_a.mean()
    dR  = 100*(R_a.max()-R_a.min())/R_a.mean()
    dL  = 100*(L_a.max()-L_a.min())/L_a.mean() if L_a.mean()>0 else 0
    dC  = 100*(C_a.max()-C_a.min())/C_a.mean()

    print("─── Table 2: degeneracy-breaking summary ───")
    print(f"{'Lbl':>4} {'alpha':>14} {'kappa':>6} {'M':>8} {'R(km)':>9} "
          f"{'C':>8} {'k2':>7} {'Lambda':>12}")
    print("-"*72)
    for r in contour:
        print(f"  {r['label']:>3}  {r['alpha']:>14.3e}  {r['kappa']:>6.2f}  "
              f"{r['M']:>8.4f}  {r['R']:>9.3f}  {r['C']:>8.5f}  "
              f"{r['k2']:>7.4f}  {r['Lambda']:>12.4e}")
    if fid:
        print("-"*72)
        print(f"  Fid  {fid['alpha']:>14.3e}  {fid['kappa']:>6.2f}  "
              f"{fid['M']:>8.4f}  {fid['R']:>9.3f}  {fid['C']:>8.5f}  "
              f"{fid['k2']:>7.4f}  {fid['Lambda']:>12.4e}")
    print()
    print(f"  DeltaM/M  = {dM:.2f}%  (iso-contour tolerance)")
    print(f"  DeltaR/R  = {dR:.2f}%  ({R_a.min():.1f} - {R_a.max():.1f} km)")
    print(f"  DeltaC/C  = {dC:.2f}%")
    print(f"  DeltaL/L  = {dL:.2f}%  ({L_a.min():.3e} - {L_a.max():.3e})")
    print(f"  Lambda max/min ratio = {L_a.max()/max(L_a.min(),1e-30):.2f}x")

    # save table
    with open('table_phase3_lambda.txt','w') as f:
        f.write("# Table 2 — iso-Mmax contour tidal deformability\n")
        f.write("# k2 uses GR-limit Hinderer (2008) on the modified background profile.\n")
        f.write("# f(R) and kappa corrections to k2 are O(alpha,kappa) — advisor review needed.\n\n")
        for r in contour + ([fid] if fid else []):
            f.write(f"{r['label']:>4}  {r['alpha']:>14.4e}  {r['kappa']:>6.3f}  "
                    f"{r['M']:>9.5f}  {r['R']:>9.4f}  {r['C']:>9.6f}  "
                    f"{r['k2']:>8.5f}  {r['Lambda']:>14.5e}\n")
        f.write(f"\n# DeltaR/R  = {dR:.3f}%\n")
        f.write(f"# DeltaL/L  = {dL:.3f}%\n")
    print("\nSaved table_phase3_lambda.txt")

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(14,5))
    fig.suptitle(r'Iso-$M_{\max}$ contour: degeneracy broken by $R$ and $\Lambda$'
                 f'\nDeltaR/R = {dR:.1f}%    DeltaLambda/Lambda = {dL:.1f}%',
                 fontsize=12, fontweight='bold')
    labels = [r['label'] for r in contour]
    for ax, arr, title, cmap in [
        (axes[0], M_a,  r'$M_{\max}\ (M_\odot)$', 'Blues'),
        (axes[1], R_a,  r'$R\ \rm(km)$',           'viridis'),
        (axes[2], L_a,  r'$\Lambda$',               'inferno'),
    ]:
        cols = plt.get_cmap(cmap)(np.linspace(0.3,0.9,len(labels)))
        ax.bar(labels, arr, color=cols, edgecolor='k', width=0.6)
        ax.axhline(arr.mean(), color='red', ls='--', lw=1.5, label='Mean')
        ax.set_ylabel(title, fontsize=11); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        if 'Lambda' in title and arr.min() > 0:
            ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('figure_phase3_lambda.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved figure_phase3_lambda.png")

print("\nDone.")
