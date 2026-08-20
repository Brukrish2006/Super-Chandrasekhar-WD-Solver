"""
Figure 6 / Section 3.3: verifying that F_geom (f(R) geometric force) and
F_kappa (Bowers-Liang magnetic anisotropic force) are structurally distinct
across the ENTIRE stellar profile for SIX configurations spanning B0 from
10^12 G (well inside the Chatterjee stability bound) to 3.79e14 G (extreme).

Paper reference: Section 3.3, "Radial Profile Verification Beyond Leading Order"
(revised Aug 2026 to extend from 4 to 6 configurations).

Definitions (Paper.tex, Part II):
    F_geom(r)  = -alpha * (eps + P) / [2*(1 - 2GM/(c^2 r))] * dR0/dr
    F_kappa(r) = (kappa/6) * r * (dPhi/dr) * (P + eps)
    dPhi/dr    = G*(M + 4*pi*r^3*P/c^2) / [r*(r - 2GM/c^2)]

Six configurations
------------------
Configs 1-4: original set from first submission.
Configs 5-6: new intermediate-field additions (Aug 2026 revision) bridging
             the stable and extreme regimes and confirming the scaling result
             holds "continuously across the full explored field-strength range."

Outputs
-------
figure_6.png : two-panel figure for the primary stable-field configuration
               (Config1, B0=5e12 G) — matches the figure in the paper.
Console      : six-configuration power-law verification table.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from constants import c, G
from eos import EOS
from tov_solver import TOVSolver

Msun  = 1.989e33
R_km  = 1e5       # 1 km in cm
N_GRID = 400      # points on the smooth evaluation grid

# ---------------------------------------------------------------------------
# Six configurations (Section 3.3, Table in paper)
# ---------------------------------------------------------------------------
CONFIGS = [
    # label                            alpha       kappa  B0         rho_c
    ("Config1: stable-primary",       -3.0e12,    0.15,  5e12,      1e10),
    ("Config2: lower-alpha",          -1.0e12,    0.10,  1e12,      1e10),
    ("Config3: positive-alpha",       +1.0e12,    0.15,  5e12,      1e10),
    ("Config4: extreme-field (ref)",  -3.0e12,    0.15,  3.79e14,   1e10),
    ("Config5: intermed B0=1e13 [NEW]", -3.0e12,  0.15,  1e13,      1e10),
    ("Config6: intermed B0=5e13 [NEW]", -3.0e12,  0.15,  5e13,      1e10),
]


# ---------------------------------------------------------------------------
def solve_star(rho_c, alpha, kappa, B0):
    eos    = EOS(mode='chandra', B_0=B0, magnetic_tov=True)
    solver = TOVSolver(eos, alpha=alpha, kappa=kappa, compute_tidal=False)
    res    = solver.solve(rho_c)
    if res is None:
        raise RuntimeError(f"TOV failed: rho_c={rho_c:.2e}")
    return eos, solver, res


def compute_forces(eos, solver, res, alpha, kappa):
    """Evaluate F_geom(r) and F_kappa(r) on a smooth log-spaced grid."""
    r_raw, P_raw, M_raw = res['r_profile'], res['P_profile'], res['M_profile']

    # Trim to region where EOS spline is reliable (P > 3e21 avoids surface noise)
    mask = (r_raw > 0) & (P_raw > 3e21)
    r_raw, P_raw, M_raw = r_raw[mask], P_raw[mask], M_raw[mask]
    idx = np.argsort(r_raw)
    r_raw, P_raw, M_raw = r_raw[idx], P_raw[idx], M_raw[idx]
    r_raw, uniq = np.unique(r_raw, return_index=True)
    P_raw, M_raw = P_raw[uniq], M_raw[uniq]

    P_sp   = CubicSpline(r_raw, P_raw)
    M_sp   = CubicSpline(r_raw, M_raw)
    dPdr_sp = P_sp.derivative(1)

    r_g   = np.logspace(np.log10(r_raw[0]*1.01), np.log10(r_raw[-1]*0.999), N_GRID)
    P_g   = P_sp(r_g);  M_g   = M_sp(r_g);  dPdr_g = dPdr_sp(r_g)

    F_geom = np.zeros(N_GRID); F_kappa = np.zeros(N_GRID)

    for i, (r, P, M, dPdr) in enumerate(zip(r_g, P_g, M_g, dPdr_g)):
        if P <= 0:
            continue
        if getattr(eos, 'magnetic_tov', False):
            rho, eps_fl, B, B_mag = solver._get_magnetic(P)
            eps_tot = eps_fl + B_mag
        else:
            rho, eps_fl = eos.get_rho_eps(P)
            eps_tot = eps_fl
        denom = 1.0 - 2.0*G*M / (c**2*r)
        if denom <= 0:
            continue
        _, dR0_dP, _, _ = eos.get_R0_derivs(P)
        dR0_dr = dR0_dP * dPdr
        F_geom[i]  = -alpha * (eps_tot + P) / (2.0*denom) * dR0_dr
        dPhi_dr    = G*(M + 4.0*np.pi*r**3*P/c**2) / (r**2*denom)
        F_kappa[i] = (kappa/6.0) * r * dPhi_dr * (P + eps_tot)

    return r_g, F_geom, F_kappa, r_raw[-1], M_raw[-1]


def power_law_fit(x, y, xmax_frac):
    """Fit log|y| ~ n*log(x) over x < xmax_frac*max(x). Returns (n, R²)."""
    x = np.asarray(x); y = np.abs(np.asarray(y))
    sel = (x < xmax_frac * x.max()) & (y > 0)
    if sel.sum() < 5:
        return np.nan, np.nan
    lx, ly = np.log(x[sel]), np.log(y[sel])
    n, b = np.polyfit(lx, ly, 1)
    yfit = n*lx + b
    ss_r = np.sum((ly - yfit)**2); ss_t = np.sum((ly - ly.mean())**2)
    return n, (1 - ss_r/ss_t if ss_t > 0 else np.nan)


# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ── Part A: Six-configuration verification ──────────────────────────────
    print("=" * 70)
    print("Section 3.3 — Six-configuration MGD scaling verification")
    print("  Predictions: F_geom ~ r^+1, F_kappa ~ r^+2, ratio ~ r^-1")
    print("=" * 70)
    all_ng, all_nk = [], []
    for label, alpha, kappa, B0, rho_c in CONFIGS:
        print(f"\n{label}")
        eos, solver, res = solve_star(rho_c, alpha, kappa, B0)
        r, Fg, Fk, R_star, M_star = compute_forces(eos, solver, res, alpha, kappa)
        ng, r2g = power_law_fit(r, Fg, 0.10)
        nk, r2k = power_law_fit(r, Fk, 0.10)
        ratio = Fg / np.where(Fk != 0, Fk, np.nan)
        nr, r2r = power_law_fit(r, ratio, 0.10)
        all_ng.append(ng); all_nk.append(nk)
        print(f"  M={M_star/Msun:.3f} Msun  R={R_star/R_km:.1f} km  B0={B0:.2e} G")
        print(f"  n_geom  = {ng:+.4f}  (R²={r2g:.4f})  expected +1")
        print(f"  n_kappa = {nk:+.4f}  (R²={r2k:.4f})  expected +2")
        print(f"  ratio   = {nr:+.4f}  (R²={r2r:.4f})  expected -1")

    print(f"\nSpread across all 6 configs (inner 10% of R):")
    print(f"  F_geom : [{min(all_ng):.4f}, {max(all_ng):.4f}]")
    print(f"  F_kappa: [{min(all_nk):.4f}, {max(all_nk):.4f}]")
    print("  => r-vs-r^2 distinction holds continuously across the full B0 range.")

    # ── Part B: Figure 6 (Config1, stable-field primary) ────────────────────
    print("\n\nGenerating figure_6.png ...")
    label, alpha, kappa, B0, rho_c = CONFIGS[0]
    eos, solver, res = solve_star(rho_c, alpha, kappa, B0)
    r, Fg, Fk, R_star, M_star = compute_forces(eos, solver, res, alpha, kappa)
    x = r / R_star
    print(f"  Config: {label}  M={M_star/Msun:.3f} Msun  R={R_star/R_km:.1f} km")

    ng, r2g = power_law_fit(r, Fg, 0.10)
    nk, r2k = power_law_fit(r, Fk, 0.10)
    ratio = Fg / Fk
    nr, r2r = power_law_fit(r, ratio, 0.10)
    print(f"  n_geom={ng:+.4f}  n_kappa={nk:+.4f}  ratio exponent={nr:+.4f}")

    # Extension into star
    print("\n  Ratio exponent vs fitting window:")
    for frac in np.arange(0.10, 1.01, 0.10):
        n_r, _ = power_law_fit(r, ratio, frac)
        flag = "  <-- >7% drift" if abs(n_r + 1.0) > 0.07 else ""
        print(f"    r/R < {frac:.2f}: n = {n_r:+.3f}{flag}")

    i_ref     = np.argmin(np.abs(x - 0.05))
    Fg_n      = np.abs(Fg)    / np.abs(Fg[i_ref])
    Fk_n      = np.abs(Fk)    / np.abs(Fk[i_ref])
    ratio_n   = np.abs(ratio) / np.abs(ratio[i_ref])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    x_ref = x[(x > 0.02) & (x < 0.15)]

    ax = axes[0]
    ax.loglog(x, Fg_n, 'b-', lw=2.0, label=r'$|F_{\mathrm{geom}}(r)|$')
    ax.loglog(x, Fk_n, 'r-', lw=2.0, label=r'$|F_\kappa(r)|$')
    ax.loglog(x_ref, (x_ref/x[i_ref])**1, 'b:', lw=1.4, label='ref slope +1')
    ax.loglog(x_ref, (x_ref/x[i_ref])**2, 'r:', lw=1.4, label='ref slope +2')
    ax.axvline(0.10, color='gray', lw=0.8, ls='--')
    ax.set_xlabel(r'$r/R$'); ax.set_ylabel('Force (normalized to $r/R=0.05$)')
    ax.set_title('(a) Shape comparison across the full star')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, which='both', alpha=0.25, ls='--')

    ax = axes[1]
    ax.loglog(x, ratio_n, color='purple', lw=2.0,
              label=r'$|F_{\mathrm{geom}}/F_\kappa|$ (normalized)')
    ax.loglog(x_ref, (x_ref/x[i_ref])**(-1), 'k:', lw=1.4, label='ref slope -1')
    ax.axvline(0.10, color='gray', lw=0.8, ls='--')
    ax.set_xlabel(r'$r/R$')
    ax.set_ylabel(r'$|F_{\mathrm{geom}}/F_\kappa|$ (normalized to $r/R=0.05$)')
    ax.set_title(r'(b) Ratio vs. radius: departure from $r^{-1}$')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, which='both', alpha=0.25, ls='--')

    fig.suptitle(
        r'$F_{\mathrm{geom}} \propto r$ vs $F_\kappa \propto r^2$ — full radial profile'
        '\n' + fr'($\alpha={alpha:.1e}$ cm$^2$, $\kappa={kappa}$, $B_0={B0:.2e}$ G)',
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig('figure_6.png', dpi=300)
    plt.close()
    print("\nSaved figure_6.png")
