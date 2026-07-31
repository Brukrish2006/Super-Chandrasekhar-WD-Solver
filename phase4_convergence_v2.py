"""
phase4_convergence_v2.py  —  Numerical convergence validation (decoupled)
==========================================================================
Uses the SAME two-pass decoupled approach as phase3_lambda.py:
  Pass 1: TOVSolver with compute_tidal=False  (fast, no LSODA issues)
  Pass 2: CubicSpline + Riccati ODE (DOP853)  (fast, never stiff)

Sweeps:
  4a. sigma  (EOS Gaussian smoothing): 5, 10, 20, 40
  4b. N_points (EOS table size):       300, 500, 1000, 2000
  4c. rtol    (ODE solver tolerance):  1e-4, 1e-5, 1e-6, 1e-7

Two points compared at each parameter combination:
  fid: (alpha=-3e12, kappa=0.15)  — fiducial
  alt: (alpha=-1e13, kappa=0.00)  — contour point A (max R contrast)

Convergence metric: |DeltaR| and |DeltaLambda| between fid and alt.
If these are stable across the parameter sweeps, the result is numerically robust.

Deliverables:
    table_phase4_convergence.txt
    figure_phase4_convergence.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from eos import EOS
from tov_solver import TOVSolver

G    = 6.67430e-8
c    = 2.99792458e10
Msun = 1.989e33
km   = 1e5

ALPHA_FID = -3.0e12;  KAPPA_FID = 0.15
ALPHA_ALT = -1.0e13;  KAPPA_ALT = 0.00
RHO_COARSE = np.logspace(8.5, 10.0, 15)


def find_peak(eos, alpha, kappa):
    """Fast non-tidal M_max density finder."""
    solver = TOVSolver(eos, alpha=alpha, kappa=kappa, compute_tidal=False)
    best_m, best_rho = 0.0, RHO_COARSE[5]
    for rc in RHO_COARSE:
        res = solver.solve(rc)
        if res and res['M']/Msun > best_m:
            best_m, best_rho = res['M']/Msun, rc
    fine = np.logspace(np.log10(best_rho)-0.4, np.log10(best_rho)+0.4, 15)
    for rc in fine:
        res = solver.solve(rc)
        if res and res['M']/Msun > best_m:
            best_m, best_rho = res['M']/Msun, rc
    return best_rho


def compute_love_decoupled(res, eos):
    """
    Decoupled Riccati solver — same as phase3_lambda.py.
    Returns (y_R, k2, Lambda, C). Returns (None,...) on failure.
    """
    r_arr = res['r_profile'];  P_arr = res['P_profile'];  M_arr = res['M_profile']
    R_s = res['R'];  M_s = res['M']

    r_min = 0.01 * R_s;  r_max = 0.99 * R_s
    idx   = np.argsort(r_arr)
    r_raw, P_raw, M_raw = r_arr[idx], P_arr[idx], M_arr[idx]
    _, uniq = np.unique(r_raw, return_index=True)
    r_raw = r_raw[uniq]; P_raw = P_raw[uniq]; M_raw = M_raw[uniq]

    mask = (r_raw >= r_min) & (r_raw <= r_max) & (P_raw > 1e18)
    if mask.sum() < 10:
        return None, None, None, None
    r = r_raw[mask]; P = P_raw[mask]; Mv = M_raw[mask]

    eps_arr  = np.empty_like(r)
    deps_arr = np.empty_like(r)
    for i, p in enumerate(P):
        p_s = max(p, 1e21)
        _, eps_i       = eos.get_rho_eps(p_s)
        _, _, _, dep_i = eos.get_R0_derivs(p_s)
        eps_arr[i]  = eps_i
        deps_arr[i] = dep_i if dep_i > 0 else 3.0

    beta    = np.clip(2.*G*Mv/(c**2*r), 0., 0.99)
    e2lam   = 1./(1.-beta)
    nu_p    = G*(Mv + 4.*np.pi*r**3*P/c**2)/(c**2*r**2*(1.-beta))
    h_arr   = (eps_arr + P)*deps_arr

    eps_sp  = CubicSpline(r, eps_arr,  extrapolate=False)
    P_sp    = CubicSpline(r, P,        extrapolate=False)
    e2l_sp  = CubicSpline(r, e2lam,   extrapolate=False)
    nu_sp   = CubicSpline(r, nu_p,    extrapolate=False)
    h_sp    = CubicSpline(r, h_arr,   extrapolate=False)

    def safe(sp, rv, fb=1.): 
        try: v=float(sp(rv)); return v if np.isfinite(v) else fb
        except: return fb

    def F_r(rv):
        return (safe(e2l_sp,rv,1.)*(1.+4.*np.pi*G/c**4*rv**2*
                (safe(P_sp,rv,1e29)-safe(eps_sp,rv,1e29)))-1.)/rv \
               - rv*safe(nu_sp,rv,0.)**2

    def Q_r(rv):
        e2l=safe(e2l_sp,rv,1.); ep=max(safe(eps_sp,rv,1e29),0.)
        p=max(safe(P_sp,rv,1e29),0.); nu=safe(nu_sp,rv,0.)
        h=max(safe(h_sp,rv,ep),0.)
        return 4.*np.pi*G/c**4*e2l*(5.*ep+9.*p+h)-6.*e2l/rv**2-nu**2

    def riccati(rv, y_vec):
        y=float(y_vec[0]); F=F_r(rv); Q=Q_r(rv)
        rhs=-(y**2+y*F+rv**2*Q)/rv
        return [rhs if np.isfinite(rhs) else 0.]

    sol = solve_ivp(riccati, (float(r[0]), float(r[-1])), [2.0],
                    method='DOP853', rtol=1e-6, atol=1e-4,
                    max_step=(float(r[-1])-float(r[0]))/400)
    if not sol.success:
        sol = solve_ivp(riccati, (float(r[0]), float(r[-1])), [2.0],
                        method='Radau', rtol=1e-5, atol=1e-3)
    if not sol.success:
        return None, None, None, None

    y_R = float(sol.y[0,-1])
    C_s = G*M_s/(c**2*R_s)

    def k2_func(yR, C):
        t=2.*C
        A=2.*C*(6.-3.*yR+3.*C*(5.*yR-8.))
        B=4.*C**3*(13.-11.*yR+C*(3.*yR-2.)+2.*C**2*(1.+yR))
        D=3.*(1.-t)**2*(2.-yR+2.*C*(yR-1.))*np.log(1.-t)
        num=(8.*C**5/5.)*(1.-t)**2*(2.+2.*C*(yR-1.)-yR)
        den=A+B+D
        return max(num/den, 0.) if abs(den)>1e-30 else 0.

    k2  = k2_func(y_R, C_s)
    Lam = (2./3.)*k2/C_s**5 if C_s>0 else 0.
    return y_R, k2, Lam, C_s


def run_point(eos, alpha, kappa):
    """Full two-pass solve for one (alpha, kappa) point."""
    rho_peak = find_peak(eos, alpha, kappa)
    solver   = TOVSolver(eos, alpha=alpha, kappa=kappa, compute_tidal=False)
    res      = solver.solve(rho_peak)
    if res is None:
        return None
    M_sol = res['M']/Msun;  R_km = res['R']/km
    y_R, k2, Lam, C_s = compute_love_decoupled(res, eos)
    if y_R is None:
        return {'M': M_sol, 'R': R_km, 'C': None, 'k2': None, 'Lambda': None}
    return {'M': M_sol, 'R': R_km, 'C': C_s, 'k2': k2, 'Lambda': Lam}


def run_pair(sigma, N_points):
    """Run fid + alt for given EOS params."""
    eos = EOS(mode='chandra', B_0=3.79e14, magnetic_tov=True,
              sigma=sigma, N_points=N_points)
    fid = run_point(eos, ALPHA_FID, KAPPA_FID)
    alt = run_point(eos, ALPHA_ALT, KAPPA_ALT)
    return fid, alt


def diff(fid, alt, key):
    try:
        return abs(fid[key] - alt[key])
    except (TypeError, KeyError):
        return float('nan')


# ─── Sigma sweep ──────────────────────────────────────────────────────────────
sigma_vals = [5, 10, 20, 40]
rows_s = []
print("Phase 4a: sigma sweep (N=1000)")
print(f"  {'sigma':>5}  {'Mfid':>7}  {'Rfid':>7}  {'Lfid':>12}  "
      f"{'Malt':>7}  {'Ralt':>7}  {'Lalt':>12}  {'dR':>6}  {'dL/L%':>7}")
for s in sigma_vals:
    fid, alt = run_pair(sigma=s, N_points=1000)
    dM = diff(fid, alt, 'M');  dR = diff(fid, alt, 'R')
    dL = diff(fid, alt, 'Lambda')
    Lm = (fid['Lambda']+alt['Lambda'])/2 if fid['Lambda'] and alt['Lambda'] else 1.
    dL_pct = 100*dL/Lm if Lm else float('nan')
    rows_s.append((s, fid, alt, dM, dR, dL, dL_pct))
    print(f"  {s:>5}  {fid['M']:>7.4f}  {fid['R']:>7.2f}  {fid['Lambda']:>12.4e}  "
          f"{alt['M']:>7.4f}  {alt['R']:>7.2f}  {alt['Lambda']:>12.4e}  "
          f"{dR:>6.2f}  {dL_pct:>7.2f}%")

# ─── N_points sweep ───────────────────────────────────────────────────────────
N_vals = [300, 500, 1000, 2000]
rows_n = []
print("\nPhase 4b: N_points sweep (sigma=20)")
print(f"  {'N':>5}  {'Mfid':>7}  {'Rfid':>7}  {'Lfid':>12}  "
      f"{'Malt':>7}  {'Ralt':>7}  {'Lalt':>12}  {'dR':>6}  {'dL/L%':>7}")
for n in N_vals:
    fid, alt = run_pair(sigma=20, N_points=n)
    dM = diff(fid, alt, 'M');  dR = diff(fid, alt, 'R')
    dL = diff(fid, alt, 'Lambda')
    Lm = (fid['Lambda']+alt['Lambda'])/2 if fid['Lambda'] and alt['Lambda'] else 1.
    dL_pct = 100*dL/Lm if Lm else float('nan')
    rows_n.append((n, fid, alt, dM, dR, dL, dL_pct))
    print(f"  {n:>5}  {fid['M']:>7.4f}  {fid['R']:>7.2f}  {fid['Lambda']:>12.4e}  "
          f"{alt['M']:>7.4f}  {alt['R']:>7.2f}  {alt['Lambda']:>12.4e}  "
          f"{dR:>6.2f}  {dL_pct:>7.2f}%")

# ─── rtol sweep ──────────────────────────────────────────────────────────────
rtol_vals = [1e-4, 1e-5, 1e-6, 1e-7]
rows_r = []
eos_ref = EOS(mode='chandra', B_0=3.79e14, magnetic_tov=True, sigma=20, N_points=1000)
print("\nPhase 4c: rtol sweep (sigma=20, N=1000, Riccati only)")
print(f"  {'rtol':>8}  {'Mfid':>7}  {'Rfid':>7}  {'Lfid':>12}  "
      f"{'Malt':>7}  {'Ralt':>7}  {'Lalt':>12}  {'dR':>6}  {'dL/L%':>7}")

# Pre-compute background profiles once (rtol only varies Riccati, not structure)
rho_fid = find_peak(eos_ref, ALPHA_FID, KAPPA_FID)
rho_alt = find_peak(eos_ref, ALPHA_ALT, KAPPA_ALT)
res_fid = TOVSolver(eos_ref, alpha=ALPHA_FID, kappa=KAPPA_FID,
                    compute_tidal=False).solve(rho_fid)
res_alt = TOVSolver(eos_ref, alpha=ALPHA_ALT, kappa=KAPPA_ALT,
                    compute_tidal=False).solve(rho_alt)

for rv in rtol_vals:
    row_pair = {}
    for label, res in [('fid', res_fid), ('alt', res_alt)]:
        if res is None:
            row_pair[label] = {'M': 0, 'R': 0, 'Lambda': None}; continue
        # Re-run only the Riccati ODE with different rtol
        r_arr = res['r_profile'];  P_arr = res['P_profile'];  M_arr = res['M_profile']
        R_s = res['R'];  M_s = res['M']
        r_min = 0.01*R_s;  r_max = 0.99*R_s
        idx   = np.argsort(r_arr)
        r_raw, P_raw, M_raw = r_arr[idx], P_arr[idx], M_arr[idx]
        _, u = np.unique(r_raw, return_index=True)
        r_r, P_r, M_r = r_raw[u], P_raw[u], M_raw[u]
        mask = (r_r>=r_min)&(r_r<=r_max)&(P_r>1e18)
        if mask.sum()<10:
            row_pair[label]={'M':M_s/Msun,'R':R_s/km,'Lambda':None}; continue
        r=r_r[mask]; P=P_r[mask]; Mv=M_r[mask]
        eps_arr=np.empty_like(r); deps_arr=np.empty_like(r)
        for i, p in enumerate(P):
            p_s=max(p,1e21)
            _,e=eos_ref.get_rho_eps(p_s); _,_,_,d=eos_ref.get_R0_derivs(p_s)
            eps_arr[i]=e; deps_arr[i]=d if d>0 else 3.
        beta=np.clip(2.*G*Mv/(c**2*r),0.,0.99)
        e2l=1./(1.-beta)
        nu_p=G*(Mv+4.*np.pi*r**3*P/c**2)/(c**2*r**2*(1.-beta))
        h_arr=(eps_arr+P)*deps_arr
        eps_sp=CubicSpline(r,eps_arr,extrapolate=False)
        P_sp=CubicSpline(r,P,extrapolate=False)
        e2l_sp=CubicSpline(r,e2l,extrapolate=False)
        nu_sp=CubicSpline(r,nu_p,extrapolate=False)
        h_sp=CubicSpline(r,h_arr,extrapolate=False)
        def safe(sp,rv,fb=1.):
            try: v=float(sp(rv)); return v if np.isfinite(v) else fb
            except: return fb
        def F_rv(rv):
            return (safe(e2l_sp,rv,1.)*(1.+4.*np.pi*G/c**4*rv**2*
                    (safe(P_sp,rv,1e29)-safe(eps_sp,rv,1e29)))-1.)/rv \
                   - rv*safe(nu_sp,rv,0.)**2
        def Q_rv(rv):
            el=safe(e2l_sp,rv,1.); ep=max(safe(eps_sp,rv,1e29),0.)
            pv=max(safe(P_sp,rv,1e29),0.); nu=safe(nu_sp,rv,0.)
            h=max(safe(h_sp,rv,ep),0.)
            return 4.*np.pi*G/c**4*el*(5.*ep+9.*pv+h)-6.*el/rv**2-nu**2
        def ric(rv,y):
            F=F_rv(rv); Q=Q_rv(rv); rhs=-(y[0]**2+y[0]*F+rv**2*Q)/rv
            return [rhs if np.isfinite(rhs) else 0.]
        sol=solve_ivp(ric,(float(r[0]),float(r[-1])),[2.],method='DOP853',
                      rtol=rv, atol=rv*100,
                      max_step=(float(r[-1])-float(r[0]))/400)
        if not sol.success:
            row_pair[label]={'M':M_s/Msun,'R':R_s/km,'Lambda':None}; continue
        y_R=float(sol.y[0,-1]); C_s=G*M_s/(c**2*R_s)
        t=2.*C_s
        A=2.*C_s*(6.-3.*y_R+3.*C_s*(5.*y_R-8.))
        B=4.*C_s**3*(13.-11.*y_R+C_s*(3.*y_R-2.)+2.*C_s**2*(1.+y_R))
        D=3.*(1.-t)**2*(2.-y_R+2.*C_s*(y_R-1.))*np.log(1.-t)
        num=(8.*C_s**5/5.)*(1.-t)**2*(2.+2.*C_s*(y_R-1.)-y_R)
        den=A+B+D
        k2=max(num/den,0.) if abs(den)>1e-30 else 0.
        Lam=(2./3.)*k2/C_s**5 if C_s>0 else 0.
        row_pair[label]={'M':M_s/Msun,'R':R_s/km,'Lambda':Lam}

    fid_=row_pair.get('fid',{}); alt_=row_pair.get('alt',{})
    dR=abs(fid_.get('R',0)-alt_.get('R',0))
    Lf=fid_.get('Lambda'); La=alt_.get('Lambda')
    dL=abs(Lf-La) if Lf and La else float('nan')
    Lm=(Lf+La)/2 if Lf and La else 1.
    dL_pct=100*dL/Lm if np.isfinite(dL) else float('nan')
    rows_r.append((rv,fid_,alt_,dR,dL,dL_pct))
    Lf_s=f"{Lf:.4e}" if Lf else "FAIL"
    La_s=f"{La:.4e}" if La else "FAIL"
    print(f"  {rv:>8.0e}  {fid_.get('M',0):>7.4f}  {fid_.get('R',0):>7.2f}  {Lf_s:>12}  "
          f"{alt_.get('M',0):>7.4f}  {alt_.get('R',0):>7.2f}  {La_s:>12}  "
          f"{dR:>6.2f}  {dL_pct:>7.2f}%")

# ─── Save table ───────────────────────────────────────────────────────────────
with open('table_phase4_convergence.txt', 'w') as f:
    f.write("# Phase 4 Convergence — decoupled Riccati approach\n")
    f.write("# fid=(alpha=-3e12, kappa=0.15)   alt=(alpha=-1e13, kappa=0.00)\n")
    f.write("# dR = |R_fid - R_alt| (km),  dL% = |Lambda_fid - Lambda_alt| / mean\n\n")
    for name, rows, xname in [
        ("Sigma sweep", rows_s, "sigma"),
        ("N_points sweep", rows_n, "N"),
        ("rtol sweep", rows_r, "rtol"),
    ]:
        f.write(f"# {name}\n")
        f.write(f"# {xname:>8}  Rfid    Ralt    dR(km)   Lfid        Lalt        dL%\n")
        for row in rows:
            p=row[0]; fid_=row[1]; alt_=row[2]
            Lf=fid_.get('Lambda','nan'); La=alt_.get('Lambda','nan')
            dR_v=row[3 if len(row)>5 else 3]
            dL_p=row[-1]
            f.write(f"  {p:>8g}  {fid_.get('R',0):>7.2f}  {alt_.get('R',0):>7.2f}  "
                    f"{row[-3]:>7.2f}  {str(Lf):>12}  {str(La):>12}  {dL_p:>7.2f}%\n")
        f.write("\n")
print("\nSaved table_phase4_convergence.txt")

# ─── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Phase 4 — Numerical Convergence: |R_fid - R_alt| and |Lambda_fid - Lambda_alt|',
             fontsize=11, fontweight='bold')

for col, (rows, xvals, xlabel, xsc) in enumerate([
    (rows_s, sigma_vals,  r'$\sigma$ (EOS smoothing)',   'linear'),
    (rows_n, N_vals,      r'$N_{\rm pts}$ (EOS table)',  'log'),
    (rows_r, rtol_vals,   r'rtol (Riccati ODE)',          'log'),
]):
    # dR panel
    dR_vals = [r[-3] for r in rows]
    axes[0, col].plot(xvals, dR_vals, 'o-', color='#1565C0', lw=2, ms=8,
                      markeredgecolor='k', markeredgewidth=0.5)
    axes[0, col].set_xlabel(xlabel, fontsize=10)
    axes[0, col].set_ylabel(r'$|\Delta R|$ (km)', fontsize=10)
    axes[0, col].set_xscale(xsc)
    axes[0, col].grid(True, alpha=0.3, ls='--')
    axes[0, col].set_title(r'$|\Delta R_{\rm fid-alt}|$', fontsize=10)

    # dL% panel
    dL_vals = [r[-1] for r in rows]
    axes[1, col].plot(xvals, dL_vals, 's-', color='#B71C1C', lw=2, ms=8,
                      markeredgecolor='k', markeredgewidth=0.5)
    axes[1, col].set_xlabel(xlabel, fontsize=10)
    axes[1, col].set_ylabel(r'$|\Delta\Lambda|/\bar\Lambda$ (%)', fontsize=10)
    axes[1, col].set_xscale(xsc)
    axes[1, col].grid(True, alpha=0.3, ls='--')
    axes[1, col].set_title(r'$|\Delta\Lambda_{\rm fid-alt}|/\bar\Lambda$', fontsize=10)

plt.tight_layout()
plt.savefig('figure_phase4_convergence.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved figure_phase4_convergence.png")
print("\nPhase 4 convergence done.")
