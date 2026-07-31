"""
Phase 0: Radius Contour from Existing Grid
============================================
Re-run the (alpha, kappa) degeneracy grid from Figure 3, this time storing
R(M_max) alongside M_max for all grid points.

Decision point:
  - If iso-M_max contours are ALSO close to iso-R contours => need Lambda
  - If they visibly diverge => R alone is a discriminator (publishable now)

Deliverable: phase0_grid_data.npz + figure_phase0_contour.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
from eos import EOS
from tov_solver import TOVSolver

Msun  = 1.989e33
km    = 1e5   # cm per km

# ── Grid definition (matching the Figure 3 / Section 3.4 parameter space) ─────
# Narrow alpha range to stay within perturbative validity |alpha*R0| << 1
alpha_vals = np.linspace(-1.0e13, -1e12, 12)
kappa_vals = np.linspace(0.0,    0.40,  12)

# Central density sweep — paper's Section 3.4 range
rhos = np.logspace(8.5, 10.0, 30)

A, K = np.meshgrid(alpha_vals, kappa_vals)
M_max    = np.zeros_like(A)
R_at_Mmax = np.zeros_like(A)

# Chandra EOS + magnetic TOV (same as Figure 3 / "Unified" fiducial)
eos = EOS(mode='chandra', B_0=3.79e14, magnetic_tov=True)

print("Phase 0: Running (alpha, kappa) grid — storing M_max and R(M_max)")
print(f"Grid size: {len(kappa_vals)} x {len(alpha_vals)} = {A.size} points")
print(f"Central density sweep: {rhos[0]:.2e} – {rhos[-1]:.2e} g/cm³ ({len(rhos)} pts)")
print()

for i in range(len(kappa_vals)):
    for j in range(len(alpha_vals)):
        solver = TOVSolver(eos, alpha=A[i, j], kappa=K[i, j], compute_tidal=False)
        best_m = 0.0
        best_r = 0.0
        for rc in rhos:
            res = solver.solve(rc)
            if res is None:
                continue
            m = res['M'] / Msun
            if m > best_m:
                best_m = m
                best_r = res['R'] / km   # store in km
        M_max[i, j]     = best_m
        R_at_Mmax[i, j] = best_r
        print(f"  α={A[i,j]:.2e}, κ={K[i,j]:.2f}  => "
              f"M_max={best_m:.3f} M☉,  R={best_r:.2f} km")

# ── Save raw data ──────────────────────────────────────────────────────────────
np.savez('phase0_grid_data.npz',
         alpha=alpha_vals, kappa=kappa_vals,
         A=A, K=K, M_max=M_max, R_at_Mmax=R_at_Mmax)
print("\nSaved phase0_grid_data.npz")

# ── Plot overlaid contours ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ── Left panel: M_max contours ─────────────────────────────────────────────────
ax = axes[0]
cf = ax.contourf(A / 1e12, K, M_max, levels=14, cmap='plasma')
cs = ax.contour(A / 1e12, K, M_max, levels=14, colors='white',
                linewidths=0.7, alpha=0.6)
ax.clabel(cs, inline=True, fmt=r'%.2f $M_\odot$', fontsize=8)
cbar = fig.colorbar(cf, ax=ax)
cbar.set_label(r'$M_{\max}\ (M_\odot)$', fontsize=11)
ax.plot(-3.0, 0.15, 'w*', markersize=14, zorder=5,
        label=r'Fiducial $(\alpha{=}{-}3\!\times\!10^{12},\ \kappa{=}0.15)$')
ax.set_xlabel(r'$\alpha\ (10^{12}\ \mathrm{cm}^2)$', fontsize=12)
ax.set_ylabel(r'$\kappa$', fontsize=12)
ax.set_title(r'$M_{\max}\ (M_\odot)$ over $(\alpha,\kappa)$ space', fontsize=12)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2, linestyle='--')

# ── Right panel: R(M_max) contours ────────────────────────────────────────────
ax = axes[1]
cf2 = ax.contourf(A / 1e12, K, R_at_Mmax, levels=14, cmap='viridis')
cs2 = ax.contour(A / 1e12, K, R_at_Mmax, levels=14, colors='white',
                 linewidths=0.7, alpha=0.6)
ax.clabel(cs2, inline=True, fmt=r'%.1f km', fontsize=8)
cbar2 = fig.colorbar(cf2, ax=ax)
cbar2.set_label(r'$R(M_{\max})\ \mathrm{[km]}$', fontsize=11)
# Overlay M_max iso-lines as dashed black curves for comparison
cs_m = ax.contour(A / 1e12, K, M_max, levels=8, colors='red',
                  linewidths=1.2, linestyles='--', alpha=0.85)
ax.clabel(cs_m, inline=True, fmt=r'$M_{\max}$=%.2f', fontsize=7.5)
ax.plot(-3.0, 0.15, 'r*', markersize=14, zorder=5,
        label=r'Fiducial $(\alpha{=}{-}3\!\times\!10^{12},\ \kappa{=}0.15)$')
ax.set_xlabel(r'$\alpha\ (10^{12}\ \mathrm{cm}^2)$', fontsize=12)
ax.set_ylabel(r'$\kappa$', fontsize=12)
ax.set_title(r'$R(M_{\max})$ [km] with $M_{\max}$ iso-lines (dashed red)', fontsize=12)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2, linestyle='--')

plt.suptitle(r'Phase 0 — Degeneracy Diagnostic: Does $R(M_{\max})$ Break the $M_{\max}$ Degeneracy?',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('figure_phase0_contour.png', dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Saved figure_phase0_contour.png")

# ── Quantitative report ────────────────────────────────────────────────────────
print("\n─── Phase 0 Quantitative Summary ───")
print(f"M_max range: {M_max.min():.3f} – {M_max.max():.3f} M☉")
print(f"R_at_Mmax range: {R_at_Mmax.min():.2f} – {R_at_Mmax.max():.2f} km")

# Find points near a target iso-M_max contour (closest to fiducial M_max)
# We look for grid points within ±0.03 M☉ of the fiducial M_max
fid_solver = TOVSolver(eos, alpha=-3e12, kappa=0.15)
fid_m_best, fid_r_best = 0.0, 0.0
for rc in np.logspace(8.5, 10.0, 50):
    res = fid_solver.solve(rc)
    if res and res['M'] / Msun > fid_m_best:
        fid_m_best = res['M'] / Msun
        fid_r_best = res['R'] / km
print(f"\nFiducial point: M_max = {fid_m_best:.3f} M☉,  R = {fid_r_best:.2f} km")

tol = 0.05  # M☉
mask = np.abs(M_max - fid_m_best) < tol
if mask.sum() > 2:
    r_on_contour = R_at_Mmax[mask]
    delta_r = r_on_contour.max() - r_on_contour.min()
    delta_r_pct = 100.0 * delta_r / r_on_contour.mean()
    print(f"\nPoints within ±{tol} M☉ of fiducial M_max: {mask.sum()}")
    print(f"  R range on contour: {r_on_contour.min():.2f} – {r_on_contour.max():.2f} km")
    print(f"  ΔR/R = {delta_r_pct:.1f}%")
    if delta_r_pct > 5:
        print("  => R varies significantly on the iso-mass contour — DEGENERACY BROKEN by R alone!")
    else:
        print("  => R variation is small — need tidal deformability Λ for discrimination.")
else:
    print(f"\nToo few grid points near fiducial contour — proceed to Phase 3 with finer grid.")

print("\nPhase 0 complete.")
