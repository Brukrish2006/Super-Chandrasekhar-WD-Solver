"""
Figure 4: Part II — Synthesis M-R curves combining:
  - Microscopic Landau quantization EOS
  - Macroscopic magnetic anisotropy (kappa)
  - Modified gravity f(R)

Curves:
  (1) Standard GR   : Chandra EOS, alpha=0, kappa=0,  no B-TOV
  (2) Landau Only   : hybrid EOS,  alpha=0, kappa=0,  WITH B-TOV  (Deb kappa=0 baseline ~2.36)
  (3) Pure Mag      : Chandra EOS, alpha=0, kappa=0.15, WITH B-TOV (~2.49-2.56)
  (4) Unified       : Chandra EOS, alpha=-3e12, kappa=0.15, WITH B-TOV
  (5) SYNTHESIS     : hybrid EOS,  alpha=-3e12, kappa=0.15, WITH B-TOV (full model)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from eos import EOS
from tov_solver import TOVSolver

Msun  = 1.989e33
R_km  = 1e5

def sweep(eos, alpha, kappa, rho_min=8.0, rho_max=10.0, N=25):
    solver = TOVSolver(eos, alpha=alpha, kappa=kappa)
    rhos   = np.logspace(rho_min, rho_max, N)
    R_list, M_list = [], []
    for rc in rhos:
        res = solver.solve(rc)
        if res:
            R_list.append(res['R'] / R_km)
            M_list.append(res['M'] / Msun)
    M_arr = np.array(M_list)
    R_arr = np.array(R_list)
    return R_arr, M_arr


def Mmax(arr):
    return round(float(max(arr)), 2) if len(arr) > 0 else 0.0

B0 = 3.79e14   # G

# EOS objects
eos_gr    = EOS(mode='chandra', magnetic_tov=False)
eos_mag   = EOS(mode='chandra', B_0=B0, magnetic_tov=True)   # Chandra + B-TOV
eos_hybrid= EOS(mode='hybrid',  B_0=B0, magnetic_tov=True)   # Landau/hybrid + B-TOV

print("Standard GR...")
R_gr,     M_gr     = sweep(eos_gr,     alpha=0.0,    kappa=0.0)
print(f"  Mmax={Mmax(M_gr):.4f}  (want 1.42)")

print("Landau Only (hybrid EOS, kappa=0, B-TOV)...")
R_landau, M_landau = sweep(eos_hybrid, alpha=0.0,    kappa=0.0)
print(f"  Mmax={Mmax(M_landau):.4f}  (Deb kappa=0: 2.36)")

print("Pure Mag (Chandra EOS, kappa=0.15, B-TOV)...")
R_mag,    M_mag    = sweep(eos_mag,    alpha=0.0,    kappa=0.15)
print(f"  Mmax={Mmax(M_mag):.4f}  (want ~2.49-2.56)")

print("Unified (Chandra EOS, alpha=-3e12, kappa=0.15, B-TOV)...")
R_uni,    M_uni    = sweep(eos_mag,    alpha=-3e12,  kappa=0.15)
print(f"  Mmax={Mmax(M_uni):.4f}  (want ~2.60)")

print("SYNTHESIS (hybrid EOS, alpha=-3e12, kappa=0.15, B-TOV)...")
R_syn,    M_syn    = sweep(eos_hybrid, alpha=-3e12,  kappa=0.15)
print(f"  Mmax={Mmax(M_syn):.4f}")

# -------- plot --------
fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(M_gr,     R_gr,     'k-',   lw=1.8, label=fr'Standard GR ($M_\mathrm{{max}}={Mmax(M_gr)}$)')
ax.plot(M_landau, R_landau, 'b-.',  lw=1.8, label=fr'Landau Only ($M_\mathrm{{max}}={Mmax(M_landau)}$)')
ax.plot(M_mag,    R_mag,    'g--',  lw=1.8, label=fr'Pure Mag ($M_\mathrm{{max}}={Mmax(M_mag)}$)')
ax.plot(M_uni,    R_uni,    'r--',  lw=1.8, label=fr'Unified ($M_\mathrm{{max}}={Mmax(M_uni)}$)')
ax.plot(M_syn,    R_syn,    'm-',   lw=2.5, label=fr'SYNTHESIS ($M_\mathrm{{max}}={Mmax(M_syn)}$)')

ax.set_xlabel(r'Mass  $M\ (M_\odot)$', fontsize=13)
ax.set_ylabel(r'Radius  $R\ \mathrm{(km)}$', fontsize=13)
ax.set_title('Part II: Synthesis Mass–Radius Relations\n'
             r'$\kappa=0.15$,  $\alpha=-3\times10^{12}\ \mathrm{cm}^2$,  '
             r'$B_0=3.79\times10^{14}\ \mathrm{G}$', fontsize=11)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.25, linestyle='--')
ax.set_xlim(left=0.5)
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(100))
plt.tight_layout()
plt.savefig('figure_4.png', dpi=300)
plt.close()
print("Saved figure_4.png")
