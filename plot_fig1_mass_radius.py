"""
Figure 1: Mass-Radius Relations — Part I: Macroscopic Synthesized Models
Four curves:
  (1) Standard GR   : Chandra EOS, alpha=0, kappa=0         (no B-field TOV)
  (2) Pure f(R)     : Chandra EOS, alpha=-3e12, kappa=0      (no B-field TOV)
  (3) Pure Mag      : Chandra EOS, kappa=0.15, alpha=0       (WITH B-field TOV, TO)
  (4) Unified       : Chandra EOS, alpha=-3e12, kappa=0.15   (WITH B-field TOV, TO)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from eos import EOS
from tov_solver import TOVSolver

Msun  = 1.989e33
R_km  = 1e5        # 1 km in cm

# ------------------------------------------------------------------ helpers
def sweep(eos, alpha, kappa, rho_min=7.0, rho_max=10.0, N=35):
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
    return R_arr, M_arr   # full curve: stable + unstable branch (gives nose/turnaround)

def Mmax(M_arr):
    return round(float(max(M_arr)), 2) if len(M_arr) > 0 else 0.0

# ------------------------------------------------------------------ EOS objects
eos_gr  = EOS(mode='chandra', magnetic_tov=False)              # no B-field TOV
eos_mag = EOS(mode='chandra', B_0=3.79e14, magnetic_tov=True)  # TO B-field TOV

print("Sweeping Standard GR...")
R_gr,  M_gr  = sweep(eos_gr,  alpha=0.0,    kappa=0.0)
print(f"  M_max = {Mmax(M_gr)} (want 1.42)")

print("Sweeping Pure f(R)...")
R_fR,  M_fR  = sweep(eos_gr,  alpha=-3e12,  kappa=0.0)
print(f"  M_max = {Mmax(M_fR)} (this curve has NOT turned over yet at the "
      f"rho_c=1e10 cutoff -- see note below)")

print("Sweeping Pure Mag (kappa=0.15, B-field TOV)...")
R_mag, M_mag = sweep(eos_mag, alpha=0.0,    kappa=0.15)
print(f"  M_max = {Mmax(M_mag)} (want ~2.56)")

print("Sweeping Unified (kappa=0.15, alpha=-3e12, B-field TOV)...")
R_uni, M_uni = sweep(eos_mag, alpha=-3e12,  kappa=0.15)
print(f"  M_max = {Mmax(M_uni)} (want ~2.60)")

# ------------------------------------------------------------------ plot
fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(M_gr,  R_gr,  'k-',   lw=1.8, label=fr'Standard GR ($M_\mathrm{{max}}={Mmax(M_gr)}\ M_\odot$)')
ax.plot(M_fR,  R_fR,  'b--',  lw=1.8, label=fr'Pure $f(R)$  ($M_\mathrm{{max}}={Mmax(M_fR)}\ M_\odot$)')
ax.plot(M_mag, R_mag, 'g-.',  lw=1.8, label=fr'Pure Mag    ($M_\mathrm{{max}}={Mmax(M_mag)}\ M_\odot$)')
ax.plot(M_uni, R_uni, 'r-',   lw=2.2, label=fr'Unified     ($M_\mathrm{{max}}={Mmax(M_uni)}\ M_\odot$)')

ax.set_xlabel(r'Mass  $M\ (M_\odot)$', fontsize=13)
ax.set_ylabel(r'Radius  $R\ \mathrm{(km)}$', fontsize=13)
ax.set_title('Part I: Mass–Radius Relations\n'
             r'$\kappa=0.15$,  $\alpha=-3\times10^{12}\ \mathrm{cm}^2$,  '
             r'$B_0=3.79\times10^{14}\ \mathrm{G}$', fontsize=11)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.25, linestyle='--')
ax.set_xlim(left=0.5)
ax.xaxis.set_minor_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(100))
plt.tight_layout()
plt.savefig('figure_1.png', dpi=300)
plt.close()
print("Saved figure_1.png")
