"""
Figure 5: Central Density vs Mass (rho_c - M relation)
Same curves as Figure 4 but plotted as M vs rho_c instead of M vs R.
Includes neutronization threshold line at rho_c = 10^10 g/cm^3.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eos import EOS
from tov_solver import TOVSolver

Msun = 1.989e33

def sweep_rho_m(eos, alpha, kappa, rho_min=7.0, rho_max=11.0, N=60):
    solver = TOVSolver(eos, alpha=alpha, kappa=kappa)
    rhos   = np.logspace(rho_min, rho_max, N)
    rho_list, M_list = [], []
    for rc in rhos:
        res = solver.solve(rc)
        if res:
            rho_list.append(rc)
            M_list.append(res['M'] / Msun)
    rho_arr = np.array(rho_list)
    M_arr   = np.array(M_list)
    return rho_arr, M_arr


def Mmax(arr):
    return round(float(max(arr)), 2) if len(arr) > 0 else 0.0

B0 = 3.79e14

eos_gr     = EOS(mode='chandra', magnetic_tov=False)
eos_mag    = EOS(mode='chandra', B_0=B0, magnetic_tov=True)
eos_hybrid = EOS(mode='hybrid',  B_0=B0, magnetic_tov=True)

print("Standard GR...")
rho_gr,     M_gr     = sweep_rho_m(eos_gr,     alpha=0.0,    kappa=0.0)

print("Landau Only...")
rho_landau, M_landau = sweep_rho_m(eos_hybrid, alpha=0.0,    kappa=0.0)

print("Pure Mag...")
rho_mag,    M_mag    = sweep_rho_m(eos_mag,    alpha=0.0,    kappa=0.15)

print("Unified...")
rho_uni,    M_uni    = sweep_rho_m(eos_mag,    alpha=-3e12,  kappa=0.15)

print("SYNTHESIS...")
rho_syn,    M_syn    = sweep_rho_m(eos_hybrid, alpha=-3e12,  kappa=0.15)

# -------- plot --------
fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(rho_gr,     M_gr,     'k-',  lw=1.8, label=fr'Standard GR ($M_\mathrm{{max}}={Mmax(M_gr)}$)')
ax.plot(rho_landau, M_landau, 'b-.', lw=1.8, label=fr'Landau Only ($M_\mathrm{{max}}={Mmax(M_landau)}$)')
ax.plot(rho_mag,    M_mag,    'g--', lw=1.8, label=fr'Pure Mag ($M_\mathrm{{max}}={Mmax(M_mag)}$)')
ax.plot(rho_uni,    M_uni,    'r--', lw=1.8, label=fr'Unified ($M_\mathrm{{max}}={Mmax(M_uni)}$)')
ax.plot(rho_syn,    M_syn,    'm-',  lw=2.5, label=fr'SYNTHESIS ($M_\mathrm{{max}}={Mmax(M_syn)}$)')

# Neutronization threshold
ax.axvline(x=1e10, color='darkorange', linestyle=':', lw=1.5,
           label=r'Neutronization threshold ($\rho_c = 10^{10}\ \mathrm{g/cm}^3$)')

ax.set_xscale('log')
ax.set_xlabel(r'Central Density  $\rho_c\ (\mathrm{g/cm}^3)$', fontsize=13)
ax.set_ylabel(r'Mass  $M/M_\odot$', fontsize=13)
ax.set_title(r'Part II: $\rho_c$–$M$ Relation: Synthesis' + '\n'
             r'$\kappa=0.15$,  $\alpha=-3\times10^{12}\ \mathrm{cm}^2$,  '
             r'$B_0=3.79\times10^{14}\ \mathrm{G}$', fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.25, linestyle='--')
plt.tight_layout()
plt.savefig('figure_5.png', dpi=300)
plt.close()
print("Saved figure_5.png")
