"""
Figure 7: Conservative-B0 Companion Case (rho_c - M relation)

Companion to Figure 6 (plot_fig5_neutronization.py), computed at a much weaker,
observationally-defensible field strength B0 = 1e13 G (~0.23 B_c) instead of the
fiducial B0 = 3.79e14 G (~8.6 B_c) used throughout the rest of the paper.

Motivation (see Section 5.5, "MHD stability at B ~ 1e14 G"): Manreza Paret et al.
(2015) and Coelho et al. (2014) find that stable 3D equilibria may not exist for
B >~ 1e13 G, roughly one order of magnitude below the paper's fiducial field. This
script computes the actual Mmax at that more conservative field strength, holding
kappa = 0.15 and alpha = -3e12 cm^2 fixed (same fiducial values used everywhere
else), so the headline 2.60 Msun result can be directly compared against a
companion number computed in the field regime current 3D-stability literature
regards as safer.

Key finding reproduced here: at B0 = 1e13 G the mass is *still rising* when the
central density reaches the adopted 12C/16O electron-capture cutoff
(rho_c = 1e10 g/cm^3, Section 5.4). Unlike the fiducial case -- where the static
turning point and the electron-capture cutoff nearly coincide (Figure 6) -- here
electron capture, not the static turning-point criterion, is the operative
physical limiter. The mass quoted at the cutoff (M ~ 1.53-1.56 Msun) is therefore
the physically meaningful "conservative companion" number, not the larger
unconstrained mathematical turning point (which occurs at rho_c ~ 3-5e10 g/cm^3,
already beyond the electron-capture-safe zone).

Run with:
    python plot_fig7_conservative_b0.py

Requires: numpy, scipy, matplotlib, and this repo's eos.py / tov_solver.py /
constants.py in the same directory. Takes roughly 3 minutes to run (most of it
spent building the discrete Landau EOS lookup table for the SYNTHESIS curve).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eos import EOS
from tov_solver import TOVSolver

Msun = 1.989e33

# electron-capture cutoff adopted throughout the paper (Section 5.4)
RHO_CUTOFF = 1e10  # g/cm^3

# fiducial coupling constants (unchanged from the rest of the paper)
KAPPA = 0.15
ALPHA = -3.0e12  # cm^2

# conservative companion field strength (Section 5.5)
B0_CONSERVATIVE = 1e13  # G  (~0.23 B_c, within the B0 <~ 1e13 G stability window)


def sweep_rho_m(eos, alpha, kappa, rho_min=7.5, rho_max=10.7, N=18):
    """Sweep central density, returning (rho_c, M/Msun, R[km]) arrays."""
    solver = TOVSolver(eos, alpha=alpha, kappa=kappa)
    rhos = np.logspace(rho_min, rho_max, N)
    rho_list, M_list, R_list = [], [], []
    for rc in rhos:
        res = solver.solve(rc)
        if res:
            rho_list.append(rc)
            M_list.append(res['M'] / Msun)
            R_list.append(res['R'] / 1e5)
    return np.array(rho_list), np.array(M_list), np.array(R_list)


def M_at_cutoff(eos, alpha, kappa, rho_cutoff=RHO_CUTOFF):
    """Mass evaluated exactly at the adopted electron-capture cutoff density."""
    solver = TOVSolver(eos, alpha=alpha, kappa=kappa)
    res = solver.solve(rho_cutoff)
    return res['M'] / Msun if res else None


def Mmax(arr):
    return round(float(arr.max()), 3) if len(arr) > 0 else 0.0


if __name__ == "__main__":
    print(f"Conservative companion field: B0 = {B0_CONSERVATIVE:.2e} G "
          f"({B0_CONSERVATIVE/4.414e13:.2f} B_c)")
    print(f"Fixed couplings: kappa = {KAPPA}, alpha = {ALPHA:.2e} cm^2")
    print(f"Electron-capture cutoff: rho_c = {RHO_CUTOFF:.0e} g/cm^3\n")

    # ---- EOS objects ----
    eos_gr = EOS(mode='chandra', magnetic_tov=False)
    eos_mag_cons = EOS(mode='chandra', B_0=B0_CONSERVATIVE, magnetic_tov=True)
    eos_hyb_cons = EOS(mode='hybrid', B_0=B0_CONSERVATIVE, magnetic_tov=True)

    # ---- Standard GR baseline (B-independent) ----
    print("Sweeping Standard GR...")
    rho_gr, M_gr, R_gr = sweep_rho_m(eos_gr, alpha=0.0, kappa=0.0,
                                      rho_min=7.0, rho_max=11.0, N=15)
    print(f"  Mmax = {Mmax(M_gr)} (want ~1.42)")

    # ---- Pure Mag, conservative B0 ----
    print("Sweeping Pure Mag (conservative B0, kappa=0.15)...")
    rho_mag, M_mag, R_mag = sweep_rho_m(eos_mag_cons, alpha=0.0, kappa=KAPPA)
    Mcut_mag = M_at_cutoff(eos_mag_cons, alpha=0.0, kappa=KAPPA)
    print(f"  Unconstrained Mmax = {Mmax(M_mag)}   "
          f"M(rho_c={RHO_CUTOFF:.0e}) = {Mcut_mag:.3f}")

    # ---- Unified, conservative B0 ----
    print("Sweeping Unified (conservative B0, kappa=0.15, alpha=-3e12)...")
    rho_uni, M_uni, R_uni = sweep_rho_m(eos_mag_cons, alpha=ALPHA, kappa=KAPPA)
    Mcut_uni = M_at_cutoff(eos_mag_cons, alpha=ALPHA, kappa=KAPPA)
    print(f"  Unconstrained Mmax = {Mmax(M_uni)}   "
          f"M(rho_c={RHO_CUTOFF:.0e}) = {Mcut_uni:.3f}")

    # ---- Synthesis (hybrid Landau EOS), conservative B0 ----
    print("Sweeping SYNTHESIS (hybrid EOS, conservative B0, kappa=0.15, alpha=-3e12)...")
    rho_syn, M_syn, R_syn = sweep_rho_m(eos_hyb_cons, alpha=ALPHA, kappa=KAPPA)
    Mcut_syn = M_at_cutoff(eos_hyb_cons, alpha=ALPHA, kappa=KAPPA)
    print(f"  Unconstrained Mmax = {Mmax(M_syn)}   "
          f"M(rho_c={RHO_CUTOFF:.0e}) = {Mcut_syn:.3f}")

    print(f"\nConservative companion summary at rho_c = {RHO_CUTOFF:.0e} g/cm^3:")
    print(f"  Pure Mag  : M = {Mcut_mag:.2f} Msun")
    print(f"  Unified   : M = {Mcut_uni:.2f} Msun")
    print(f"  Synthesis : M = {Mcut_syn:.2f} Msun "
          f"(agrees with Unified to {abs(Mcut_syn-Mcut_uni)/Mcut_uni*100:.2f}%)")

    # ------------------------------------------------------------ plot
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(rho_gr, M_gr, 'k-', lw=1.8,
            label=fr'Standard GR ($M_\mathrm{{max}}={Mmax(M_gr)}$)')
    ax.plot(rho_mag, M_mag, 'g--', lw=1.8,
            label=fr'Pure Mag, conservative $B_0$ ($M(\rho_{{\rm cut}})={Mcut_mag:.2f}$)')
    ax.plot(rho_uni, M_uni, 'r--', lw=1.8,
            label=fr'Unified, conservative $B_0$ ($M(\rho_{{\rm cut}})={Mcut_uni:.2f}$)')
    ax.plot(rho_syn, M_syn, 'm-', lw=2.5,
            label=fr'SYNTHESIS, conservative $B_0$ ($M(\rho_{{\rm cut}})={Mcut_syn:.2f}$)')

    ax.axvline(x=RHO_CUTOFF, color='darkorange', linestyle=':', lw=1.5,
               label=r'$^{12}$C/$^{16}$O electron-capture cutoff ($\rho_c=10^{10}\ \mathrm{g/cm}^3$)')

    ax.plot([Mcut_mag and RHO_CUTOFF], [Mcut_mag], 'go', ms=7, zorder=5)
    ax.plot([RHO_CUTOFF], [Mcut_uni], 'ro', ms=7, zorder=5)
    ax.plot([RHO_CUTOFF], [Mcut_syn], 'mo', ms=7, zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel(r'Central Density  $\rho_c\ (\mathrm{g/cm}^3)$', fontsize=13)
    ax.set_ylabel(r'Mass  $M/M_\odot$', fontsize=13)
    ax.set_title(
        r'Conservative Companion Case: $\rho_c$–$M$ Relation at $B_0 = 1\times10^{13}$~G'
        '\n' r'$\kappa=0.15$,  $\alpha=-3\times10^{12}\ \mathrm{cm}^2$  '
        r'(cf. Figure 6, fiducial $B_0=3.79\times10^{14}$~G)', fontsize=11)
    ax.legend(fontsize=9.5, loc='upper left')
    ax.grid(True, alpha=0.25, linestyle='--')
    plt.tight_layout()
    plt.savefig('figure_7.png', dpi=300)
    plt.close()
    print("\nSaved figure_7.png")
