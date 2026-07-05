"""
Figure 2: Contour Map of Maximum Stable Mass M_max/M_sun
over the (alpha, kappa) parameter space.

Physics: Chandra EOS + parametric (alpha, kappa) — NO explicit B-field TOV.
This is the macroscopic parameter survey (Part I of paper).
The B-field enters implicitly through kappa (anisotropy parameter), not
through explicit B^2/8pi magnetic stress terms.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eos import EOS
from tov_solver import TOVSolver

Msun = 1.989e33

def plot_figure_2():
    print("Generating Figure 2 data... (this takes a while)")

    # Parameter grid — same as paper
    alpha_vals = np.linspace(-4e13, -1e12, 12)
    kappa_vals = np.linspace(0.0, 0.4, 12)

    A, K = np.meshgrid(alpha_vals, kappa_vals)
    M_max = np.zeros_like(A)

    # Chandra EOS, NO magnetic TOV (pure parametric survey)
    eos  = EOS(mode='chandra', magnetic_tov=False)
    # Scan central densities around expected M_max peak
    rhos = np.logspace(8.5, 10.5, 15)

    for i in range(len(kappa_vals)):
        for j in range(len(alpha_vals)):
            solver = TOVSolver(eos, alpha=A[i,j], kappa=K[i,j])
            best = 0.0
            for rc in rhos:
                res = solver.solve(rc)
                if res:
                    m = res['M'] / Msun
                    if m > best:
                        best = m
            M_max[i,j] = best
            print(f"  alpha={A[i,j]:.1e}, kappa={K[i,j]:.2f} => M_max={best:.3f}")

    # -------- plot --------
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Contour LINES only on white background — matches paper style
    levels = np.linspace(M_max.min(), M_max.max(), 14)
    # Use a perceptually-uniform colormap for line colours
    cs = ax.contour(A/1e12, K, M_max, levels=levels,
                    cmap='nipy_spectral', linewidths=1.6)
    ax.clabel(cs, inline=True, fmt=r'%.2f $M_\odot$', fontsize=8.5)

    # Mark fiducial point used in paper
    ax.plot(-3.0, 0.15, 'r*', markersize=14, zorder=5,
            label=r'Fiducial: $\kappa=0.15,\ \alpha=-3\times10^{12}\ \mathrm{cm}^2$')

    ax.set_xlabel(r'Modified Gravity Parameter $\alpha\ (10^{12}\ \mathrm{cm}^2)$', fontsize=13)
    ax.set_ylabel(r'Magnetic Anisotropy $\kappa$', fontsize=13)
    ax.set_title(r'Contour Map of $M_{\max}\ (M_\odot)$ in $(\alpha,\kappa)$ Space' + '\n'
                 r'Chandra EOS + Magnetic TOV', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.25, linestyle='--', color='grey')

    plt.tight_layout()
    plt.savefig('figure_2.png', dpi=300, facecolor='white')
    plt.close()
    print("Saved figure_2.png")


if __name__ == "__main__":
    plot_figure_2()
