import numpy as np
import matplotlib.pyplot as plt
from eos import EOS
from tov_solver import TOVSolver

M_sun = 1.989e33
R_scale = 1e5

def plot_figure_3():
    print("Generating Figure 3 data...")
    # B = 10 B_c = 4.414e14 G
    # Constant B everywhere.
    B_0 = 4.414e14
    
    # We will compute the continuous chandra EOS as a baseline
    eos_chandra = EOS(mode='chandra')
    solver_chandra = TOVSolver(eos_chandra)
    
    rhos_chandra = np.logspace(5, 10.5, 100)
    R_chandra = []
    M_chandra = []
    for r_c in rhos_chandra:
        res = solver_chandra.solve(r_c)
        if res is not None:
            R_chandra.append(res['R'] / R_scale)
            M_chandra.append(res['M'] / M_sun)
            
    # And the discrete Landau EOS
    eos_landau = EOS(mode='landau', B_0=B_0)
    solver_landau = TOVSolver(eos_landau)
    
    # For landau, the kinks are very sharp, so we need more points
    rhos_landau = np.logspace(5, 10.5, 300)
    R_landau = []
    M_landau = []
    for r_c in rhos_landau:
        res = solver_landau.solve(r_c)
        if res is not None:
            R_landau.append(res['R'] / R_scale)
            M_landau.append(res['M'] / M_sun)
            
    M_chandra = np.array(M_chandra)
    R_chandra = np.array(R_chandra)
    if len(M_chandra) > 0:
        max_idx_c = np.argmax(M_chandra)
        M_chandra = M_chandra[:max_idx_c+1]
        R_chandra = R_chandra[:max_idx_c+1]
        
    M_landau = np.array(M_landau)
    R_landau = np.array(R_landau)
    if len(M_landau) > 0:
        max_idx_l = np.argmax(M_landau)
        M_landau = M_landau[:max_idx_l+1]
        R_landau = R_landau[:max_idx_l+1]
        
    plt.figure(figsize=(8, 6))
    plt.plot(M_chandra, R_chandra, 'k--', label='Continuous Fermi-Dirac (Standard)')
    plt.plot(M_landau, R_landau, 'b-', label='Discrete Landau Quantized (B = 10.0 Bc)')
    
    plt.xlabel('Mass ($M_\odot$)')
    plt.ylabel('Radius (km)')
    plt.title('Figure 3: Effect of Microscopic Landau Quantization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figure_3.png', dpi=300)
    plt.close()
    print("Saved figure_3.png")

if __name__ == "__main__":
    plot_figure_3()
