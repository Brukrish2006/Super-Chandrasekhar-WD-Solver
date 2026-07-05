import numpy as np
from constants import *
from eos import EOS
from tov_solver import TOVSolver

M_sun = 1.989e33

def find_m_max(eos_mode='chandra', B_0=3.79e14, alpha=0.0, kappa=0.0):
    eos = EOS(mode=eos_mode, B_0=B_0)
    solver = TOVSolver(eos, alpha=alpha, kappa=kappa)
    
    # We sweep central densities from 1e8 to 1e11 (near neutronization threshold)
    rhos = np.logspace(8, 10.5, 30)
    m_list = []
    
    print(f"Finding M_max for mode={eos_mode}, alpha={alpha}, kappa={kappa}, B_0={B_0:.2e}")
    
    M_max = 0.0
    for r_c in rhos:
        res = solver.solve(r_c)
        if res is not None:
            m_val = res['M'] / M_sun
            m_list.append(m_val)
            if m_val > M_max:
                M_max = m_val
        else:
            m_list.append(0.0)
            
    print(f"M_max = {M_max:.3f} M_sun")
    return M_max

if __name__ == "__main__":
    # 1. Standard Chandrasekhar (GR)
    find_m_max(eos_mode='chandra', alpha=0.0, kappa=0.0)
    
    # 2. Pure f(R) gravity (alpha = -3e12, kappa = 0)
    find_m_max(eos_mode='chandra', alpha=-3.0e12, kappa=0.0)
    
    # 3. Pure Magnetic (kappa = 0.15, alpha = 0)
    find_m_max(eos_mode='chandra', alpha=0.0, kappa=0.15)
    
    # 4. Continuous Unified (kappa = 0.15, alpha = -3e12)
    find_m_max(eos_mode='chandra', alpha=-3.0e12, kappa=0.15)
    
    # 5. Full Synthesis (Landau + kappa + alpha)
    # Using smaller density range to speed up the landau EOS tests
    find_m_max(eos_mode='hybrid', B_0=3.79e14, alpha=-3.0e12, kappa=0.15)
