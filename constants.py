import numpy as np

# CGS Constants
c = 2.99792458e10       # cm / s
G = 6.67430e-8          # cm^3 / (g s^2)
hbar = 1.054571817e-27  # erg s
h = 2 * np.pi * hbar
m_e = 9.1093837e-28     # g
m_u = 1.66053906660e-24 # g
mu_e = 2.0              # mean molecular weight per electron
e_charge = 4.80320425e-10 # statC

# Critical Magnetic Field
B_c = (m_e**2 * c**3) / (e_charge * hbar)  # ~4.414e13 G

# Pre-factors for Chandrasekhar EOS
K_rho = (8 * np.pi * mu_e * m_u * (m_e * c)**3) / (3 * h**3)
K_P = (np.pi * m_e**4 * c**5) / (3 * h**3)
K_eps = (np.pi * m_e**4 * c**5) / (3 * h**3)
