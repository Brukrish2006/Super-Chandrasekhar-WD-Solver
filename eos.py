import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from constants import *

def chandra_eos(rho):
    x = (rho / K_rho)**(1/3)
    eps_e = K_eps * (x * (2*x**2 + 1) * np.sqrt(x**2 + 1) - np.arcsinh(x))
    eps = rho * c**2 + eps_e
    P = K_P * (x * (2*x**2 - 3) * np.sqrt(x**2 + 1) + 3 * np.arcsinh(x))
    return P, eps

# Integrals can be solved analytically
def analytical_eps_int(x, A):
    return 0.5 * x * np.sqrt(A**2 + x**2) + 0.5 * A**2 * np.arcsinh(x / A)

def analytical_P_int(x, A):
    return 0.5 * x * np.sqrt(A**2 + x**2) - 0.5 * A**2 * np.arcsinh(x / A)

def landau_eos_fixed_B(rho, B):
    B_star = B / B_c
    n_e_target = rho / (mu_e * m_u)
    pref_n = (e_charge * B) / (2 * np.pi**2 * hbar**2 * c)
    
    def n_e_func(E_F):
        arg = (E_F**2 - m_e**2 * c**4) / (2 * m_e**2 * c**4 * B_star)
        if arg < 0: return -n_e_target
        nu_max = int(np.floor(arg))
        
        n_e = 0.0
        for nu in range(nu_max + 1):
            g_nu = 1 if nu == 0 else 2
            p_F_nu_c = np.sqrt(max(0, E_F**2 - m_e**2 * c**4 * (1 + 2*nu*B_star)))
            n_e += g_nu * (p_F_nu_c / c)
        return n_e * pref_n - n_e_target

    E_F_min = m_e * c**2
    x_cl = (rho / K_rho)**(1/3)
    E_F_max_guess = np.sqrt((x_cl * m_e * c**2)**2 + (m_e*c**2)**2) * 5.0
    
    f_a = n_e_func(E_F_min)
    if f_a >= 0:
        E_F = E_F_min
    else:
        while n_e_func(E_F_max_guess) <= 0:
            E_F_max_guess *= 2.0
            
        E_F = brentq(n_e_func, E_F_min, E_F_max_guess)
    
    arg = (E_F**2 - m_e**2 * c**4) / (2 * m_e**2 * c**4 * B_star)
    nu_max = int(np.floor(arg))
    
    pref_eps = (e_charge * B * m_e**2 * c**2) / (2 * np.pi**2 * hbar**2)
    pref_P = pref_eps
    
    eps_e = 0.0
    P_e = 0.0
    
    for nu in range(nu_max + 1):
        g_nu = 1 if nu == 0 else 2
        gamma_nu_sq = 1 + 2*nu*B_star
        gamma_nu = np.sqrt(gamma_nu_sq)
        
        x_nu = np.sqrt(max(0.0, (E_F / (m_e*c**2))**2 - gamma_nu_sq))
        
        if x_nu > 0:
            res_eps = analytical_eps_int(x_nu, gamma_nu)
            eps_e += g_nu * res_eps
            
            res_P = analytical_P_int(x_nu, gamma_nu)
            P_e += g_nu * res_P
        
    eps = rho * c**2 + eps_e * pref_eps
    P_e = P_e * pref_P
    return P_e, eps

class EOS:
    """
    Equation of State for magnetized white dwarfs.

    Parameters
    ----------
    mode : str
        'chandra'  — pure Chandrasekhar EOS (no Landau correction)
        'landau'   — pure Landau EOS at fixed B = B_0
        'hybrid'   — Landau where B(rho) > 0.1 B_c, Chandra elsewhere
    B_0 : float
        Peak magnetic field strength [G]. Used in both the B(rho) profile
        and as the fixed field for 'landau' mode.
    magnetic_tov : bool
        If True, expose the B(rho) profile so the TOV solver can include
        the explicit magnetic-stress-tensor terms in the structure equations
        (Deb et al. 2022 formulation). Set True for physical magnetic runs
        (Pure Mag, Unified, Synthesis). Set False for pure parametric
        (alpha, kappa) studies without explicit B-field in the TOV.
    sigma : int
        Gaussian smoothing width for R0 derivatives.
    N_points : int
        Number of tabulation points.
    """
    def __init__(self, mode='chandra', B_0=3.79e14, magnetic_tov=False,
                 sigma=20, N_points=1000):
        self.mode = mode
        self.B_0 = B_0
        self.magnetic_tov = magnetic_tov
        self.sigma = sigma
        # Deb et al. B(rho) profile parameters
        self.Bs     = 1e9     # surface field [G]
        self.eta    = 0.2
        self.gamma_B = 0.9
        self.rho_0  = 1e9    # reference density [g/cm³]

        rho_arr = np.logspace(4, 11.5, N_points)
        P_arr   = np.zeros_like(rho_arr)
        eps_arr = np.zeros_like(rho_arr)
        
        for i, r in enumerate(rho_arr):
            B = self.get_B(r)
            if mode == 'chandra':
                P, e = chandra_eos(r)
            elif mode == 'landau':
                P, e = landau_eos_fixed_B(r, B_0)
            elif mode == 'hybrid':
                if B > 0.1 * B_c:
                    P, e = landau_eos_fixed_B(r, B)
                else:
                    P, e = chandra_eos(r)
            else:
                P, e = chandra_eos(r)
            
            P_arr[i] = P
            eps_arr[i] = e
            
        # Ensure strict monotonicity
        P_arr = P_arr + np.linspace(1e-10, 1e-9, len(P_arr))
        sort_idx = np.argsort(P_arr)
        P_arr   = P_arr[sort_idx]
        rho_arr = rho_arr[sort_idx]
        eps_arr = eps_arr[sort_idx]
        
        self.P_from_rho  = interp1d(rho_arr, P_arr,   kind='linear', fill_value="extrapolate")
        self.eps_from_rho = interp1d(rho_arr, eps_arr, kind='linear', fill_value="extrapolate")
        self.rho_interp  = interp1d(P_arr, rho_arr, kind='linear', fill_value="extrapolate")
        self.eps_interp  = interp1d(P_arr, eps_arr, kind='linear', fill_value="extrapolate")

        # Smooth Chandra-based arrays for R0 derivatives (avoids Landau kinks)
        P_smooth   = np.zeros_like(rho_arr)
        eps_smooth = np.zeros_like(rho_arr)
        for i, r in enumerate(rho_arr):
            P_s, e_s = chandra_eos(r)
            P_smooth[i]   = P_s
            eps_smooth[i] = e_s
            
        R0_arr = -(8 * np.pi * G / c**4) * (-eps_smooth + 3 * P_smooth)
        
        if sigma > 0:
            R0_arr = gaussian_filter1d(R0_arr, sigma=sigma)
            
        valid_idx = P_smooth > 0
        P_valid   = P_smooth[valid_idx]
        R0_valid  = R0_arr[valid_idx]
        eps_valid = eps_smooth[valid_idx]
        
        log_P = np.log(P_valid)
        sort_idx2 = np.argsort(log_P)
        log_P    = log_P[sort_idx2]
        R0_valid = R0_valid[sort_idx2]
        
        log_P, unique_idx = np.unique(log_P, return_index=True)
        R0_valid  = R0_valid[unique_idx]
        eps_valid = eps_valid[unique_idx]
        
        self.R0_spline    = InterpolatedUnivariateSpline(log_P, R0_valid, k=3)
        self.dR0_dlogP    = self.R0_spline.derivative(1)
        self.d2R0_dlogP2  = self.R0_spline.derivative(2)
        
        self.eps_spline   = InterpolatedUnivariateSpline(log_P, eps_valid, k=3)
        self.deps_dlogP   = self.eps_spline.derivative(1)

    # ------------------------------------------------------------------
    # B-field profile
    # ------------------------------------------------------------------
    def get_B(self, rho):
        """Return B(rho) [Gauss] following Deb et al. (2022) exponential profile."""
        return self.Bs + self.B_0 * (1.0 - np.exp(-self.eta * (rho / self.rho_0)**self.gamma_B))

    # ------------------------------------------------------------------
    # EOS look-ups
    # ------------------------------------------------------------------
    def get_rho_eps(self, P):
        return float(self.rho_interp(P)), float(self.eps_interp(P))
        
    def get_R0_derivs(self, P):
        if P <= 1e21:
            return 0.0, 0.0, 0.0, 0.0
        logP = np.log(P)
        R0       = float(self.R0_spline(logP))
        dR0_dP   = float(self.dR0_dlogP(logP) / P)
        d2R0_dP2 = float((self.d2R0_dlogP2(logP) - self.dR0_dlogP(logP)) / (P**2))
        deps_dP  = float(self.deps_dlogP(logP) / P)
        return R0, dR0_dP, d2R0_dP2, deps_dP

    def get_P_eps_from_rho(self, rho):
        return float(self.P_from_rho(rho)), float(self.eps_from_rho(rho))
