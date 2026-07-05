"""
TOV solver implementing:
  (a) Deb, Mukhopadhyay & Weber (2022) TO-orientation magnetic TOV:
        P_eff = P_fluid + B^2/(8*pi)
        eps_tot = eps_fluid + B^2/(8*pi)
        Bowers-Liang factor (1 - kappa/3) on pressure gradient
  (b) Das & Mukhopadhyay (2015) first-order f(R) modified TOV:
        f(R) = R + alpha*R^2  (Starobinsky model)
        State: [P0_bg, M0_bg, P_alpha, M_alpha]
          - (P0_bg, M0_bg) : GR background track for evaluating R(0) brackets
          - (P_alpha, M_alpha) : full modified-star quantities
        GR terms in dP_alpha/dr and dM_alpha/dr use M_alpha, P_alpha (modified)
        Correction brackets (Das 2015 eqs 3.7, 3.8, 3.9) use background M0, P0

References:
    Deb, Mukhopadhyay & Weber 2022, ApJ 926 66
    Das & Mukhopadhyay 2015, JCAP 05 (2015) 045
"""
import numpy as np
from scipy.integrate import solve_ivp
from constants import *
from eos import EOS


class TOVSolver:
    def __init__(self, eos, alpha=0.0, kappa=0.0):
        self.eos   = eos
        self.alpha = alpha   # f(R) coupling [cm^2], negative for Starobinsky
        self.kappa = kappa   # Bowers-Liang anisotropy (dimensionless)
        self._magnetic = getattr(eos, 'magnetic_tov', False)

    # ------------------------------------------------------------------
    def _get_magnetic(self, P_eff):
        """
        Iteratively solve for (rho, eps_fluid, B, B_mag) given P_eff:
            P_fluid = P_eff - B(rho)^2/(8*pi)
        """
        B_mag = 0.0
        for _ in range(5):
            P_fluid = max(P_eff - B_mag, 0.0)
            rho, eps = self.eos.get_rho_eps(P_fluid)
            B = self.eos.get_B(max(rho, 1e4))
            B_mag = B**2 / (8.0 * np.pi)
        return rho, eps, B, B_mag

    # ------------------------------------------------------------------
    def derivs(self, r, y):
        """
        y = [P0_fl, M0, P_eff, M]

        P0_fl, M0  -- fluid GR background (for computing R0, R0', R0'' on bg)
        P_eff, M   -- modified star (magnetic TOV + Das 2015 f(R) correction)

        Das 2015 strategy (correct first-order approach):
          - GR terms in dP/dr and dM/dr use M (modified mass), P_eff (modified P)
          - Correction brackets use background quantities (M0, P0_fl, R0_val)
          - This is the standard way to solve modified-gravity TOV perturbatively:
            the background tells you the correction; the modified star tracks the
            full hydrostatic equilibrium under the corrected potential.
        """
        P0_fl, M0, P_eff, M = y

        # ----------------------------------------------------------------
        # Background: pure fluid GR (no magnetic, no f(R))
        # This track provides R(0), R'(0), R''(0) at the FLUID GR background.
        # ----------------------------------------------------------------
        rho0, eps0 = self.eos.get_rho_eps(max(P0_fl, 0.0))
        A0     = eps0 + P0_fl
        Btrm0  = M0 + (4.0*np.pi * r**3 * P0_fl) / c**2
        C0     = r**2 - (2.0*G*M0*r) / c**2
        if C0 <= 0:
            return [0.0, 0.0, 0.0, 0.0]

        dP0_dr = -(G/c**2) * (A0 * Btrm0) / C0
        dM0_dr =  4.0*np.pi * r**2 * (eps0 / c**2)

        # R(0) and spatial derivatives R(0)', R(0)'' via chain rule on background
        R0_val, dR0_dP, d2R0_dP2, deps_dP = self.eos.get_R0_derivs(
            max(P0_fl, 1e21)
        )
        dA0_dr   = (deps_dP + 1.0) * dP0_dr
        dBt0_dr  = dM0_dr + (4.0*np.pi/c**2)*(3.0*r**2*P0_fl + r**3*dP0_dr)
        dC0_dr   = 2.0*r - (2.0*G/c**2)*(dM0_dr*r + M0)
        d2P0_dr2 = -(G/c**2)*((dA0_dr*Btrm0 + A0*dBt0_dr)*C0
                               - A0*Btrm0*dC0_dr) / C0**2

        R0p  = dR0_dP * dP0_dr                               # R(0)'
        R0pp = d2R0_dP2*(dP0_dr**2) + dR0_dP*d2P0_dr2       # R(0)''

        # ----------------------------------------------------------------
        # Modified star: fluid + optional magnetic
        # ----------------------------------------------------------------
        if self._magnetic and P_eff > 0:
            rho, eps_fl, B, B_mag = self._get_magnetic(P_eff)
            eps_tot = eps_fl + B_mag
        else:
            rho, eps_fl = self.eos.get_rho_eps(max(P_eff, 0.0))
            eps_tot = eps_fl
            B_mag   = 0.0

        rho_eff = rho + B_mag/c**2       # effective mass density

        # Compactness factor for modified star
        C     = r**2 - (2.0*G*M*r) / c**2
        if C <= 0:
            return [0.0, 0.0, 0.0, 0.0]
        denom = 1.0 - 2.0*G*M/(c**2 * r)

        # GR potential gradient (uses modified-star M, P_eff)
        GR_dphi = G*(M + 4.0*np.pi*r**3*P_eff/c**2) / (r**2 * denom)

        # ----------------------------------------------------------------
        # For the UNIFIED (magnetic + f(R)) case the perturbative expansion
        # is around the magnetic GR background whose pressure is P_eff
        # (not the fluid-only P0_fl).  Re-evaluate R(0) and R'(0) there.
        # For pure f(R) (non-magnetic), P_eff = P0_fl so there is no change.
        # ----------------------------------------------------------------
        if self._magnetic and P_eff > 1e21:
            R0_val, dR0_dP, d2R0_dP2, _ = self.eos.get_R0_derivs(P_eff)
            # R0' from the magnetic GR pressure gradient (zeroth-order mag bg)
            dP_mag_dr = -(rho_eff + P_eff/c**2) * GR_dphi * (1.0 - self.kappa/3.0)
            R0p  = dR0_dP * dP_mag_dr
            R0pp = d2R0_dP2*(dP_mag_dr**2) + dR0_dP*d2P0_dr2   # d2P from fluid bg
        # else: keep R0_val, R0p, R0pp from fluid background (already computed)

        # phi bracket pressure: use P_eff for magnetic (background = mag GR)
        P_phi = P_eff if self._magnetic else P0_fl

        # ----------------------------------------------------------------
        # Das 2015 Eq. (3.7): modified mass equation
        # dM_alpha/dr = 4pi r² rho_alpha - alpha × [bracket0]
        # bracket evaluated at BACKGROUND (M0, P0_fl, rho0, R0, R0', R0'')
        # ----------------------------------------------------------------
        rho0_mass = eps0 / c**2          # background mass density (fluid GR)
        denom0    = 1.0 - 2.0*G*M0/(c**2 * r)

        das_mass_bracket = (
            8.0*np.pi * r**2 * rho0_mass * R0_val
            - (c**2 / (4.0*G)) * r**2 * R0_val**2
            + R0p * (4.0*np.pi * r**3 * rho0_mass + 3.0*M0 - (2.0*c**2/G)*r)
            - (c**2/G) * r**2 * R0pp * denom0
        )

        dM_dr = 4.0*np.pi*r**2*(eps_tot/c**2) - self.alpha*das_mass_bracket

        # ----------------------------------------------------------------
        # Das 2015 Eq. (3.8): gravitational potential gradient
        # dφ/dr = GR term (uses M_alpha, P_alpha) - alpha × bracket0/denom_alpha
        # GR term uses MODIFIED quantities; correction uses BACKGROUND quantities
        # ----------------------------------------------------------------
        # GR_dphi already computed above

        das_phi_bracket = (
            8.0*np.pi * G * r * R0_val * P_phi / c**2
            - (c**2/4.0) * r * R0_val**2
            + R0p * (2.0*c**2 - 3.0*G*M0/r + 4.0*np.pi*G*P_phi*r**2/c**2)
        )

        das_dphi = GR_dphi - self.alpha * das_phi_bracket / denom

        # ----------------------------------------------------------------
        # Das 2015 Eq. (3.9): pressure equation
        # dP_alpha/dr = -(rho_alpha + P_alpha/c²) × dφ_alpha/dr × (1-kappa/3)
        # ----------------------------------------------------------------
        dP_dr = -(rho_eff + P_eff/c**2) * das_dphi * (1.0 - self.kappa/3.0)

        return [dP0_dr, dM0_dr, dP_dr, dM_dr]

    # ------------------------------------------------------------------
    def solve(self, rho_c, r0=10.0, r_max=2e9):
        P_c_fl, eps_c = self.eos.get_P_eps_from_rho(rho_c)
        if P_c_fl <= 0:
            return None

        if self._magnetic:
            B_c       = self.eos.get_B(rho_c)
            B_mag_c   = B_c**2 / (8.0*np.pi)
            P_c_eff   = P_c_fl + B_mag_c
            eps_c_tot = eps_c + B_mag_c
        else:
            P_c_eff   = P_c_fl
            eps_c_tot = eps_c

        M0_init = (4.0*np.pi/3.0) * r0**3 * (eps_c_tot/c**2)
        y0 = [P_c_fl, M0_init, P_c_eff, M0_init]

        def event_surface(r, y):
            return y[2] - 1e15
        event_surface.terminal  = True
        event_surface.direction = -1

        try:
            sol = solve_ivp(
                fun=self.derivs,
                t_span=(r0, r_max),
                y0=y0,
                method='LSODA',
                events=event_surface,
                rtol=1e-6,
                atol=1e-8,
                max_step=1e5,
            )
        except Exception:
            return None

        if not sol.success and sol.status != 1:
            return None

        return {
            'R': sol.t[-1],
            'M': sol.y[3, -1],
            'r_profile': sol.t,
            'P_profile': sol.y[2],
            'M_profile': sol.y[3],
        }
