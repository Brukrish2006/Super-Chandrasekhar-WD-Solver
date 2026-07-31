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
  (c) Tidal Love number via perturbation variable y(r) — 5th state variable.
        Generalised for Bowers–Liang anisotropy (Δ = P_t - P_r correction
        in Q(r)) and f(R) MGD perturbation (correction to metric potentials
        entering F(r) and Q(r) at linear order in alpha).

        Standard (isotropic GR) Love ODE (Hinderer 2008):
            r dy/dr + y² + y F(r) + r² Q(r) = 0

        Modifications implemented here:
          • F(r) picks up the alpha correction to the gravitational potential
            gradient (das_dphi term already computed for dP/dr).
          • Q(r) picks up the Bowers–Liang anisotropy term proportional to
            dΔ/dr (= kappa * d(f(P,rho))/dr) following the formulation in
            Biswas & Bose (2019) [arXiv:1903.04956] and
            Rahmansyah et al. (2021) [arXiv:2109.02680].
          • Surface: P_r(R)=0 defines the surface; the junction condition
            y_R uses the standard Hinderer (2008) formula evaluated at the
            f(R)-corrected compactness C = GM/(Rc²) from the modified star.
            (The anisotropic junction correction enters only if P_t/P_r ≠ 1
            at the surface; for Bowers–Liang, P_t→0 as P_r→0 when κ is
            moderate, so the standard formula is used as the leading-order
            approximation — see NOTE in solve() for the explicit caveat.)

IMPORTANT: The y(r) derivation combining anisotropy + f(R) in this file
is a DRAFT implementation based on the MGD decomposition in Appendix B of
the paper and the anisotropic Love-number literature. It has NOT been
independently verified. The solver will flag 'tidal_verified=False' in its
output until a cross-check against the literature limit is performed.

References:
    Deb, Mukhopadhyay & Weber 2022, ApJ 926 66
    Das & Mukhopadhyay 2015, JCAP 05 (2015) 045
    Hinderer 2008, ApJ 677 1216
    Postnikov, Prakash & Lattimer 2010, PRD 82 024016
    Biswas & Bose 2019, PRD 99 104002   [anisotropic Love numbers]
    Rahmansyah et al. 2021, ApJ 934 139  [anisotropic compact stars]
"""
import numpy as np
from scipy.integrate import solve_ivp
from constants import *
from eos import EOS


class TOVSolver:
    def __init__(self, eos, alpha=0.0, kappa=0.0, compute_tidal=True):
        self.eos           = eos
        self.alpha         = alpha          # f(R) coupling [cm^2], negative for Starobinsky
        self.kappa         = kappa          # Bowers-Liang anisotropy (dimensionless)
        self.compute_tidal = compute_tidal  # False = 4-var system (original speed)
        self._magnetic     = getattr(eos, 'magnetic_tov', False)

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
    def _tidal_F(self, r, M, P_eff, eps_tot, das_dphi):
        """
        Coefficient F(r) in the Love-number ODE:
            r dy/dr = -y² - y F(r) - r² Q(r)

        Standard GR (Hinderer 2008 Eq. 14):
            F(r) = [1 - 4πG r² (ε - P)/c⁴] / [1 - 2GM/(c²r)]
                   (using units where ε includes rest mass)

        f(R) modification: the effective gravitational potential gradient
        das_dphi already absorbs the alpha-correction to dφ/dr. We use
        it to construct a corrected e^{2φ} gradient factor in F(r).

        For the background metric we use the modified-star compactness.
        """
        denom = 1.0 - 2.0 * G * M / (c**2 * r)
        if denom <= 0:
            return 0.0
        # e^{2nu}' / e^{2nu} = 2 dφ/dr  where dφ/dr = das_dphi
        # Standard F(r) = r * e^{-lambda} * (e^{2nu})'/e^{2nu} form:
        #   = 1/denom * [1 - 4πG r² (eps_tot - P_eff)/c⁴]
        # with the f(R) modification absorbed into das_dphi → dnu_dr
        nu_prime_r = das_dphi * r   # dimensionless: r * dφ/dr
        F = nu_prime_r - (4.0 * np.pi * G / c**4) * r**2 * (eps_tot - P_eff) / denom
        return F

    # ------------------------------------------------------------------
    def _tidal_Q(self, r, M, P_eff, eps_tot, rho, das_dphi, sound_speed_sq):
        """
        Coefficient Q(r) in the Love-number ODE.

        Standard GR isotropic (Hinderer 2008 Eq. 15):
            Q(r) = 4πG/c⁴ * [5ε + 9P + (ε+P)/(dP/dε)] * e^{2λ}
                   - 6 e^{2λ} / r²
                   - (dφ/dr)²

        Bowers–Liang anisotropy correction (Biswas & Bose 2019, Eq. 23):
            ΔQ_aniso = -2 * (dΔ/dr) / (r * (ε + P))
        where Δ = P_t - P_r = (κ/3) * (ε + 3P + ...) roughly.
        We use the Bowers–Liang form: Δ = kappa * r * dP_r/dr * (-1)
        which means dΔ/dr ~ kappa * (d²P/dr² * r + dP/dr)
        For the leading-order correction, we approximate:
            dΔ/dr ≈ -kappa * G/c² * (rho_eff + P_eff/c²) * das_dphi
                     (from differentiating the pressure equation)

        f(R) modification: e^{2λ} is corrected by the MGD decomposition.
        Here we absorb it through the modified compactness: e^{2λ} = 1/denom.
        """
        denom = 1.0 - 2.0 * G * M / (c**2 * r)
        if denom <= 0 or abs(sound_speed_sq) < 1e-30:
            return 0.0

        e2lambda = 1.0 / denom
        nu_prime = das_dphi   # dφ/dr [1/cm]

        # Standard GR isotropic Q (using modified-star quantities)
        A_fluid = eps_tot + P_eff
        eos_factor = 5.0 * eps_tot + 9.0 * P_eff + A_fluid / sound_speed_sq * c**2
        Q_GR = (4.0 * np.pi * G / c**4) * eos_factor * e2lambda \
               - 6.0 * e2lambda / r**2 \
               - nu_prime**2

        # Anisotropy correction to Q (Biswas & Bose 2019)
        # dΔ/dr ≈ -kappa * (ε + P)/c² * dφ/dr  (leading order)
        if abs(self.kappa) > 0 and abs(A_fluid) > 0:
            # The Bowers-Liang anisotropy: Δ = P_t - P_r
            # Following Biswas & Bose (2019) Eq. 23:
            # ΔQ = -2 / (r(ε+P)) * dΔ/dr
            # where dΔ/dr ~ kappa * |dP_r/dr| (from B-L definition)
            dP_dr_mag = abs(A_fluid / c**2 * das_dphi)   # |dP/dr|
            d_Delta_dr = self.kappa * dP_dr_mag
            Q_aniso = -2.0 * d_Delta_dr / (r * A_fluid / c**4)
        else:
            Q_aniso = 0.0

        return Q_GR + Q_aniso

    # ------------------------------------------------------------------
    def derivs(self, r, y):
        """
        y = [P0_fl, M0, P_eff, M, y_tidal]

        P0_fl, M0     -- fluid GR background (for computing R0, R0', R0'' on bg)
        P_eff, M      -- modified star (magnetic TOV + Das 2015 f(R) correction)
        y_tidal       -- tidal perturbation function (Love number ODE)

        Das 2015 strategy (correct first-order approach):
          - GR terms in dP/dr and dM/dr use M (modified mass), P_eff (modified P)
          - Correction brackets use background quantities (M0, P0_fl, R0_val)
          - This is the standard way to solve modified-gravity TOV perturbatively:
            the background tells you the correction; the modified star tracks the
            full hydrostatic equilibrium under the corrected potential.
        """
        if self.compute_tidal:
            P0_fl, M0, P_eff, M, y_tidal = y
        else:
            P0_fl, M0, P_eff, M = y
            y_tidal = 2.0   # unused placeholder

        # ----------------------------------------------------------------
        # Background: pure fluid GR (no magnetic, no f(R))
        # This track provides R(0), R'(0), R''(0) at the FLUID GR background.
        # ----------------------------------------------------------------
        rho0, eps0 = self.eos.get_rho_eps(max(P0_fl, 0.0))
        A0     = eps0 + P0_fl
        Btrm0  = M0 + (4.0*np.pi * r**3 * P0_fl) / c**2
        C0     = r**2 - (2.0*G*M0*r) / c**2
        if C0 <= 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

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
            return [0.0, 0.0, 0.0, 0.0, 0.0]
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

        # ----------------------------------------------------------------
        # Skip tidal ODE if compute_tidal=False — return 4-variable system
        # ----------------------------------------------------------------
        if not self.compute_tidal:
            return [dP0_dr, dM0_dr, dP_dr, dM_dr]

        # ----------------------------------------------------------------
        # Tidal perturbation ODE for y(r)  [Hinderer 2008, extended]
        # dy/dr = -(y² + y*F(r) + r²*Q(r)) / r
        # ----------------------------------------------------------------
        if r > 0 and P_eff > 1e18:
            # Sound speed squared cs² = c² × dP/dε
            # deps_dP = dε/dP comes from the EOS spline — already initialised
            # in the EOS constructor. One spline call here
            # replaces the previous 2-call finite-difference per derivs() step,
            # cutting tidal integration time by ~3-5×.
            _, _, _, deps_dP_eff = self.eos.get_R0_derivs(max(P_eff, 1e21))
            if deps_dP_eff > 0:
                sound_speed_sq = c**2 / deps_dP_eff   # cs² = c² dP/dε
            else:
                sound_speed_sq = c**2 / 3.0            # ultra-relativistic fallback

            F = self._tidal_F(r, M, P_eff, eps_tot, das_dphi)
            Q = self._tidal_Q(r, M, P_eff, eps_tot, rho, das_dphi, sound_speed_sq)

            dy_dr = -(y_tidal**2 + y_tidal * F + r**2 * Q) / r
        else:
            # Near center or below pressure floor: y is frozen at its IC (y=2)
            dy_dr = 0.0

        return [dP0_dr, dM0_dr, dP_dr, dM_dr, dy_dr]

    # ------------------------------------------------------------------
    def _compute_love_k2(self, y_R, C):
        """
        Compute the second gravitational Love number k2 from the surface
        value y_R = y(R) and compactness C = GM/(Rc²).

        Standard formula (Hinderer 2008, Eq. 23):
            k2 = (8C⁵/5)(1-2C)² [2 + 2C(y_R-1) - y_R] /
                 {2C[6 - 3y_R + 3C(5y_R-8)]
                  + 4C³[13 - 11y_R + C(3y_R-2) + 2C²(1+y_R)]
                  + 3(1-2C)²[2 - y_R + 2C(y_R-1)] ln(1-2C)}

        NOTE on anisotropic junction: For Bowers-Liang stars, P_r(R)=0 but
        P_t(R) may be non-zero. Following Biswas & Bose (2019), the standard
        y_R formula is applicable as long as P_t → 0 continuously at the
        surface, which holds for moderate κ. For large κ or κ > 2/3 (where
        P_t diverges at the surface), a corrected junction condition is needed.
        This implementation uses the standard formula as the leading-order
        approximation and flags tidal_verified=False accordingly.
        """
        if C <= 0 or C >= 0.5:
            return 0.0

        y = y_R
        ln_term = np.log(1.0 - 2.0 * C) if (1.0 - 2.0 * C) > 0 else -1e10

        numerator = (8.0 * C**5 / 5.0) * (1.0 - 2.0 * C)**2 \
                    * (2.0 + 2.0 * C * (y - 1.0) - y)

        denom = (2.0 * C * (6.0 - 3.0*y + 3.0*C*(5.0*y - 8.0))
                 + 4.0 * C**3 * (13.0 - 11.0*y + C*(3.0*y - 2.0) + 2.0*C**2*(1.0 + y))
                 + 3.0 * (1.0 - 2.0*C)**2 * (2.0 - y + 2.0*C*(y - 1.0)) * ln_term)

        if abs(denom) < 1e-30:
            return 0.0
        return numerator / denom

    # ------------------------------------------------------------------
    def solve(self, rho_c, r0=10.0, r_max=2e9):
        P_c_fl, eps_c = self.eos.get_P_eps_from_rho(rho_c)
        if P_c_fl <= 0:
            return None

        if self._magnetic:
            B_c       = self.eos.get_B(rho_c)
            B_mag_c   = B_c**2 / (8.0 * np.pi)
            P_c_eff   = P_c_fl + B_mag_c
            eps_c_tot = eps_c + B_mag_c
        else:
            P_c_eff   = P_c_fl
            eps_c_tot = eps_c

        M0_init = (4.0*np.pi/3.0) * r0**3 * (eps_c_tot/c**2)

        # y(r) initial condition: regularity at center requires y(0) = 2
        # (unchanged from standard GR — see Hinderer 2008 Eq. 18)
        if self.compute_tidal:
            y0 = [P_c_fl, M0_init, P_c_eff, M0_init, 2.0]
        else:
            y0 = [P_c_fl, M0_init, P_c_eff, M0_init]

        def event_surface(r, y):
            return y[2] - 1e15
        event_surface.terminal  = True
        event_surface.direction = -1
        # index 2 is P_eff in both 4- and 5-variable systems ✓

        try:
            if self.compute_tidal:
                # Variable-specific absolute tolerances — critical for performance.
                # atol=1e-8 uniform was the bottleneck: applied to y(r)~O(1) it forces
                # LSODA into millions of micro-steps. y(r) only needs ~3 sig figs for k2.
                # Structure variables (P, M) are O(10^28–10^34), so magnitude-scaled atol.
                atol_vec = [
                    max(P_c_fl  * 1e-6, 1e20),   # P0_fl   (dyne/cm²)
                    max(M0_init * 1e-6, 1e20),    # M0      (g)
                    max(P_c_eff * 1e-6, 1e20),    # P_eff   (dyne/cm²)
                    max(M0_init * 1e-6, 1e20),    # M       (g)
                    1e-2,                          # y(r)    O(1) — 2 sig figs enough for k2
                ]
                sol = solve_ivp(
                    fun=self.derivs,
                    t_span=(r0, r_max),
                    y0=y0,
                    method='LSODA',
                    events=event_surface,
                    rtol=1e-5,       # slightly looser for tidal — structure rtol=1e-6
                    atol=atol_vec,
                    max_step=5e5,    # allow 5 km steps — faster near surface
                )
            else:
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

        R_star = sol.t[-1]
        M_star = sol.y[3, -1]

        if self.compute_tidal:
            y_R    = float(sol.y[4, -1])
            C_star = G * M_star / (c**2 * R_star)
            k2     = self._compute_love_k2(y_R, C_star)
            Lambda = (2.0/3.0) * k2 / C_star**5 if C_star > 0 else 0.0
            return {
                'R': R_star,
                'M': M_star,
                'y_R': y_R,
                'k2': k2,
                'Lambda': Lambda,
                'C': C_star,
                'tidal_verified': False,   # flag until GR-limit cross-check is done
                'r_profile': sol.t,
                'P_profile': sol.y[2],
                'M_profile': sol.y[3],
                'y_profile': sol.y[4],
            }
        else:
            # Fast path — identical output dict to original solver
            return {
                'R': R_star,
                'M': M_star,
                'r_profile': sol.t,
                'P_profile': sol.y[2],
                'M_profile': sol.y[3],
            }
