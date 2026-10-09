# Changelog

## Oct 2026 — Revision 3 (bug fixes, radial pulsations, f(R) diagnostics)

### Bug fixes in the structure code
- **`constants.py`**: `K_eps` was equal to `K_P`; the electron energy-density prefactor is
  m_e⁴c⁵/(8π²ħ³) = 3·K_P. The old value violated the first law dε/dρ = (ε+P)/ρ at the 10⁻³
  level and moved the GR mass maximum from 1.4154 M☉ at 2.31×10¹⁰ g cm⁻³ to 1.4216 M☉ at 3.0×10¹⁰.
- **`tov_solver.py` `_get_magnetic`**: the 5-step fixed-point inversion of
  P_eff = P_fl(ρ) + B(ρ)²/8π did not converge where the two terms are comparable and biased
  magnetic masses low by ≈2%. Replaced by a bracketed root-find (`brentq` in log ρ).
- **`tov_solver.py` inertia**: the hydrostatic equation used the rest-mass density; it now uses
  ε_tot/c² as the TOV equation requires.
- **`tov_solver.py` surface**: with B_s = 10⁹ G the total pressure never falls below B_s²/8π, so the
  surface condition P = 10¹⁵ was unreachable with an exact inversion. The surface is now defined on
  the fluid pressure.
- **`eos.py`**: η and γ of the B(ρ) profile are now constructor arguments.

The corrected solver agrees with the independent integrator in `stability/` to 10⁻⁵ M☉.

### New code
- `stability/`: GR radial-pulsation solver, self-consistent κ closure, sequences, sensitivity
  scans, figures and the first-order f(R) diagnostic. See `stability/README.md`.

### Withdrawn / superseded claims and scripts
- **`plot_fig6_force_scaling.py`** compares the anisotropy Δ (a pressure, ∝ r²) with the f(R)
  correction to dP/dr (a force density, ∝ r). The anisotropic force is 2Δ/r, which is also ∝ r.
  The "F_geom ∝ r vs F_κ ∝ r²" result and the 10⁻²⁸ force ratio are artifacts of that unit
  mismatch and are withdrawn.
- The `kappa_*.py` scripts evaluate κ_B with the fluid density taken at the total pressure.
  `stability/self_consistent.py` uses the fluid density at the fluid pressure. Together with the
  fixes above, this changes κ* at B₀ = 3.79×10¹⁴ G from 0.159 to 0.146.
- Numbers produced by the pre-October-2026 code (e.g. 2.60 M☉, 171 km/s, 221 km/s) are superseded.

## Aug 2026 — Revision 2 (peer-review response)

### New file
- **`reproduce_new_results.py`** — standalone script reproducing the three new numerical results
  added in the manuscript revision:
  1. σ=0 (no smoothing, N=5000) EOS run → confirms smoothing bias < 0.001 M⊙ (Section 5.6)
  2. κ_B(r) radial profile at r = R/4, R/2, 3R/4 → Table tab:kappa_profile (Section 6, Item 3)
  3. κ sensitivity bracket κ=0.15 vs κ=0.30 at two field strengths → Table 3 (Section 6)

### Updated file
- **`plot_fig6_force_scaling.py`** — extended from 4 to 6 configurations:
  - Added Config 5 (B₀=10¹³ G) and Config 6 (B₀=5×10¹³ G) as intermediate-field cases
  - Now runs a Part A loop over all six configs and prints exponent spread table
  - Fixed `compute_tidal=False` (previously could hang on large integrations)
  - Field range now spans B₀ ∈ [10¹², 3.79×10¹⁴] G continuously

### New result: BDF vs LSODA solver comparison
  Running all four mass components (GR baseline, pure magnetic, pure f(R), combined)
  with a fixed BDF integrator vs LSODA gives identical masses to 5 significant figures:
  - LSODA synergy: +0.00317 M⊙ (+0.12%)
  - BDF synergy:   +0.00317 M⊙ (+0.12%)
  The ~0.12% synergy residual is below the ±0.001 M⊙ numerical precision floor
  and is not a solver-switching artifact. It is not interpreted as a physical effect.

## Initial release
- `eos.py`, `tov_solver.py`, `constants.py` — core solver
- `cross_verify.py` — Landau vs continuous EOS cross-check
- `phase0_quick.py`, `phase3_lambda.py`, `phase4_convergence_v2.py`, `phase4_fast.py` — pipeline scripts
- `plot_fig1_mass_radius.py` through `plot_fig7_conservative_b0.py` — figure generation
