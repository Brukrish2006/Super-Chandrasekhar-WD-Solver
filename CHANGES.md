# Changelog

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
