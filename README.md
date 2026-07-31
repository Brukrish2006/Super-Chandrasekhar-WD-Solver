# Disentangling f(R) Curvature from Magnetic Anisotropy in Super-Chandrasekhar White Dwarfs

**Author:** Harsha Adhikary · Indian Institute of Science, Bangalore · harshaa@iisc.ac.in

---

## The Problem

The Chandrasekhar limit (~1.44 M☉) sets the classical maximum mass for a white dwarf. Yet a growing population of over-luminous Type Ia supernovae — SN 2006gz, SN 2007if, SN 2009dc — require progenitor masses of **2.0–2.8 M☉**, firmly above this limit. Two theoretical mechanisms have been proposed to explain this:

1. **Bowers–Liang magnetic anisotropy (κ)** — strong internal magnetic fields generate a tangential pressure excess that supplements degeneracy pressure, directly stiffening the equation of state.
2. **Starobinsky f(R) = R + αR² modified gravity (α)** — geometric corrections to the TOV equations from higher-order curvature provide an effective extra pressure support.

Each mechanism has been studied *in isolation* (Das & Mukhopadhyay 2015; Deb, Mukhopadhyay & Weber 2022). But a critical set of questions was **left unanswered**:

> *Are κ and α genuinely distinct physical mechanisms, or are they just two parametrizations of the same effect? If they are distinct locally, can a single mass measurement tell them apart? And if not — what observable can?*

---

## The Gap in the Literature

Prior work established the following:
- f(R) gravity *alone* can push M_max to 1.77–2.70 M☉ for large |α| (Das & Mukhopadhyay 2015)
- Magnetic anisotropy *alone* can push M_max to ~2.51 M☉ (Deb et al. 2022)
- Both were studied with a continuous Chandrasekhar EOS, never checking whether the discrete Landau-quantized microphysics changes the answer

What was missing:
1. **No analytic proof** that f(R) curvature and Bowers–Liang anisotropy are structurally distinct (vs. being redundant parametrizations)
2. **No quantification** of whether a mass measurement alone can disentangle the two — the degeneracy contour in (α, κ) space had not been mapped
3. **No degeneracy-breaking observable** identified — the field knew the problem was degenerate but had no proposed discriminator
4. **No micro/macro consistency check** — nobody had directly compared the continuous Chandrasekhar EOS against the discrete Landau EOS within the same unified solver

---

## What This Paper Does

### Result 1 — Local Distinguishability (analytic + numerical)
Using the Minimal Geometric Deformation (MGD) framework (Ovalle 2017), we prove analytically that the two force corrections scale *differently* near the stellar centre:

```
F_geom  ∝  r     (f(R) curvature force)
F_κ     ∝  r²    (Bowers–Liang anisotropic force)
```

We then verify numerically along the **full solved radial profile** of the fiducial star that this r vs r² scaling holds to within ~6% across the entire star — not merely in the r→0 limit used to derive it. This proves that including both κ and α is **not double-counting**: they are structurally distinct mechanisms.

### Result 2 — Macroscopic Degeneracy (the bad news)
We ran a 12×12 grid of full stellar structure integrations over (α, κ) ∈ [-10¹³, -10¹²] cm² × [0, 0.4]. The resulting iso-mass contours are **diagonal**: a measurement of M_max alone cannot separately constrain α and κ. The posterior in (α, κ) space forms a ridge, not a point. This is a fundamental observational limitation — a new mechanism does not automatically give a new observable.

### Result 3 — Degeneracy Breaking (the good news)
We computed the stellar radius R and dimensionless tidal deformability Λ for six configurations along the iso-M_max ≈ 2.55 M☉ contour. Despite masses fixed to within ΔM/M = 2.4%, the secondary observables vary dramatically:

| Observable | Min | Max | Variation |
|------------|-----|-----|-----------|
| Radius R | 1192 km | 1429 km | **ΔR/R = 18%** |
| Tidal Λ | 1.08×10¹¹ | 3.18×10¹¹ | **ΔΛ/Λ ≈ 100%** |

The dominant driver is the C⁻⁵ scaling of Λ: large-|α| configurations are more compact, large-κ configurations are more extended. A LISA measurement of Λ to within ~30% would be sufficient to select between the f(R)-dominated and anisotropy-dominated scenarios at this mass scale — even when the mass is only known to ±0.05 M☉.

### Result 4 — Micro/Macro Consistency Check
We computed the discrete Landau-quantized EOS from first principles and fed it directly into the unified TOV solver alongside the continuous Chandrasekhar EOS. The two M_max values agree to **0.2%** — confirming that the continuous approximation used throughout the macroscopic literature is valid even at B ~ 8.6 B_c where only ~18–20 Landau levels are populated. This check had not been performed in any prior work.

---

## Headline Numbers

```
Pure f(R) alone  (α = -3×10¹² cm², conservative):    M_max ≈ 1.44 M☉
Pure magnetic    (B₀ = 3.79×10¹⁴ G):                 M_max ≈ 2.51 M☉
Unified combined (fiducial α, κ = 0.15):              M_max ≈ 2.60 M☉
Conservative B₀  (B₀ = 10¹³ G, stability window):    M     ≈ 1.56 M☉
Degeneracy break: ΔR/R = 18%,  ΔΛ/Λ ≈ 100%  at fixed M ≈ 2.55 M☉
Micro/macro gap:  0.2%  (Landau vs. Chandrasekhar EOS)
```

---

## Repository Structure

```
ROOT/
├── constants.py                  # CGS physical constants
├── eos.py                        # Chandrasekhar + Landau + hybrid EOS
├── tov_solver.py                 # Unified TOV + f(R) + Riccati tidal solver
│
├── phase0_quick.py               # 12×12 grid scan → figure_phase0_quick.png
├── phase3_lambda.py              # Tidal Λ along iso-Mmax → Table 2, Figure 8
├── phase4_convergence_v2.py      # Convergence of ΔΛ/Λ signal → Figure 9
│
├── plot_fig1_mass_radius.py      # Figure 1: M–R relations
├── plot_fig2_contour.py          # Figure 2: (α, κ) degeneracy contour map
├── plot_fig3_landau_mr.py        # Figure 3: Landau M–R curve (quantum kinks)
├── plot_fig4_synthesis.py        # Figure 4: Full micro+macro synthesis
├── plot_fig5_neutronization.py   # Figure 5: ρ_c vs M (electron-capture cutoff)
├── plot_fig6_force_scaling.py    # Figure 6: F_geom vs F_κ radial profile
├── plot_fig7_conservative_b0.py  # Figure 7: Conservative B₀ companion case
│
├── table_phase3_lambda.txt       # Raw data for Table 2
├── table_phase4_convergence.txt  # Raw convergence data
└── README.md                     # This file
```

---

## Running the Code

```bash
pip install numpy scipy matplotlib

# Reproduce Table 2 + Figure 8 (degeneracy breaking)
python phase3_lambda.py

# Reproduce convergence analysis + Figure 9
python phase4_convergence_v2.py

# Reproduce all paper figures
python plot_fig1_mass_radius.py
python plot_fig2_contour.py
python plot_fig3_landau_mr.py
python plot_fig4_synthesis.py
python plot_fig5_neutronization.py
python plot_fig6_force_scaling.py
python plot_fig7_conservative_b0.py
```

**Expected runtimes:** `phase0` ~2 min · `phase3` ~30 s · `phase4` ~3 min · figure scripts ~10 s each

---

## Key Design Decision: Decoupled Tidal Solver

The standard approach couples the tidal perturbation y(r) as a 5th ODE variable alongside the stiff TOV system. This causes repeated LSODA convergence failures in the f(R) + anisotropy regime due to the stiffness mismatch between the background fluid equations and the tidal perturbation.

This code uses a **two-pass decoupled approach**:
1. Solve the 4-variable TOV background (`compute_tidal=False`) to get P(r), M(r)
2. Fit cubic splines to the background profile
3. Solve the Riccati ODE for y(r) as an independent post-processing step using DOP853

This is numerically stable, physically equivalent, and convergence-verified across four decades of ODE tolerance.

---

## References

| Reference | Role in this work |
|-----------|-------------------|
| Das & Mukhopadhyay (2015), JCAP 05, 045 | f(R) TOV equations (perturbative strategy) |
| Deb, Mukhopadhyay & Weber (2022), ApJ 926, 66 | Magnetic anisotropy + TO orientation |
| Hinderer (2008), ApJ 677, 1216 | Tidal Love number ODE |
| Biswas & Bose (2019), PRD 99, 104002 | Anisotropic Love number correction |
| Ovalle (2017), PRD 95, 104019 | Minimal Geometric Deformation framework |
| Bowers & Liang (1974), ApJ 188, 657 | Anisotropic pressure parametrization |

---

## License

MIT License — free to use and adapt with attribution.

## Citation

If you use this code, please cite the associated paper. ArXiv preprint forthcoming.
