# Stability, self-consistency and f(R) diagnostics

Code for the October 2026 manuscript *Testing Super-Chandrasekhar White Dwarf Models: Self-Consistent Magnetic Anisotropy, Radial-Mode Stability and the Viability of R + αR² Gravity*.

Run every script from inside this folder (`cd stability`). Precomputed outputs are in `results/`.

| Script | What it does |
|---|---|
| `radial_pulsation.py` | GR equilibrium + radial-pulsation solver for magnetized, Bowers–Liang anisotropic white dwarfs (α = 0). Three field-perturbation prescriptions: `eq` (field follows B(ρ)), `toroidal` (B/(ρr) conserved), `frozen43` (tangled, B ∝ ρ^{2/3}). `omega0sq(model, rho_c)` returns ω₀² [s⁻²], M [M☉], R [km]. |
| `self_consistent.py` | Volume-averaged κ_B and the fixed-point closure κ* = ⟨κ_B⟩_V; running it writes `kstar_vs_B0.json`. |
| `stability_boundaries.py` | Turning point vs ω₀² = 0 for fixed-κ sequences (Table 1 of the paper). |
| `scseq.py B0` | Self-consistent sequence at one B₀ (re-closes κ* at each ρ_c, computes ω₀² for all prescriptions). |
| `summarize_sequences.py` | Turning point, instability onset and maximum stable mass from `scseq_*.json` (Table 2). |
| `fixedtp.py` | Turning-point masses for fixed κ = 0.15. |
| `sens.py` | κ* sensitivity to (η, γ), inner cut, and Δ = B²/4π. |
| `paper_configurations.py` | ω₀² for selected configurations. |
| `pulsation_sequences.py`, `fig1data.py`, `make_figures.py` | Figure data and figures. |
| `fr_perturbative_diagnostic.py` | First-order R + αR² stars with the corrected root solver: masses, ε_α (fractional change of dP/dr), \|αR₀\|, scalaron length and growth time (Table 3). |

Validation: the non-magnetic GR sequence gives M_max = 1.4154 M☉ at ρ_c = 2.31×10¹⁰ g cm⁻³ (Mathew & Nandy 2014: 1.4166 M☉ at ≈2.3×10¹⁰), and with the `eq` prescription ω₀² = 0 coincides with the turning point to ≤0.2%.
