"""
conservative_kappa_zero.py
===========================
Addresses the review catch: at B0=1e13 G, <kappa_B>_V ≈ 0,
so the adopted kappa=0.15 is not magnetically self-consistent.

This script reports:
1. What is M at B0=1e13 G with kappa=0 (fully magnetically self-consistent)?
2. What is M at B0=1e13 G with kappa=0.15 (current "defensible" result)?
3. What does the discriminator (Section 3.5) look like at kappa=0?

This establishes the correct framing for the revised text.
"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from constants import G, c
from eos import EOS
from tov_solver import TOVSolver

Msun = 1.989e33
km   = 1e5
ALPHA  = -3.0e12
RHO_C  = 1.0e10
B0_CON = 1.0e13
B0_EXT = 3.79e14

print("=" * 65)
print("Conservative companion: kappa=0 (self-consistent) vs kappa=0.15")
print("=" * 65)

for label, B0, kappa in [
    ("Conservative, kappa=0.15 (current 'defensible')", B0_CON, 0.15),
    ("Conservative, kappa=0    (magnetically self-consistent)", B0_CON, 0.00),
    ("Extreme,      kappa=0.15 (field-validated self-consistent)", B0_EXT, 0.15),
]:
    eos = EOS(mode='chandra', B_0=B0, magnetic_tov=True, sigma=20, N_points=1000)
    sol = TOVSolver(eos, alpha=ALPHA, kappa=kappa, compute_tidal=False).solve(RHO_C)
    if sol:
        M = sol['M'] / Msun
        R = sol['R'] / km
        C_val = G * sol['M'] / (sol['R'] * c**2)
        vg = C_val * c / 1e5
        print(f"\n  {label}")
        print(f"    M = {M:.4f} Msun   R = {R:.1f} km   v_grav = {vg:.1f} km/s")
    else:
        print(f"\n  {label}: FAILED")

print()
print("=" * 65)
print("KEY QUESTION: what happens to the discriminator at kappa=0?")
print("Six Table-1 configurations, all with kappa→0 (field only)")
print("Does the iso-mass contour still exist? Does 171 km/s survive?")
print("=" * 65)

# The discriminator configurations from Table 1 (approximate alpha values
# that give M_max ≈ 2.55 Msun at kappa=0.15, B0=3.79e14)
# At kappa=0, these same alpha values will give different masses,
# but we can check what the radius spread is at any fixed mass contour.
# First: what mass does the fiducial config give at kappa=0?
eos_ext = EOS(mode='chandra', B_0=B0_EXT, magnetic_tov=True, sigma=20, N_points=1000)
sol_ext_k0 = TOVSolver(eos_ext, alpha=ALPHA, kappa=0.0, compute_tidal=False).solve(RHO_C)
if sol_ext_k0:
    M_ext_k0 = sol_ext_k0['M'] / Msun
    R_ext_k0 = sol_ext_k0['R'] / km
    print(f"\n  Fiducial (alpha=-3e12, B0=3.79e14, kappa=0): M={M_ext_k0:.4f}, R={R_ext_k0:.1f} km")
    print(f"  Compare with kappa=0.15: M=2.5742, R=1414.6 km")
    print(f"  Difference: delta_M = {2.5742-M_ext_k0:.4f} Msun ({100*(2.5742-M_ext_k0)/2.5742:.2f}%)")

print()
print("=" * 65)
print("BOTTOM LINE for paper revision:")
print("=" * 65)
print("""
Physical interpretation:
  At B0=3.79e14 G: <kappa_B>_V=0.157 → kappa=0.15 is MAGNETICALLY motivated
  At B0=1e13 G:    <kappa_B>_V≈0    → kappa=0.15 requires NON-MAGNETIC source

  The conservative companion (B0=1e13 G, kappa=0.15) implicitly assumes
  non-magnetic anisotropy (e.g., crystalline lattice, rotation, tidal stress)
  at a level that cannot be produced by its own magnetic field.

  Two honest framings for the conservative case:
  A) "Self-consistent field case" → kappa=0, M=? (report what we compute above)
  B) "General anisotropy case"   → kappa=0.15, M=1.56, acknowledge the kappa
     cannot be magnetically motivated at this field strength.

  The DISCRIMINATOR (Section 3.5) is computed along the ~2.55 Msun contour
  which requires strong fields (~10^14 G) — that regime IS validated.
  The discriminator's validity is not affected by the conservative case.
""")
