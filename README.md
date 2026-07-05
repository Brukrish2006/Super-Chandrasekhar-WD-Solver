# Unveiling the Microscopic-Macroscopic Interplay in Magnetized White Dwarfs

This repository contains Python codes to solve the Tolman-Oppenheimer-Volkoff (TOV) equations for strongly magnetized white dwarfs, addressing a critical **research gap** in modern theoretical astrophysics. 

By unifying microscopic quantum mechanics (discrete Landau levels) with macroscopic modified gravity ($f(R)$ models), these scripts reproduce and expand upon current literature, yielding fully self-consistent Mass-Radius (M-R) profiles.

## The Research Gap

A vast amount of recent literature explores the structure of highly magnetized white dwarfs (so-called "super-Chandrasekhar" white dwarfs) to explain over-luminous Type Ia supernovae. However, previous theoretical approaches typically suffer from a severe dichotomy:

1. **Macroscopic Modified Gravity Models** often utilize smooth, continuous polytropic Equations of State (EOS) or continuous Fermi-Dirac integrals, ignoring the microscopic quantization of electron orbits in ultra-strong magnetic fields.
2. **Microscopic Quantum Models** compute the discrete Landau energy levels but fail to integrate these stiff, non-linear thermodynamic properties into modified GR architectures ($f(R)$ gravity) and anisotropic magnetic stress equations.

### Our Unified Approach
These codes bridge this gap by explicitly calculating the **discrete Landau quantized EOS** and feeding it into a robust ODE integrator capable of handling the extreme stiffness of the resulting TOV equations in $f(R)$ gravity.

### Key Finding: The 1D Topological Inversion
When running these simulations, a profound structural deviation emerges. In standard non-magnetic models, stars shrink as they gain mass ($R \propto M^{-1/3}$, a 3D Fermi gas). 

However, under extreme magnetic quantization ($B \gtrsim B_c$), low-density electrons are trapped entirely in the ground Landau level ($\nu = 0$). The electron gas becomes effectively **1-dimensional** ($n=0.5$ polytrope). Unlike standard stars, a 1D star *expands* as it gains mass ($R \propto M^{1/5}$). 

This forces the Mass-Radius curve to undergo a topological "turn over" (bending backwards towards the origin at low masses), characterized by sharp macroscopic "kinks" every time a new microscopic Landau level is populated. Our numerical solver fully resolves this structure, tracing both the stable sequence and the unstable gravitational branch beyond the maximum mass peak to perfectly replicate theoretical expectations.

## Repository Structure

The code is highly modularized, ensuring that every physical component can be tested, verified, and plotted independently:

* `constants.py`: Defines all fundamental physics constants in CGS units, including the critical magnetic field $B_c$.
* `eos.py`: The thermodynamic engine. Computes the standard Chandrasekhar EOS, the highly stiff Discrete Landau Quantized EOS, and a hybrid continuous-to-discrete EOS.
* `tov_solver.py`: The ODE integrator. Uses SciPy's `LSODA` method to solve the stiff differential equations for modified $f(R)$ gravity and magnetic pressure anisotropy.
* `cross_verify.py`: Verification script to test numerical outputs against established literature benchmarks (e.g., Das 2015, Deb et al. 2021).

### Plotting Scripts
Each plotting script focuses on generating a distinct scientific figure:

* `plot_fig1_mass_radius.py`: Demonstrates the distinct M-R curves across Standard GR, Pure $f(R)$, Pure Magnetic, and the Unified model.
* `plot_fig2_contour.py`: Sweeps a 12x12 parameter space of magnetic anisotropy ($\kappa$) and modified gravity ($\alpha$) to map the maximum stable mass limit.
* `plot_fig3_landau_mr.py`: Isolates the microscopic effect, explicitly showing the topological kinks and the 1D quantum inversion of the Landau EOS.
* `plot_fig4_synthesis.py`: A combined plot showing how the discrete Landau EOS modifies the standard macroscopic solutions.
* `plot_fig5_neutronization.py`: Plots Central Density ($\rho_c$) against Mass, charting the stability threshold prior to neutronization/inverse beta decay.

## Usage

Simply run any of the plotting scripts to generate their respective high-resolution `.png` figures:

```bash
python plot_fig3_landau_mr.py
```
