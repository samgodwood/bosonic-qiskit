# Playground

This directory my work playing around with hybrid oscillator-qubit architechture.

- **`GKP/`**: Explores Gottesman-Kitaev-Preskill (GKP) codes, including:
  - An introduction to GKP codes (`GKP_Intro.ipynb`).
  - An (incomplete) comparison of state preparation protocols for a GKP state from the vacuum using "small-big-small," "sharpen-trim," and "big-small-big" methods (`GKP_State_Prep.ipynb`).

- **`Z2Higgs/`**: Investigates hybrid-oscillator hardware for simulating strongly interacting boson-fermion systems, with:
  - Motivation for the use of this hardware (`gate_complexity.ipynb`).
  - Trotterized dynamics of the Z2 Higgs lattice in 1+1D (`dynamics/`).

- **`Rabi_Model/`**: Initial comparison of quantum simulations for a simple spin-boson Hamiltonian using all-qubit and oscillator-qubit architectures.

- **`bosonic_operator_costs/`**: Initial comparisons of noisy bosonic operations, and the cost of implementing them on an all-qubit architectures.