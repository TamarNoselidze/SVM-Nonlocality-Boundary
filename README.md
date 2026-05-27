# Machine Learning and the LHS Boundary: An SVM Approach to Quantum Steerability

This repository contains the source code, datasets, and trained Machine Learning models for the M1 Quantum Information project: **Machine Learning and the LHS Boundary: An SVM Approach to Quantum Steerability**, developed at Sorbonne Université in collaboration with UNICAMP.

## Overview

Quantum steerability represents an asymmetric correlation intermediate between entanglement and Bell nonlocality. Analytically certifying the geometric boundary between steerable and unsteerable (Local Hidden State, or LHS) states is a computationally demanding task, often relying on slow Linear Programming (LP) algorithms. 

Instead of relying on deep neural networks acting as "black boxes" on measurement statistics, this project uses **Support Vector Machines (SVMs) trained directly on the density matrices of two-qubit systems**. 

By compressing the state space into a 9-dimensional Fano feature representation and engineering a highly targeted training dataset near the LHS boundary, we successfully:
1. Classify quantum states significantly faster than traditional LP methods.
2. Recover the exact geometric contour of steerability (verified via Werner states).
3. Extract a novel, explicit algebraic formula ($d=2$ polynomial) that natively captures the physics of quantum nonlocality.

## Repository Structure

The repository is roughly divided into two pipelines: **Data Generation** (Julia) and **Machine Learning** (Python).

```text
├── DATA/                       # HDF5 files containing generated states and labels
├── images/                     # Output plots and diagrams
├── julia_scripts/              # All the Julia scripts
│   ├── CAPIBARA.jl/                    # Module for LP labelling using polytopes
│   ├── generate_and_label.jl           # Generating and labeling entangled states simultaneously
│   ├── generate_boundary_lhs.jl        # Generates LHS states on the boundary 
│   ├── generate_boundary_nonlhs.jl     # Generates spread-out non-LHS states near the boundary
│   ├── generate_random_ent_states.jl   # Generates a given number of entangled states
|   └── test_steerability.jl            # Labels the entangled states
├── models/                     # Saved SVM models (.pkl)
├── python_ml/                  # Python scripts for SVM training and evaluation
│   ├── train_initial.py             # Trains the SVM and extracts the analytical formula
│   ├── convert_to_T.py              # Helper functions for 9D Fano feature extraction
│   └── evaluate_werner.py           # Plots the model's decision function against Werner states
└── README.md


## Dependencies

### Julia (Data Generation & Labelling)

The dataset generation relies on geometric state approximation via spherical polytope coverings. You will need Julia installed with the following packages:

- `MosekTools.jl` (Requires a valid Mosek license)
- `QuantumInformation.jl`
- `LinearAlgebra`
- `HDF5.jl`

### Python (Machine Learning)

The SVM training and feature extraction are done in Python. We recommend using a virtual environment (e.g., `conda` or `venv`).

```bash
pip install numpy scipy scikit-learn matplotlib h5py joblib
```

---

## Usage

### 1. Generating the Dataset

To generate the targeted boundary states, run the Julia scripts. The scripts use the depolarising map to mathematically push uniformly random entangled states precisely against the LHS boundary bounds ($v_{\text{lower}}$ and $v_{\text{upper}}$).

```bash
julia julia_generation/generate_boundary_lhs.jl [BatchID]
julia julia_generation/generate_boundary_nonlhs.jl [BatchID]
```

This will output `.h5` files containing the $4 \times 4$ density matrices and their respective binary labels (+1 for LHS, -1 for Non-LHS).

### 2. Training the Model & Extracting the Formula

The Python pipeline loads the `.h5` files, applies a Singular Value Decomposition (SVD) on the Fano correlation matrix to reduce the states to 9 invariant features, and trains the SVM.

```bash
python python_ml/train_initial.py
```

Running this script will output the model's accuracy, classification report, and automatically print the Analytical Nonlocality Formula (the extracted SVM weights from the $d=2$ polynomial kernel).

### 3. Evaluating on Werner States

To verify the physical validity of the model, you can test the trained SVM against the Werner state family.

```bash
python python_ml/evaluate_werner.py
```

This will generate a plot comparing the model's decision boundary $f(p)$ to the exact theoretical truth ($p = 0.5$), demonstrating the geometric superiority of the Phase 2 targeted dataset.

---

## Key Results

- **Dimensionality Reduction:** Successfully mapped 32-dimensional flattened density matrices to a 9-dimensional vector $[a_1, a_2, a_3, b_1, b_2, b_3, s_1, s_2, s_3]$ while retaining all necessary non-local information.

- **Geometric Accuracy:** The Phase 2 targeted model accurately identified the Werner state steerability threshold at $p \approx 0.508$, drastically reducing the geometric error of baseline models.

- **Interpretable Physics:** The extracted quadratic formula autonomously learned that steerability is fundamentally asymmetric (heavily driven by Bob's local purity) and dominated by the invariant coupled correlations ($s_i s_j$).

---

## Author and Acknowledgments

**Tamar Noselidze**  
M1 Quantum Information, Department of Physics, Sorbonne Université, Paris, France.

Supervised by **Dr. Rafael Rabelo**, Instituto de Física Gleb Wataghin, Universidade Estadual de Campinas (UNICAMP), Brazil.