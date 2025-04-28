# Plasma PINN Project: Reconstructing Turbulent Fields

## Overview

This project aims to **reproduce**, **cultivate**, and **propose new ideas** (working or not) to the methodologies used in:

> Mathews et al., \"Turbulent Field Fluctuations in Gyrokinetic and Fluid Plasmas\" (2021).

The original paper investigates how Physics-Informed Neural Networks (PINNs) can infer turbulent electric field structures in plasma based on partial observations.  
Our goal is to **recreate their data preparation**, **solve for electric potential fields**, and **set up training pipelines** to match the figures and results shown in the paper.

## Proposed Extensions

As a potential improvement, we propose investigating the impact of **adaptive loss weighting** inside the PINN, dynamically balancing the physics loss and data loss based on local error magnitudes. This could enhance model robustness, especially in regions where physical constraints are more dominant than observational data.

Other possible ideas include:

- Using alternative neural network architectures, such as Fourier feature mappings.
- Adding controlled noise to the training inputs to improve generalization.
- Testing different collocation strategies or preconditioners to accelerate convergence.

## Notes

- The focus is on **methodology reproduction** and **creative extensions**, rather than perfect reproduction of results.
- Even partial or exploratory results are acceptable as long as the methodology and reasoning are sound.
