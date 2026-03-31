# predict_analysis_v2.py

Simulation and analysis code for:

**A Sharp Recoverability Boundary in Delayed Adaptive Feedback Systems**

## Overview

This script runs the corrected unified simulations for the paper and generates the main numerical outputs for both:

- a 1D delayed adaptive phase model
- a 2D delayed adaptive phase-velocity model

The code estimates the recoverability boundary, computes recovery-time statistics, performs a 2D parameter sweep over delay and damping, and outputs the main figures and CSV summary files.

## Main corrections implemented

Compared with the earlier version, this script includes the following corrections:

1. The delayed feedback term `K*sin(e_del)` is applied in the phase equation `dtheta`, not in `dnu`.
2. The reference phase `theta_ref = omega*t` is tracked explicitly in the 2D model.
3. The velocity noise is scaled as `sigma_nu = sigma*sqrt(2*gamma)` so that the steady-state variance of `nu` is comparable across `gamma`.
4. The same sustained recovery criterion (`T_hold = 1.0 s`) is used in both 1D and 2D analyses.
5. `MU = 0.1` is used consistently throughout.
6. The observation time is `T_OBS = 40 s`, giving `T_total = 60 s`.
7. The number of trials is `n_trials = 100`.

## Requirements

- Python 3
- numpy
- pandas
- matplotlib

## Run

```bash
python3 predict_analysis_v2.py
