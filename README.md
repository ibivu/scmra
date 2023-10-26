# Single Cell Network Reconstruction

This repository contains a python module for single cell network reconstruction based on Modular Response Analysis and Comparative Network Reconstruction.
The repository includes two methods:

  - **scMRA**: reconstructs one signaling network from single cell (phospho)-protein count data
  - **scCNR**: builds on scMRA for comparative network reconstruction of a set of cell populations (e.g., cell states).
               ScCNR reconstructs one shared signaling network with population-specific interaction strengths
   
## Method

![Method](https://github.com/ibivu/scmra/blob/main/docs/scCNR_scheme.pdf)

ScMRA/ scCNR are methods to infer signalling networks from single-cell data of active (e.g. phospho) and total protein abundances. 
The methods exploits the stochastic variability between single cells to identify edges in the interaction network. 
Abundances of total protein are used as ’natural perturbation experiments’ to infer the network. Therefore, the methods work well
even in the absence of perturbations. To improve network reconstruction the optimization can easily be complemented with a prior network/ perturbation data.

## Installation:

- To solve the optimization problem we use IBM ILOG CPLEX solver, which is freely available for academic purposes:
[IBM CPLEX DOWNLOAD](https://www.ibm.com/support/pages/downloading-ibm-ilog-cplex-optimization-studio-2010)

- Install our python module
```bash
  python3 setup.py install
```

## Tutorial

[1. Running scMRA to reconstruct a model network of the MAPK pathway](https://github.com/ibivu/scmra/blob/main/docs/scMRA_Tutorial.html)

[2. Running scCNR to comparatively reconstruct differences in the pathway between the wild type and a BRAF mutant population](https://github.com/ibivu/scmra/blob/main/docs/scCNR_Tutorial.html)
