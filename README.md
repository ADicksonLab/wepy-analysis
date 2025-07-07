# WepyAnalysis

WepyAnalysis is a modular toolkit for analyzing data generated from Weighted Ensemble (WE) simulations using the Wepy framework.

The codebase is organized into five main components:
- `featurization/`: Tools for extracting structural features from WE data
- `dataset/`: Code for generating datasets from WE data
- `msm/`: Building Markov State Models and performing kinetic analysis
- `example/`: Example scripts for running simulations with Wepy and building MSMs

This repository is under active development and intended for researchers working with WE data, especially those using the Wepy framework.

## System Requirements:
One dependency that cannot be currently installed with pip is `csnanalysis`. 
It can be found on github [here](https://github.com/ADicksonLab/CSNAnalysis), along with installation instructions.

Other Python dependencies:
```
numpy 
scipy 
h5py
mdtraj
wepy >= 1.2
geomm >= 0.3
scikit-learn
deeptime
```

## Installation

We recommend installing WepyAnalysis with a Python package manager such as conda or mamba with a conda environment using `python=3.10` or greater. All of the codes are tested for `python=3.12`.

```
conda create -n wepy-analysis python=3.12
conda activate wepy-analysis
```

Once `csnanalysis` is installed to your python environment, `wepy-analysis` can be installed with `pip` as follows:

```
pip install git+https://github.com/ADicksonLab/wepy-analysis
```

The installation procedure takes around 5 mins to complete at a local desktop.

## Zenodo repository
Example dataset files can be found at [this link](https://zenodo.org/records/15361245). 


