# WepyAnalysis

WepyAnalysis is a modular toolkit for analyzing data generated from Weighted Ensemble (WE) simulations using the [Wepy](https://github.com/ADicksonLab/wepy) framework.

The codebase is organized into five main components:
- `featurization/`: Tools for extracting structural features from WE data
- `dataset/`: Code for generating datasets from WE data
- `msm/`: Building Markov State Models (MSMs) and performing kinetic analysis
- `example/`: Example scripts for running simulations with Wepy and building MSMs

This repository is under active development and intended for researchers working with WE data, especially those using the Wepy framework.


## Installation

We recommend installing WepyAnalysis with a Python package manager such as conda or mamba. The package is tested and fully compatible with Python 3.12, and we strongly encourage using python>=3.10 for compatibility.

```
conda create -n wepy-analysis python=3.12
conda activate wepy-analysis
```

Once your python environment is ready, `wepy-analysis` can be installed with `pip` as follows:

```
pip install git+https://github.com/ADicksonLab/wepy-analysis
```

which will also install all dependencies. The installation procedure takes less than 10 seconds to complete at a local desktop.

## Dependencies:

```
[Wepy](https://github.com/ADicksonLab/wepy) >= 1.2
[geomm](https://github.com/ADicksonLab/geomm) >= 0.3
[csnanalysis](https://github.com/ADicksonLab/CSNAnalysis) >= 0.6.0
numpy >= 2.3.1
scipy >= 1.16.0
h5py >= 3.14.0
mdtraj >= 1.11.0
scikit-learn  >= 1.7.0
deeptime >= 0.4.5
```

## More Information
- Example dataset files can be found at current [Zenodo DOI](https://zenodo.org/records/15361245)
- This repository is a part of the preprint ["Determinants of Improved CGRP Peptide Binding Kinetics Revealed by Enhanced Molecular Simulations"](https://www.biorxiv.org/content/10.1101/2025.06.13.659569v1) and can be used to reproduce the results of it. 



