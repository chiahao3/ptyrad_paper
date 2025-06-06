# Repo for reporducing "PtyRAD: A High-performance and Flexible Ptychographic Reconstruction Framework with Automatic Differentiation"
This repo collects the params files and scripts needed to reconstruct the results used in the paper.

# Requirements
- PtyRAD
- py4DSTEM
- PtychoShelves (fold_slice)

## Installation
- For PtyRAD installation, please refer to the PtyRAD repo
- For PtychoShelves, you'll need a Matlab license, and install from fold_slice using commit `d9a1204` from the original “main” branch. Our results were executed with MATLAB 2021a.
- For py4DSTEM, we use version 0.14.18, specifically the commit `baf9e30` from the original “dev” branch. The environment was built with Python 3.11.10, CuPy 13.3.0, and CUDA 11.8. We introduced several lightweight modifi cations to the package—such as results saving, timing utilities, and wrapper functions to facilitate systematic reconstruction from input fi les. These changes do not modify the core algorithms or aff ect performance timing. Our modifi ed version is openly available at https://github.com/chiahao3/py4DSTEM/tree/benchmark, and an installation guide is below:
  - Download the py4DSTEM repo from benchmark branch and cd into the working directory
    ```bash
    conda update conda
    conda create -n py4dstem python=3.11
    conda activate py4dstem
    conda install -c conda-forge cudatoolkit=11.0 cudnn=8.1 pymatgen
    pip install cupy-cuda11x 
    pip install -e .
    ```
# Workflow
This repo is organized in a way that it conceptually reproduces the workflow, but since reconstruction can take quite some time, the reconstructed results are included in `03_output` so you can directly use the notebooks in `04_plotting` to generate the figures in `05_figures`.

If you're interested in reproducing the results, make sure you've downloaded the data from our Zenodo record and place them under `00_data`, then check the directory and path in `01_params`, lastly you may ise the scripts in `02_scripts` to reconstruct the results.