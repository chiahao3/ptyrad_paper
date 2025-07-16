# Repo for reporducing "PtyRAD: A High-performance and Flexible Ptychographic Reconstruction Framework with Automatic Differentiation"
This repo organizes the params files and scripts needed to reconstruct the results used in the paper. Raw data and reconstructed output needed for figures are provided in the [Zenodo record](https://doi.org/10.5281/zenodo.15273176).

# Requirements
- PtyRAD
- py4DSTEM
- PtychoShelves (fold_slice)

# Installation
- For PtyRAD installation, please refer to the [PtyRAD GitHub repo](https://github.com/chiahao3/ptyrad)
- For PtychoShelves, you'll need a Matlab license, and install from fold_slice using commit `d9a1204` from the original “main” branch. Our results were executed with MATLAB 2021a.
- For py4DSTEM, we use version 0.14.18, specifically the commit `baf9e30` from the original “dev” branch. The environment was built with Python 3.11.10, CuPy 13.3.0, and CUDA 11.8. We introduced several lightweight modifications to the package—such as results saving, timing utilities, and wrapper functions to facilitate systematic reconstruction from input files. These changes do not modify the core algorithms or affect performance timing. Our modified version is openly available at https://github.com/chiahao3/py4DSTEM/tree/benchmark, and an installation guide is provided below:
  - Download the py4DSTEM repo from `benchmark` branch and cd into the working directory
    ```bash
    conda update conda
    conda create -n py4dstem python=3.11
    conda activate py4dstem
    conda install -c conda-forge cudatoolkit=11.0 cudnn=8.1 pymatgen
    pip install cupy-cuda11x 
    pip install -e .
    ```

# Folder Structure
- 00_data: 4D-STEM data. **14.4 GB.**
- 01_params: Example params files for reconstructions.
- 02_scripts: Slurm and python/matlab scripts used to initiate the reconstructions.
- 03_output: Reconstructed outputs, hypertune databases, logs. **24.0 GB.**
- 04_plotting: Notebooks used to generate every figure except schematics (Fig 1 and Fig S1).
- 05_figures: Generated figures.

# Workflow
This repo is organized in a way that it conceptually reproduces the workflow, but since reconstruction can take quite some time, you can directly run the notebooks in `04_plotting/` to generate the figures in `05_figures/` via the reconstructed results included in `03_output/`.

If you're interested in reproducing the results, make sure you've downloaded the data from our Zenodo record and place them under `00_data/`, then check the directory and path in `01_params/`, lastly you may use the scripts in `02_scripts/` to reconstruct the results.

# Reproducibility Note
Because this manuscript was prepared over eight months of active developement of PtyRAD (including revisions), some results were generated with earlier versions of PtyRAD. For maximal reproducibility of manuscript figures, we have provided the exact output files and their corresponding parameter files in their output folders in `03_output/`. For demonstration purpose, the parameter files in `01_params/` have been updated to work with the current latest version of PtyRAD (v0.1.0b9 as of 2025-07-16), but we have not retroactively updated all historical parameter files.