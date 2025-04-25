# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.0.4] - 2024-12-12
### Added
- Add measurement preprocessing in `init_datacube`, including negative value correction and value normalization so the averaged diffraction pattern has max value ~ 1.

## [v0.0.3] - 2024-12-02
### Added
- Add `load_np` to be consistent with `PtyRAD`
- Add average and std iteration time calculation printing at the end of the p74DSTEM ptycho solver for easier performance benchmarking

## [v0.0.2] - 2024-10-26
### Added
- Allow py4DSTEM results to be saved just like PtyRAD with tiff files of object, probe, propagated probe
- Add `save_result` and `result_modes` for finer controls like list of selected output, cropping, and bit depth while saving py4DSTEM results
- Add `utils_CHL.py` module for more utility and wrapper funcitons 
- Create wrapper functions like `init_datacube` with same measurements preprocessing workflow as PtyRAD and `init_ptycho`
### Changed
- Extract the custom saving logic from `mixedstate_ptychography` and `mixedstate_multislice_ptychography` into `save_results()` and move it to `utils_CHL.py` module
- Refine `make_output_folder` and include the `update_step_size` and `kzf`

## [v0.0.1] - 2024-10-09
### Added
- Add `save_iters` and `output_path`to `mixedstate_ptychography.py` and `mixedstate_multislice_ptychography.py` to save results at certain iterations. The object, probe, and position_px are all saved.
- Add `iter_times` using cuda.Event for timing. The `iter_times` are printed at each iteration and saved with the model output
- Add print to `__init__.py` to specify it's loaded from a local editable repo