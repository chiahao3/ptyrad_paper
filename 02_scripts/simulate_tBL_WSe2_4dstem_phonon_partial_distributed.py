import warnings
warnings.filterwarnings('ignore') # Because the supercell is relatively small, Numba will keep complaining NumbaPerformanceWarning due to under utilization

import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

import ase
import abtem
import dask
import cupy as cp
from cupyx.scipy import ndimage

abtem.config.set({"local_diagnostics.progress_bar": False})
abtem.config.set({"device": "gpu"})
abtem.config.set({"dask.chunk-size-gpu" : "2048 MB"})
dask.config.set({"num_workers": 1})

############################################################

def potential_to_phase(projected_atomic_potential, acceleration_voltage):
    
    # proj_potential: V-Ang
    # acceleration_voltage: kV
    
    # Physical Constants
    PLANCKS = 6.62607015E-34 # m^2*kg / s
    REST_MASS_E = 9.1093837015E-31 # kg
    CHARGE_E = 1.602176634E-19 # coulomb 
    SPEED_OF_LIGHT = 299792458 # m/s
    
    # Useful constants in EM unit 
    hc = PLANCKS * SPEED_OF_LIGHT / CHARGE_E*1E-3*1E10 # 12.398 keV-Ang, h*c
    REST_ENERGY_E = REST_MASS_E*SPEED_OF_LIGHT**2/CHARGE_E*1E-3 # 511 keV, m0c^2
    
    # Derived values
    gamma = 1 + acceleration_voltage / REST_ENERGY_E # m/m0 = 1 + e*V/m0c^2, dimensionless, Lorentz factor
    wavelength = hc/np.sqrt((2*REST_ENERGY_E + acceleration_voltage)*acceleration_voltage) # Angstrom, lambda = hc/sqrt((2*m0c^2 + e*V)*e*V))
    sigma = 2*np.pi*gamma*REST_MASS_E*CHARGE_E*wavelength/PLANCKS**2 * 1E-20 * 1E3 # interaction parameter, 2 pi*gamma*m0*e*lambda/h^2, 1/kV-Ang
    phase_shift = np.angle(np.exp(1j*sigma * projected_atomic_potential/1E3)) # radian in strong phase approximation
    
    return gamma, wavelength, sigma, phase_shift

import sys
PATH_TO_PTYRAD = "/home/fs01/cl2696/workspace/ptyrad"  # Change this for the ptyrad package path
sys.path.append(PATH_TO_PTYRAD)
from ptyrad.utils import compose_affine_matrix

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run abTEM", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--start_idx",       type=int, required=True)
    parser.add_argument("--end_idx",         type=int, required=True)
    args = parser.parse_args()
    
    start_idx = args.start_idx
    end_idx = args.end_idx

    print(f"start_idx = {start_idx}, end_idx = {end_idx}")
    
    ############################################################
    ### Set up simulation parameters
    ############################################################

    # Set up random seed
    random_seed = 42

    # Atomic model
    mx2_formula = 'WSe2'
    mx2_phase = '2H'
    lattice_constant = 3.297
    uc_thickness = 3.376
    vacuum_layers = 2
    supercell_reps = (26,15,1) #(38, 22, 1) # ~ 120 Ang extent with orthogonalized cell

    # Phonon
    use_frozen_phonon = True
    num_phonon_configs = 25
    phonon_sigma = 0.1 # Ang

    # Potential Sampling
    lateral_sampling = 0.1494/1.5 # unit: Ang, note that kmax_antialias = 1/(3*dx), so if we want to simulate up to kmax = 4.1 1/Ang, we need 1/4.1/3 Ang sampling or slightly finer ~ 0.08 Ang 
    vertical_sampling = 1 # Ang, multislice thickness

    # Random defects
    vac_density = 0.02

    # Probe parameters
    energy = 80e3 # unit: eV
    wavelength = 0.041757 # unit: Ang, this value is only used for display useful information
    convergence_angles = 24.9 #
    df = 0 # df, unit: Ang, note the df = -C1,0, so positive defocus is underfocuse just like Kirkland and fold_slice.
    C30_list = 500 * 1e-9 * 1e10 # unit: Ang, note that we convert to m and then Ang. C30 = Cs.
    aberrations = {"C30": C30_list}

    # Temporal partial coherence
    use_partial_temporal_probe = True
    chromatic_aberration = 1 * 1e-3 * 1e10 # unit: Ang, note that we convert to m and then Ang
    energy_spread = 0.35 # unit: eV, this is the std so expected FWHM of ZLP would be 2.355*0.35 ~ 0.82 eV
    num_df_configs = 5

    # Scan configurations
    N_scan_fast, N_scan_slow = 128,128
    scan_step_size = 0.429 # Unit: Ang.
    # (scale, asymmetry, rotation, shear) = (1.005, 0.03, 1.5, 1.2)
    scan_rand_std = 0.05 # Unit: Ang

    # Spatial partial coherence
    use_partial_spatial_source = False
    source_size = 0.34 # Unit: Ang. 2.355*std = FWHM. Note that this mixes the DP along scan directions

    # Final CBED
    target_Npix = 128
    material = 'simu_tBL_WSe2'
    output_dir = f'data/paper/{material}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'output_dir = {output_dir} is created!')

    ############################################################
    ### Create the object supercell 
    ############################################################

    # Note that the "top" is in real specimen space, in abTEM the "top layer" is put at 0 and is visualized as the lower layer

    atoms_top = ase.build.mx2(formula=mx2_formula, kind=mx2_phase, a=lattice_constant, thickness=uc_thickness, vacuum=vacuum_layers) # a: lattice constant, thickness: chalcogen intralayer distance, vacuum = vacuum layer thickness. All unit in Ang.
    atoms_top.cell[-1,-1] *= 2

    atoms_bottom = ase.build.mx2(formula=mx2_formula, kind=mx2_phase, a=lattice_constant, thickness=uc_thickness, vacuum=vacuum_layers) # a: lattice constant, thickness: chalcogen intralayer distance, vacuum = vacuum layer thickness. All unit in Ang.
    atoms_bottom.cell[-1,-1] *= 2
    atoms_bottom.positions[:, 2] += 6.491 # shift WSe2 layer in the z-direction, note that +z is actually illuminated later, so abTEM is having beam propagating from 0 to +z

    atoms_top_sc = abtem.orthogonalize_cell(atoms_top) * supercell_reps # lx:ly = 1:sqrt(3)
    atoms_bottom_sc = abtem.orthogonalize_cell(atoms_bottom) * supercell_reps # lx:ly = 1:sqrt(3)

    rotation_offset = 8 # deg
    inter_twist = 183 # deg

    atoms_top_sc.rotate(rotation_offset + inter_twist, "z", rotate_cell=False)
    atoms_bottom_sc.rotate(rotation_offset, "z", rotate_cell=False)

    atoms_top_sc.positions[:, :2] += (np.diag(atoms_top_sc.cell/2)[:2] - atoms_top_sc.positions.mean(0)[:2])
    atoms_bottom_sc.positions[:, :2] += (np.diag(atoms_bottom_sc.cell/2)[:2] - atoms_bottom_sc.positions.mean(0)[:2])

    tBL_sc = atoms_top_sc + atoms_bottom_sc

    print(f'tBL_sc.cell = {tBL_sc.cell} Ang') # Unit: Ang
    print(f'Supercell tBL_sc contains {len(np.where(tBL_sc.get_atomic_numbers() == 34)[0])} Se atoms and {len(np.where(tBL_sc.get_atomic_numbers() == 74)[0])} W atoms')

    ############################################################
    ### Introduce vacancies into the chalcogen sites in the supercell
    ############################################################
    num_atoms = tBL_sc.get_global_number_of_atoms()
    Se_indices = np.where(tBL_sc.get_atomic_numbers() == 34)[0]
    num_Se_atoms = len(Se_indices)

    np.random.seed(seed=random_seed)
    vac_idx = np.random.choice(Se_indices, size = int(vac_density * num_Se_atoms), replace=False)
    print(f"Introducing {len(vac_idx)} Se vacancies")
    print(f"First 5 vac_idx = {vac_idx[:5]}")
    if vac_density > 0:
        del tBL_sc[vac_idx]
    print(f'Supercell tBL_sc contains {len(np.where(tBL_sc.get_atomic_numbers() == 34)[0])} Se atoms and {len(np.where(tBL_sc.get_atomic_numbers() == 74)[0])} W atoms')

    ############################################################
    ### Calculate the potential with or without phonon
    ############################################################

    if use_frozen_phonon:
        print(f"Using FrozenPhonons potential with {num_phonon_configs} configs")
        np.random.seed(random_seed)
        phonon_seed = np.random.randint(0,1000, num_phonon_configs)
        print(f'phonon_seed = {phonon_seed}')
        atoms = abtem.FrozenPhonons(tBL_sc, num_configs=num_phonon_configs, sigmas=phonon_sigma, seed=phonon_seed)
        potential = abtem.Potential(atoms=atoms, sampling=lateral_sampling, parametrization="lobato",
            slice_thickness=vertical_sampling, projection="finite")
        potential_arr = cp.mean(potential.build().compute(progress_bar=False).array, axis=0).transpose(0,2,1)
    else:
        print("Using Static potential")
        atoms = tBL_sc
        potential = abtem.Potential(atoms=atoms, sampling=lateral_sampling, parametrization="lobato",
            slice_thickness=vertical_sampling, projection="finite")
        potential_arr = potential.build().compute(progress_bar=False).array.transpose(0,2,1)
    print(f"potential.shape = {potential.shape}, potential_arr.shape = {potential_arr.shape}.")
    print("Note that the last 2 axes are transposed because abTEM go with (z,x,y) but we want (z,y,x)")

    ############################################################
    ### Calculate the probe
    ############################################################

    kmax_antialias = 1/lateral_sampling/3 # 1/Ang #The kmax_antialiasing = 2.675 Ang-1 
    alpha_max_antialias = wavelength * kmax_antialias # rad

    print(f"Energy = {energy/1e3} kV, rel. wavelength = {wavelength} Ang")
    print(f"CBED collection kmax = {kmax_antialias:.4f} 1/Ang, collection alpha_max = {alpha_max_antialias*1000:.4f} mrad")

    if use_partial_temporal_probe:
        focal_spread = chromatic_aberration * energy_spread / energy
        defocus_distribution = abtem.distributions.gaussian(
        center = df,
        standard_deviation=focal_spread,
        num_samples=num_df_configs,
        sampling_limit=2,
        ensemble_mean=True)
        print(f"Using partial temporal coherent probe with {len(np.array(defocus_distribution))} defoci")
        print(f"Focal spread = {focal_spread:.4f} Å")
        print(f"defocus distribution = {np.array(defocus_distribution).round(3)}")
        probe = abtem.Probe(energy=energy, semiangle_cutoff=convergence_angles, defocus=defocus_distribution, **aberrations)
    else:
        print("Using coherent probe")
        probe = abtem.Probe(energy=energy, semiangle_cutoff=convergence_angles, defocus=df,                   **aberrations)
    probe.grid.match(potential)
    print(f"probe.shape = {probe.shape}")
    print(probe.axes_metadata)

    ############################################################
    ### Make scan positions, unit in Ang
    ############################################################

    pos = scan_step_size * np.array([(y, x) for y in range(N_scan_slow) for x in range(N_scan_fast)]) # (N,2), each row is (y,x)
    pos = pos - (pos.max(0) - pos.min(0))/2 + pos.min(0) # Center scan around origin

    # Apply affine transformation
    # plot_affine_transformation(scale, asymmetry, rotation, shear)

    # Apply random jitter
    np.random.seed(random_seed)
    pos_real = pos + scan_rand_std * np.random.randn(*pos.shape)

    # pos_real = pos @ compose_affine_matrix(scale, asymmetry, rotation, shear) + scan_rand_std * np.random.randn(*pos.shape)

    # Apply offset to move the scan pattern inside the supercell
    offset = pos_real.min(0) - 15
    pos_real -= offset
    pos -= offset

    # Change dx due to the antialias kMax
    recon_dx = lateral_sampling * 1.5  

    # Parse the position into the hdf5 for reconstuction, and the abTEM scan position
    pos_ang_yx = pos_real
    pos_ang_xy = np.flip(pos_real,1)

    # Preprocess the position so that it's compatible with follow up reconstruction packages
    pos_px_yx = pos_ang_yx / recon_dx
    obj_shape = 1.2 * np.ceil(pos_px_yx.max(0) - pos_px_yx.min(0) + np.array([target_Npix, target_Npix])) # Estimate the obj_shape in px
    pos_px_yx = pos_px_yx + np.ceil((np.array(obj_shape)/2) - (np.array([target_Npix, target_Npix])/2)) # Shift back to obj coordinate

    # Visualize it in conventional orientation, although abTEM would put origin at the bottom left
    # plot_scan_positions(pos_real, init_pos=pos, dot_scale=0.1, show_arrow=False)
    print(f"First 5 positions of pos_ang_xy (Ang) = {pos_ang_xy[:5]}, this is for abTEM\n")
    print(f"First 5 positions of pos_px_yx (px) = {pos_px_yx[:5]}, this is for reconstruction packages")

    # Create custom scan pattern
    # custom_scan = abtem.CustomScan(pos_ang_xy)

    ############################################################
    ### Calculate cbeds
    ############################################################

    cbeds = probe.multislice(scan = pos_ang_xy[start_idx:end_idx], potential = potential).diffraction_patterns().reduce_ensemble().compute(progress_bar=False)
    print(f"cbeds.shape = {cbeds.shape}")
    print(f"cbeds.axes_metadata = {cbeds.axes_metadata}")

    print(f"Selected cbeds.shape = {cbeds.shape}")
    cbeds = cbeds.array

    # # Apply the partial spatial coherence
    # if use_partial_spatial_source:
    #     source_size_std_ang = source_size
    #     source_size_std_px = source_size_std_ang / scan_step_size
    #     cbeds = cbeds.reshape(N_scan_slow, N_scan_fast, *cbeds.shape[-2:])
    #     cbeds = ndimage.gaussian_filter(cbeds, sigma=source_size_std_px)
    #     print(f"\nAdding source size (partial spatial coherence) of Gaussian blur std = {source_size_std_px:.4f} scan_step sizes or {source_size_std_ang:.4f} Ang to measurements along the scan directions")
    #     cbeds = cbeds.reshape(-1,*cbeds.shape[-2:])
        
    # Resample cbeds
    cbeds_shape = np.array(cbeds.shape[-2:])
    zoom_factors = np.concatenate([[1], target_Npix / cbeds_shape])
    cbeds_resample = ndimage.zoom(cbeds, zoom=zoom_factors, order=1) # Use bilinear to prevent value overshoot
    print(f"cbeds_resample.shape = {cbeds_resample.shape}")

    # Cast cupy back to numpy
    cbeds_resample = cbeds_resample.get()

    ############################################################
    ### Resample potential and then crop it to the scan region
    ############################################################

    potential_resample = ndimage.zoom(potential_arr, zoom=(1, 2/3, 2/3), order=1).get() # Don't do any value scaling when we resample laterally because we need to keep the max value
    print(f"potential_resample.shape = {potential_resample.shape}")

    # Crop the potential based on scan position converted to reconstruction px size
    pos_recon_px_yx = pos_ang_yx / recon_dx
    y_min, y_max = np.floor(pos_recon_px_yx[:,0].min()).astype(int), np.ceil(pos_recon_px_yx[:,0].max()).astype(int)
    x_min, x_max = np.floor(pos_recon_px_yx[:,1].min()).astype(int), np.ceil(pos_recon_px_yx[:,1].max()).astype(int)

    potential_crop = potential_resample[:,y_min-1:y_max,x_min-1:x_max]
    print(f"potential_crop.shape = {potential_crop.shape}")

    # Convert potential to phase shifts
    *_, gt_phase = potential_to_phase(potential_crop, energy/1e3)

    ############################################################
    ### Parse the abtem_params as metadata
    ############################################################

    abtem_params = {'material':'tBL-WSe2',
                    'lateral_sampling':lateral_sampling,
                    'vertical_sampling':vertical_sampling,
                    'random_seed':random_seed,
                    'use_frozen_phonon':use_frozen_phonon,
                    'num_phonon_configs':num_phonon_configs,
                    'phonon_sigma':phonon_sigma,
                    'vac_density':vac_density,
                    'energy':energy,
                    'wavelength':wavelength,
                    'convergence_angles':convergence_angles,
                    'df':df,
                    'C30_list':C30_list,
                    'use_partial_temporal_probe':use_partial_temporal_probe,
                    'use_partial_spatial_source':use_partial_spatial_source,
                    'chromatic_aberration':chromatic_aberration,
                    'energy_spread':energy_spread,
                    'num_df_configs':num_df_configs,
                    'N_scan_fast':N_scan_fast,
                    'N_scan_slow':N_scan_slow,
                    'scan_step_size':scan_step_size,
                    'sc_reps':'sc_reps',
                    'vac_idx':vac_idx,
                    'num_atoms':num_atoms,
                    'kmax_antialias':kmax_antialias,
                    'alpha_max_antialias':alpha_max_antialias,
                    'target_Npix':target_Npix,
                    'pos_ang_yx':pos_ang_yx,
                    'pos_recon_px_yx':pos_recon_px_yx
                    }
    ############################################################
    ### Save the hdf5 file
    ############################################################

    potential_str = 'phonon' if use_frozen_phonon else 'static'

    # coherent_str = ''
    # if not use_partial_spatial_source and not use_partial_temporal_probe:
    #     coherent_str = '_coherent'
    # if use_partial_temporal_probe:
    #     coherent_str += '_temporal'
    # if use_partial_spatial_source:
    #     coherent_str += '_spatial'

    if use_partial_temporal_probe:
        coherent_str = '_temporal'
    else:
        coherent_str += '_coherent'
    
    mode_str = potential_str + coherent_str
    filename = mode_str + f'_N{N_scan_slow*N_scan_fast}_dp{target_Npix}_start_{str(start_idx).zfill(5)}_end_{str(end_idx).zfill(5)}.hdf5'
    output_path = os.path.join(output_dir, filename)

    # Check if the file exists and delete it
    if os.path.exists(output_path):
        os.remove(output_path)  # Delete the existing file so we can overwrite

    with h5py.File(output_path, 'a') as hf:
        hf.create_dataset('/full_volume',   data = potential_resample)
        hf.create_dataset('/volume',        data = potential_crop)
        hf.create_dataset('/gt_phase',      data = gt_phase)
        hf.create_dataset('/dp',            data = cbeds_resample)
        param_group = hf.create_group('abtem_params')
        for key,value in abtem_params.items():
            param_group.create_dataset(key, data=value)
    print(f"Saved hdf5 as {output_path}")

