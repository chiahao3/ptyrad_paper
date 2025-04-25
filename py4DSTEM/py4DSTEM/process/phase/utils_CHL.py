# utility module written by CHL

import os
import h5py
from time import time

import numpy as np
import py4DSTEM
from tifffile import imwrite

def print_system_info():
    
    import os
    import platform
    import sys
    import numpy as np
    import cupy as cp
    
    print("### System information ###")
    
    # Operating system information
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"OS Version: {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # CPU cores
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        cpus =  int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        # Fallback to the total number of CPU cores on the node
        cpus = os.cpu_count()
    print(f"Available CPU cores: {cpus}")
    
    # Memory information
    if 'SLURM_MEM_PER_NODE' in os.environ:
        # Memory allocated per node by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_NODE']) / 1024  # Convert MB to GB
        print(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    elif 'SLURM_MEM_PER_CPU' in os.environ:
        # Memory allocated per CPU by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_CPU']) * cpus / 1024  # Convert MB to GB
        print(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    else:
        try:
            import psutil
            # Fallback to system memory information
            mem = psutil.virtual_memory()
            print(f"Total Memory: {mem.total / (1024 ** 3):.2f} GB")
            print(f"Available Memory: {mem.available / (1024 ** 3):.2f} GB")
        except ImportError:
            print("Memory information will be available after `conda install conda-forge::psutil`")
    
    # CUDA and GPU information
    raw_version = cp.cuda.runtime.runtimeGetVersion()
    cuda_major  = raw_version // 1000
    cuda_minor  = (raw_version % 1000) // 10
    print(f"CUDA Runtime Version: {cuda_major}.{cuda_minor}")
    print(f"GPU Device: {[cp.cuda.runtime.getDeviceProperties(0)['name'].decode() for d in range(cp.cuda.runtime.getDeviceCount())]}")
    
    # Python version and executable
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Cupy Version: {cp.__version__}")

def get_date(date_format = '%Y%m%d'):
    from datetime import date
    date_format = date_format
    date_str = date.today().strftime(date_format)
    return date_str

def parse_sec_to_time_str(seconds):
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if days > 0:
        return f"{int(days)} day {int(hours)} hr {int(minutes)} min {secs:.3f} sec"
    elif hours > 0:
        return f"{int(hours)} hr {int(minutes)} min {secs:.3f} sec"
    elif minutes > 0:
        return f"{int(minutes)} min {secs:.3f} sec"
    else:
        return f"{secs:.3f} sec"

def load_yml_params(file_path):
    import yaml

    with open(file_path, "r") as file:
        params_dict = yaml.safe_load(file)
    print("Success! Loaded .yml file path =", file_path)
    params_dict['params_path'] = file_path
    return params_dict

def load_raw(path, shape, dtype=np.float32, offset=0, gap=1024):
    # shape = (N, height, width)
    # np.fromfile with custom dtype is faster than the np.read and np.frombuffer
    # This implementaiton is also roughly 2x faster (10sec vs 20sec) than load_hdf5 with a 128x128x128x128 (1GB) EMPAD dataset
    # Note that for custom processed empad2 raw there might be no gap between the images
    N, height, width = shape
    
    # Define the custom dtype to include both data and gap
    custom_dtype = np.dtype([
        ('data', dtype, (height, width)),
        ('gap', np.uint8, gap) # unit8 is equal to 1 byte, so the gap is determined by the length
    ])

    # Read the entire file using the custom dtype
    with open(path, 'rb') as f:
        f.seek(offset)
        raw_data = np.fromfile(f, dtype=custom_dtype, count=N)

    # Extract just the 'data' part (ignoring the gaps)
    data = raw_data['data']
    
    return data

def load_hdf5(file_path, dataset_key="ds"):
    """
    Load data from an HDF5 file.
    
    Parameters:
    
    file_path (str): The full path to the HDF5 data file.
    dataset_key (str, optional): The key of the dataset to load from the HDF5 file.
    
    Returns:
    data (numpy.ndarray): The loaded data.
    
    Raises:
    FileNotFoundError: If the specified file does not exist.
    
    Example:
    file_path = 'data.h5'
    data, data_source = load_hdf5(file_path, dataset_key='ds')
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")

    with h5py.File(file_path, "r") as hf:
        if dataset_key is None:
            print("Imported entire .hdf5 as a dict:", file_path)
            f = dict()
            for key in hf.keys():
                f[key] = np.array(hf[key])
            return f
            
        else:
            data = np.array(hf[dataset_key])
            print("Success! Loaded .hdf5 file path =", file_path)
            print("Imported .hdf5 data shape =", data.shape)
            return data

def load_tif(file_path):
    from tifffile import imread

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    data = imread(file_path)
    print("Success! Loaded .tif file path =", file_path)
    print("Imported .tif data shape =", data.shape)
    return data

def load_npy(file_path):

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    data = np.load(file_path)
    print("Success! Loaded .npy file path =", file_path)
    print("Imported .npy data shape =", data.shape)
    return data

def load_fields_from_mat(file_path, target_field="All", squeeze_me=True, simplify_cells=True):
    """
    Load and extract specified fields from a MATLAB .mat file.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html

    Parameters:
        file_path (str): The path to the MATLAB .mat file to be loaded and processed.
        target_field (str or list of str): The target field name(s) to extract from the .mat file.
            Specify a single field name as a string or multiple field names as a list of strings.
            Use "All" to load the entire .mat file.

    Returns:
        result_list (list or dict): A list containing the extracted field(s) as elements.
            If target_field is "All," the entire .mat file is returned as a dictionary.

    Raises:
        ValueError: If the nesting depth of target_field exceeds the maximum supported depth of 3.
        ValueError: If target_field is neither a string nor a list of strings.

    Examples:
        # Load the entire .mat file as a dictionary
        file_path = "your_file.mat"
        target_field = "All"
        result = load_fields_from_mat(file_path, target_field)

        # Extract a single field
        file_path = "your_file.mat"
        target_field = "object.sub_field"
        result = load_fields_from_mat(file_path, target_field)

        # Extract multiple fields
        file_path = "your_file.mat"
        target_field = ["object.sub_field", "another_object.field"]
        results = load_fields_from_mat(file_path, target_field)

        # Process the results
        for i, result in enumerate(results):
            if result is not None:
                print(f"Result {i + 1}: {result}")
    """
    import scipy.io as sio
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    result_list = []

    # Load entire .mat
    if target_field == "All":
        try:
            mat_contents = sio.loadmat(
                file_path, squeeze_me=squeeze_me, simplify_cells=simplify_cells
            )
            print("Success! Loaded .mat File path =", file_path)
            return mat_contents
        except NotImplementedError:
            # If loading from MATLAB file complains, switch to HDF5
            print("Can't load .mat v7.3 with scipy. Switching to h5py.")
            mat_contents = {}
            with h5py.File(file_path, "r") as hdf_file:
                for key in hdf_file.keys():
                    mat_contents[key] = hdf_file[key][()]
            print("Success! Loaded .mat file path =", file_path)
            return mat_contents

    # Check target_field type
    if isinstance(target_field, str):
        target_fields = [target_field]
    elif isinstance(target_field, list):
        target_fields = target_field
    else:
        raise ValueError("target_field must be a string or a list of strings")

    # Load field by field in target_fields (list)
    for name in target_fields:
        try:
            mat_contents = sio.loadmat(
                file_path, squeeze_me=squeeze_me, simplify_cells=simplify_cells
            )
            fields = name.split(".")
            outputs = mat_contents

            if len(fields) > 3:
                raise ValueError("The maximum supported nesting depth is 3.")

            for field in fields:
                if field in outputs:
                    if isinstance(outputs, sio.matlab.mio5.mat_struct):
                        outputs = getattr(outputs, field)
                    else:
                        outputs = outputs[field]
                else:
                    print(f"Field '{field}' not found in file {file_path}")
                    result_list.append(None)
                    break
            else:
                result_list.append(outputs)
        except NotImplementedError:
            # If loading from MATLAB file complains, switch to HDF5
            print("Can't load .mat v7.3 with scipy. Switching to h5py.")
            with h5py.File(file_path, "r") as hdf_file:
                result_list.append(hdf_file[name][()])
    print("Success! Loaded .mat file path =", file_path)
    return result_list

def init_datacube(exp_params):
    '''
    Initialize the py4DSTEM datacube and return it with updated exp_params
    '''
    data_source = exp_params['measurements_params'].get('source')
    data_path   = exp_params['measurements_params'].get('path')
    data_key    = exp_params['measurements_params'].get('key')
    
    # Load file
    if data_source == 'tif':
        meas = load_tif(data_path) # key is ignored because it's not needed for tif files
    elif data_source == 'mat':
        meas = load_fields_from_mat(data_path, data_key)[0]
    elif data_source == 'hdf5':
        meas = load_hdf5(data_path, data_key).astype('float32')
    elif data_source == 'npy':
        meas = load_npy(data_path).astype('float32')
    elif data_source == 'raw':
        default_shape = (exp_params['N_scans'], exp_params['Npix'], exp_params['Npix'])
        meas = load_raw(data_path,
                        shape=exp_params['measurements_params'].get('shape', default_shape), 
                        offset=exp_params['measurements_params'].get('offset', 0), 
                        gap=exp_params['measurements_params'].get('gap', 1024))
    else:
        raise KeyError(f"File type {data_source} not implemented yet, please use 'custom', 'tif', 'mat', or 'hdf5'!!")
    print(f"Imported meausrements shape / dtype = {meas.shape}, {meas.dtype}")
    print(f"Imported meausrements int. statistics (min, mean, max) = ({meas.min():.4f}, {meas.mean():.4f}, {meas.max():.4f})")

    # Permute, reshape, and flip
    if exp_params['meas_permute'] is not None:
        permute_order = exp_params['meas_permute']
        print(f"Permuting measurements with {permute_order}")
        meas = meas.transpose(permute_order)
        
    if exp_params['meas_reshape'] is not None:
        meas_shape = exp_params['meas_reshape']
        print(f"Reshaping measurements into {meas_shape}")
        meas = meas.reshape(meas_shape)
        
    if exp_params['meas_flipT'] is not None:
        flipT_axes = exp_params['meas_flipT']
        print(f"Flipping measurements with [flipup, fliplr, transpose] = {flipT_axes}")
        if flipT_axes[0] != 0:
            meas = np.flip(meas, 2)
        if flipT_axes[1] != 0:
            meas = np.flip(meas, 3)
        if flipT_axes[2] != 0:
            meas = np.transpose(meas, (0,1,3,2))
    
    # Crop out a sub-region from meas
    if exp_params['meas_crop'] is not None:
        print(f"Reshaping measurements into {meas.shape} for cropping")
        meas = meas.reshape(exp_params['N_scan_slow'], exp_params['N_scan_fast'], meas.shape[-2], meas.shape[-1])

        crop_indices = np.array(exp_params['meas_crop'])
        print(f"Cropping measurements with [N_slow, N_fast, ky, kx] = {crop_indices}")
        Nslow_i, Nslow_f = crop_indices[0]
        Nfast_i, Nfast_f = crop_indices[1]
        ky_i,    ky_f    = crop_indices[2]
        kx_i,    kx_f    = crop_indices[3]
        meas = meas[Nslow_i:Nslow_f, Nfast_i:Nfast_f, ky_i:ky_f, kx_i:kx_f]
        print(f"Cropped measurements have shape (N_slow, N_fast, ky, kx) = {meas.shape}")
        
        print("Update `exp_params` (dx_spec, Npix, N_scans, N_scan_slow, N_scan_fast) after the measurements cropping")
        exp_params['dx_spec'] = exp_params['dx_spec'] * exp_params['Npix'] / meas.shape[-1]
        exp_params['Npix']    = meas.shape[-1]
        exp_params['N_scans'] = meas.shape[0] * meas.shape[1]
        exp_params['N_scan_slow'] = meas.shape[0]
        exp_params['N_scan_fast'] = meas.shape[1]
    
    # Resample diffraction patterns along the ky, kx dimension
    if exp_params['meas_resample'] is not None:
        from scipy.ndimage import zoom
        zoom_factors = np.array([1, 1, *exp_params['meas_resample']]) # scipy.ndimage.zoom applies to all axes
        meas = zoom(meas, zoom_factors, order=1)
        print("Update `exp_params` (Npix) after the measurements resampling")
        exp_params['Npix'] = meas.shape[-1]
    
    # Correct negative values if any
    if (meas < 0).any():
        min_value = meas.min()
        meas -= min_value
        # Subtraction is more general, but clipping might be more noise-robust due to the inherent denoising
        print(f"Minimum value of {min_value:.4f} subtracted due to the positive px value constraint of measurements")
         
    # Normalizing meas
    print("Normalizing measurements so the averaged measurement has max intensity at 1")
    meas = meas / (np.mean(meas, (0,1)).max()) # Normalizing the meas_data so that the averaged DP has max at 1. This will make each DP has max somewhere ~ 1
    meas = meas.astype('float32')
    print(f"Processed meausrements int. statistics (min, mean, max) = ({meas.min():.4f}, {meas.mean():.4f}, {meas.max():.4f})")
    
    # Calibrate py4DSTEM datacube
    datacube = py4DSTEM.DataCube(meas)
    datacube.calibration.set_R_pixel_size(exp_params['scan_step_size'])
    datacube.calibration.set_R_pixel_units('A')
    datacube.calibration.set_Q_pixel_size(1/(exp_params['dx_spec']*exp_params['Npix']))
    datacube.calibration.set_Q_pixel_units('A^-1')
    
    print(f"py4DSTEM datacube.shape = {datacube.shape} (N_scan_slow, N_scan_fast, ky, kx)")
    return datacube, exp_params

def init_ptycho(datacube, exp_params):
    '''
    Initialize the py4DSTEM ptycho object
    '''
    Npix           = exp_params['Npix']
    N_scan_fast    = exp_params['N_scan_fast']
    N_scan_slow    = exp_params['N_scan_slow']
    dx_spec        = exp_params['dx_spec']
    scan_step_size = exp_params['scan_step_size']
    
    pos_extent = np.array([N_scan_slow,N_scan_fast]) * scan_step_size / dx_spec
    object_extent = 1.2 * (pos_extent + Npix)
    object_padding_px = tuple((object_extent - pos_extent)//2)
    print(f"pos_extent = {pos_extent} px, object_extent = {object_extent}, object_padding_px = {object_padding_px}")
    
    if exp_params['Nlayer'] == 1:
        print("Initializing MixedstatePtychography")
        ptycho = py4DSTEM.process.phase.MixedstatePtychography(
            datacube=datacube,
            num_probes = exp_params['pmode_max'],
            verbose=True,
            energy = exp_params['kv']*1e3, # energy in eV
            defocus= exp_params['defocus'], # defocus guess in A
            semiangle_cutoff = exp_params['conv_angle'],
            object_padding_px = object_padding_px,
            device='gpu', 
            storage='cpu', 
        )
    else:
        print("Initializing MixedstateMultislicePtychography")
        ptycho = py4DSTEM.process.phase.MixedstateMultislicePtychography(
            datacube=datacube,
            num_probes = exp_params['pmode_max'],
            num_slices=exp_params['Nlayer'],
            slice_thicknesses=exp_params['slice_thickness'],
            verbose=True,
            energy = exp_params['kv']*1e3, # energy in eV
            defocus= exp_params['defocus'], # defocus guess in A
            semiangle_cutoff = exp_params['conv_angle'],
            object_padding_px = object_padding_px,
            device='gpu', 
            storage='cpu',
        )
    return ptycho

def make_output_folder(exp_params, recon_params):
    # Preprocess prefix and postfix
    prefix = recon_params['prefix']
    postfix = recon_params['postfix']
    prefix = prefix + '_' if prefix  != '' else ''
    postfix = '_'+ postfix if postfix != '' else ''

    if recon_params['prefix_date']:
        prefix = get_date() + '_' + prefix 

    # Append basic parameters to folder name
    output_dir  = recon_params['output_dir']
    meas_flipT  = exp_params['meas_flipT'] 
    folder_str = prefix + f"N{(exp_params['N_scans'])}_dp{exp_params['Npix']}"

    if meas_flipT is not None:
        folder_str = folder_str + '_flipT' + ''.join(str(x) for x in meas_flipT)

    folder_str += f"_random{recon_params['BATCH_SIZE']}_p{exp_params['pmode_max']}_{exp_params['Nlayer']}slice"

    if exp_params['Nlayer'] != 1:
        slice_thickness = np.array(exp_params['slice_thickness']).round(2)
        folder_str += f"_dz{slice_thickness:.3g}"
    
    # Append update step size
    folder_str += f"_update{recon_params['update_step_size']}"

    # Append constraint keyword
    if recon_params['kz_regularization_gamma'] is not None and exp_params['Nlayer'] != 1:
        folder_str += f"_kzf{recon_params['kz_regularization_gamma']}"
    
    output_path = os.path.join(output_dir, folder_str)
    output_path += postfix
    os.makedirs(output_path, exist_ok=True)
    print(f"output_path = '{output_path}' is generated!")
    return output_path

def get_propagated_probe(probe, n_slices, propagator):
    probe = probe if probe.ndim == 3 else probe[None,] # (pmode, Ny, Nx)
    H = propagator if propagator.ndim ==3 else propagator[None,] # (1, Ny, Nx)
    probe_prop = np.zeros((n_slices, *probe.shape), dtype=probe.dtype)
    
    psi = probe # (z, pmode, Ny, Nx)
    for n in range(n_slices):
        probe_prop[n] = psi
        psi = np.fft.ifft2(H[None,] * np.fft.fft2(psi))
    
    return probe_prop

def normalize_from_zero_to_one(arr):
    norm_arr = (arr - arr.min())/(arr.max()-arr.min())
    return norm_arr

def normalize_by_bit_depth(arr, bit_depth):

    if bit_depth == '8':
        norm_arr_in_bit_depth = np.uint8(255*normalize_from_zero_to_one(arr))
    elif bit_depth == '16':
        norm_arr_in_bit_depth = np.uint16(65535*normalize_from_zero_to_one(arr))
    elif bit_depth == '32':
        norm_arr_in_bit_depth = np.float32(normalize_from_zero_to_one(arr))
    elif bit_depth == 'raw':
        norm_arr_in_bit_depth = np.float32(arr)
    else:
        print(f'Unsuported bit_depth :{bit_depth} was passed into `result_modes`, `raw` is used instead')
        norm_arr_in_bit_depth = np.float32(arr)
    
    return norm_arr_in_bit_depth

def save_results(output_path, model, save_result, result_modes, niter):
    
    asnumpy = model._asnumpy
    
    save_result_list = save_result if save_result is not None else ['model', 'obj', 'probe']
    result_modes = result_modes if result_modes is not None else {'obj_dim': [2], 'FOV': ['crop'], 'bit': ['8']}
    iter_str = '_iter' + str(niter).zfill(4)
    
    # Retrieve parameters
    probe = model.probe_centered
    obj   = asnumpy(model._object.copy())
    positions_px = asnumpy(model._positions_px)
    probe_amp = np.abs(asnumpy(probe.reshape(-1, probe.shape[-1])).T)
    objp = np.angle(obj)
    obja = np.abs(obj)
    zslice = 1 if obj.ndim == 2 else obj.shape[0]
    crop_pos = np.round(positions_px).astype('int16')
    y_min, y_max = crop_pos[:,0].min(), crop_pos[:,0].max()
    x_min, x_max = crop_pos[:,1].min(), crop_pos[:,1].max()
    
    if zslice != 1:
        propagator = asnumpy(model._propagator_arrays[0])[None,] # Although py4DSTEM supports non-uniform slice thickness, we assume the user would pass a uniform thickness so we'll take only the 1st propagator
    
    if 'model' in save_result_list:
        # Save the reconstruction model as hdf5
        with h5py.File(os.path.join(output_path, f'model_iter{str(niter).zfill(4)}.hdf5'), "w") as f:
            f.create_dataset('probe',            data=probe)
            f.create_dataset('object',           data=obj)
            f.create_dataset('positions_px',     data=positions_px)
            f.create_dataset('error_iterations', data=model.error_iterations)
            f.create_dataset('iter_times',       data=model.iter_times)
    
    # Modify parameter dimensions to reuse the saving logic from PtyRAD
    objp = objp if objp.ndim == 3 else objp[None,]
    obja = obja if obja.ndim == 3 else obja[None,]
    probe = probe if probe.ndim == 3 else probe[None,] # (pmode, Ny, Nx)
    
    # Get propagated probe
    if zslice != 1:
        probe_prop = np.transpose(get_propagated_probe(probe, n_slices=zslice, propagator=propagator), axes=(0,2,1,3))
        shape      = probe_prop.shape
        prop_p_amp = np.abs(probe_prop.reshape(shape[0]*shape[1], shape[2]*shape[3]))
    
    for bit in result_modes['bit']:
        if bit == '8':
            bit_str = '_08bit'
        elif bit == '16':
            bit_str = '_16bit'
        elif bit == '32':
            bit_str = '_32bit'
        elif bit == 'raw':
            bit_str = ''
        else:
            bit_str = ''
        if 'probe' in save_result_list:
            imwrite(os.path.join(output_path, f"probe_amp{bit_str}{iter_str}.tif"), normalize_by_bit_depth(probe_amp, bit))
        if 'probe_prop' in save_result_list and zslice != 1:
            imwrite(os.path.join(output_path, f"probe_prop_amp{bit_str}{iter_str}.tif"), normalize_by_bit_depth(prop_p_amp, bit))
        for fov in result_modes['FOV']:
            if fov == 'crop':
                fov_str = '_crop'
                objp_crop = objp[:, y_min-1:y_max, x_min-1:x_max]
                obja_crop = obja[:, y_min-1:y_max, x_min-1:x_max]
            elif fov == 'full':
                fov_str = ''
                objp_crop = objp
                obja_crop = obja
            else:
                fov_str = ''
                objp_crop = objp
                obja_crop = obja
                
            postfix_str = fov_str + bit_str + iter_str
                
            if any(keyword in save_result_list for keyword in ['obj', 'objp', 'object']):
                
                for dim in result_modes['obj_dim']:
                    
                    if zslice == 1:
                        if dim == 2: 
                            imwrite(os.path.join(output_path, f"objp{postfix_str}.tif"),              normalize_by_bit_depth(objp_crop[0], bit))
                    else: # multislice
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"objp_zstack{postfix_str}.tif"),       normalize_by_bit_depth(objp_crop, bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"objp_zsum{postfix_str}.tif"),         normalize_by_bit_depth(objp_crop.sum(0), bit))
                            
            if any(keyword in save_result_list for keyword in ['obja']):
                
                for dim in result_modes['obj_dim']:
                    
                    if zslice == 1:
                        if dim == 2: 
                            imwrite(os.path.join(output_path, f"obja{postfix_str}.tif"),              normalize_by_bit_depth(obja_crop[0], bit))
                    else: # multislice
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"obja_zstack{postfix_str}.tif"),       normalize_by_bit_depth(obja_crop, bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"obja_zmean{postfix_str}.tif"),        normalize_by_bit_depth(obja_crop.mean(0), bit))
                            imwrite(os.path.join(output_path, f"obja_zprod{postfix_str}.tif"),        normalize_by_bit_depth(obja_crop.prod(0), bit))

def make_reconstruct_kwargs(model, recon_params):
    import inspect
    # Get all parameters for the reconstruct method
    sig = inspect.signature(model.reconstruct)
    kwargs = {
        'num_iter'             : recon_params['NITER'],
        'reconstruction_method': 'gradient-descent',
        'max_batch_size'       : recon_params['BATCH_SIZE'],
        'step_size'            : recon_params['update_step_size'], # Update step size, default is 0.5 but 0.1 is numerically more stable for multislice
        'reset'                : True, # If True, previous reconstructions are ignored
        'progress_bar'         : False, # If True, reconstruction progress is displayed
        'store_iterations'     : False, # If True, reconstructed objects and probes are stored at each iteration.
        'save_iters'           : recon_params['SAVE_ITERS'],
        'save_result'          : recon_params['save_result'],
        'result_modes'         : recon_params['result_modes'],
        'output_path'          : recon_params['output_path']
    }
    
    # Add more kwargs from recon_params if recon_kwargs exists. Note that recon_kwargs would override the existing values
    if recon_params['recon_kwargs'] is not None:
        kwargs.update(recon_params['recon_kwargs'])
    
    # Filter kwargs to only include valid arguments for reconstruct
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
    return filtered_kwargs

def py4DSTEM_ptycho_solver(params):
    
    exp_params = params['exp_params']
    recon_params = params['recon_params']
    
    ## Initialize py4DSTEM datacube and return modified exp_params
    datacube, exp_params = init_datacube(exp_params)
    
    ## Initialize py4DSTEM ptycho instance
    ptycho = init_ptycho(datacube, exp_params)
    ptycho.preprocess(
        plot_center_of_mass = False,
        plot_rotation=False,
    )
    
    # Make output folder     
    output_path = make_output_folder(exp_params, recon_params)
    recon_params['output_path'] = output_path
    
    # Reconstruct py4dstem ptycho
    solver_start_t = time()
    kwargs = make_reconstruct_kwargs(ptycho, recon_params) # Filter the kwargs in case there's a mismatch in class methods
    print(f"reconstruction kwargs = {kwargs}")
    ptycho.reconstruct(**kwargs)
    
    solver_end_t = time()
    print(f"### py4DSTEM ptycho solver is finished in {parse_sec_to_time_str(solver_end_t - solver_start_t)}###\n")