import io
import logging
import os
import warnings
from math import ceil, floor
from time import perf_counter, time

import numpy as np
import torch
import torch.distributed as dist
from scipy.spatial.distance import cdist
from tifffile import imwrite
from torch.fft import fft2, fftfreq, ifft2

class CustomLogger:
    def __init__(self, log_file='output.log', log_dir='auto', prefix_date=True, prefix_jobid=0, append_to_file=True, show_timestamp=True):
        self.logger = logging.getLogger('PtyRAD')
        self.logger.setLevel(logging.INFO)
        
        # Clear all existing handlers to re-instantiate the logger
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.log_file       = log_file
        self.log_dir        = log_dir
        self.flush_file     = log_file is not None
        self.prefix_date    = prefix_date
        self.prefix_jobid   = prefix_jobid
        self.append_to_file = append_to_file
        self.show_timestamp = show_timestamp

        # Create console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s' if show_timestamp else '%(message)s')
        self.console_handler.setFormatter(formatter)
        
        # Create a buffer for file logs
        self.log_buffer = io.StringIO()
        self.buffer_handler = logging.StreamHandler(self.log_buffer)
        self.buffer_handler.setLevel(logging.INFO)
        self.buffer_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.buffer_handler)
        
        # Print logger information
        vprint("### PtyRAD Logger configuration ###")
        vprint(f"log_file       = '{self.log_file}'. If log_file = None, no log file will be created.")
        vprint(f"log_dir        = '{self.log_dir}'. If log_dir = 'auto', then log will be saved to `output_path` or 'logs/'.")
        vprint(f"flush_file     = {self.flush_file}. Automatically set to True if `log_file is not None`")
        vprint(f"prefix_date    = {self.prefix_date}. If true, a datetime str is prefixed to the `log_file`.")
        vprint(f"prefix_jobid   = '{self.prefix_jobid}'. If not 0, it'll be prefixed to the log file. This is used for hypertune mode with multiple GPUs.")
        vprint(f"append_to_file = {self.append_to_file}. If true, logs will be appended to the existing file. If false, the log file will be overwritten.")
        vprint(f"show_timestamp = {self.show_timestamp}. If true, the printed information will contain a timestamp.")
        vprint(' ')

    def flush_to_file(self, log_dir=None, append_to_file=None):
        """
        Flushes buffered logs to a file based on user-defined file mode (append or write)
        """
        
        # Set log_dir
        if log_dir is None:
            if self.log_dir == 'auto':
                log_dir = 'logs'
            else:
                log_dir = self.log_dir

        # Set file_mode
        if append_to_file is None:
            append_to_file = self.append_to_file
        file_mode = 'a' if append_to_file else 'w'
        
        # Set file name
        log_file = self.log_file
        if self.prefix_jobid != 0:
            log_file = str(self.prefix_jobid).zfill(2) + '_' + log_file
        if self.prefix_date:
            log_file = get_date() + '_' + log_file
        
        show_timestamp = self.show_timestamp
        
        if self.flush_file:
            # Ensure the log directory exists
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, log_file)

            # Write the buffered logs to the specified file
            with open(log_file_path, file_mode) as f:
                f.write(self.log_buffer.getvalue())

            # Clear the buffer
            self.log_buffer.truncate(0)
            self.log_buffer.seek(0)

            # Set up a file handler for future logging to the file
            self.file_handler = logging.FileHandler(log_file_path, mode='a')  # Always append after initial flush
            self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s' if show_timestamp else '%(message)s'))
            self.logger.addHandler(self.file_handler)
            vprint(f"### Log file is flushed (created) as {log_file_path} ###")
        else:
            self.file_handler = None
            vprint(f"### Log file is not flushed (created) because log_file is set to {self.log_file} ###")
        vprint(' ')
        
    def close(self):
        """Closes the file handler if it exists."""
        if self.file_handler is not None:
            self.file_handler.flush()
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
            self.file_handler = None

def set_accelerator():

    try:
        from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
        dataloader_config  = DataLoaderConfiguration(split_batches=True) # This supress the warning when we do `Accelerator(split_batches=True)`
        kwargs_handlers    = [DistributedDataParallelKwargs(find_unused_parameters=False)] # This avoids the error `RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss.` We don't necessarily need this if we carefully register parameters (used in forward) and buffer in the `model`.
        accelerator        = Accelerator(dataloader_config=dataloader_config, kwargs_handlers=kwargs_handlers)
        vprint("### Initializing HuggingFace accelerator ###")
        vprint(f"Accelerator.distributed_type = {accelerator.distributed_type}")
        vprint(f"Accelerator.num_process      = {accelerator.num_processes}")
        vprint(f"Accelerator.mixed_precision  = {accelerator.mixed_precision}")
        
    except ImportError:
        vprint("### HuggingFace accelerator is not available, no multi-GPU or mixed-precision ###")
        accelerator = None
        
    vprint(" ")
    return accelerator

def print_system_info():
    
    import os
    import platform
    import sys
    import numpy as np
    import torch
    
    vprint("### System information ###")
    
    # Operating system information
    vprint(f"Operating System: {platform.system()} {platform.release()}")
    vprint(f"OS Version: {platform.version()}")
    vprint(f"Machine: {platform.machine()}")
    vprint(f"Processor: {platform.processor()}")
    
    # CPU cores
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        cpus =  int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        # Fallback to the total number of CPU cores on the node
        cpus = os.cpu_count()
    vprint(f"Available CPU cores: {cpus}")
    
    # Memory information
    if 'SLURM_MEM_PER_NODE' in os.environ:
        # Memory allocated per node by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_NODE']) / 1024  # Convert MB to GB
        vprint(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    elif 'SLURM_MEM_PER_CPU' in os.environ:
        # Memory allocated per CPU by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_CPU']) * cpus / 1024  # Convert MB to GB
        vprint(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    else:
        try:
            import psutil
            # Fallback to system memory information
            mem = psutil.virtual_memory()
            vprint(f"Total Memory: {mem.total / (1024 ** 3):.2f} GB")
            vprint(f"Available Memory: {mem.available / (1024 ** 3):.2f} GB")
        except ImportError:
            vprint("Memory information will be available after `conda install conda-forge::psutil`")
    
    # GPU information
    if torch.backends.cuda.is_built() and torch.cuda.is_available():
        vprint(f"CUDA Available: {torch.cuda.is_available()}")
        vprint(f"CUDA Version: {torch.version.cuda}")
        vprint(f"Available CUDA GPUs: {[torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]}")
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        vprint(f"MPS Available: {torch.backends.mps.is_available()}")
    elif torch.backends.cuda.is_built() or torch.backends.mps.is_built():
        vprint("GPU support built with PyTorch, but could not find any GPU device.")
    else:
        vprint("No GPU backend (CUDA or MPS) built into this PyTorch install.")
        vprint("Install a PyTorch version with GPU support if you want to utilize GPUs.")
    
    # Python version and executable
    vprint(f"Python Executable: {sys.executable}")
    vprint(f"Python Version: {sys.version}")
    vprint(f"NumPy Version: {np.__version__}")
    vprint(f"PyTorch Version: {torch.__version__}")
    try:
        import ptyrad
        vprint(f"PtyRAD Version: {ptyrad.__version__}")
    except ImportError:
        vprint("Didn't find PtyRAD")
    vprint(" ")

def set_gpu_device(gpuid=0):
    vprint("### Setting GPU Device ###")

    if gpuid is None:
        device = torch.device("cpu")
        vprint("Specified to use CPU (gpuid=None).")

    else:
        if torch.cuda.is_available():
            num_cuda_devices = torch.cuda.device_count()
            if gpuid < num_cuda_devices:
                device = torch.device(f"cuda:{gpuid}")
                torch.set_default_device(device)
                vprint(f"Selected GPU device: {device} ({torch.cuda.get_device_name(gpuid)})")
            else:
                device = torch.device("cuda")
                vprint(f"Requested CUDA device cuda:{gpuid} is out of range (only {num_cuda_devices} available)." 
                       f"Fall back to GPU device: {device}")
        
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            vprint("Selected GPU device: MPS (Apple Silicon)")
        
        else:
            device = torch.device("cpu")
            vprint(f"GPU ID specifed as {gpuid} but no GPU found. Using CPU instead.")
            
    torch.set_default_device(device)

    vprint(" ")
    return device

# Verbose print using global logger
def vprint(*args, verbose=True, **kwargs):
    """Verbose print/logging with individual control, only for rank 0 in DDP."""
    if verbose and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
        logger = logging.getLogger('PtyRAD')
        if logger.hasHandlers():
            logger.info(' '.join(map(str, args)), **kwargs)
        else:
            print(*args, **kwargs)

def get_date(date_format='%Y%m%d'):
    from datetime import datetime, date
    
    # If the format includes time-specific placeholders, return full datetime
    if any(fmt in date_format for fmt in ['%H', '%M', '%S']):
        return datetime.now().strftime(date_format)
    
    # Otherwise, just return the date
    return date.today().strftime(date_format)

def has_nan_or_inf(tensor):
    """
    Check if a torch.Tensor contains any NaN or Inf values.

    Parameters:
        tensor (torch.Tensor): Input tensor to check.

    Returns:
        bool: True if the tensor contains any NaN or Inf values, False otherwise.
    """
    # Check for NaN values
    has_nan = torch.isnan(tensor).any()

    # Check for Inf values
    has_inf = torch.isinf(tensor).any()

    return has_nan or has_inf

def get_size_bytes(x):
    
    print(f"Input tensor has shape {x.shape}, dtype {x.dtype}, and live on {x.device}")
    size_bytes = torch.numel(x) * x.element_size()
    size_mib = size_bytes / (1024 * 1024)
    size_gib = size_bytes / (1024 * 1024 * 1024)
    
    if size_bytes < 128 * 1024 * 1024:
        print(f"The size of the tensor is {size_mib:.2f} MiB")
    else:
        print(f"The size of the tensor is {size_gib:.2f} GiB")
    return size_bytes

def time_sync():
    # PyTorch doesn't have a direct exposed API to check the selected default device 
    # so we'll be checking these .is_available() just to prevent error.
    # Luckily these checks won't really affect the performance.
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Check if MPS (Metal Performance Shaders) is available (macOS only)
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    # Measure the time
    t = perf_counter()
    return t

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

def compose_affine_matrix(scale, asymmetry, rotation, shear):
    # Adapted from PtychoShelves +math/compose_affine_matrix.m
    # The input rotation and shear is in unit of degree
    rotation_rad = np.radians(rotation)
    shear_rad = np.radians(shear)
    
    A1 = np.array([[scale, 0], [0, scale]])
    A2 = np.array([[1 + asymmetry/2, 0], [0, 1 - asymmetry/2]])
    A3 = np.array([[np.cos(rotation_rad), np.sin(rotation_rad)], [-np.sin(rotation_rad), np.cos(rotation_rad)]])
    A4 = np.array([[1, 0], [np.tan(shear_rad), 1]])
    
    affine_mat = A1 @ A2 @ A3 @ A4

    return affine_mat

def decompose_affine_matrix(input_affine_mat):
    from scipy.optimize import least_squares
    def err_fun(x):
        scale, asymmetry, rotation, shear = x
        fit_affine_mat = compose_affine_matrix(scale, asymmetry, rotation, shear)
        return (input_affine_mat - fit_affine_mat).ravel()

    # Initial guess
    initial_guess = np.array([1, 0, 0, 0])
    result = least_squares(err_fun, initial_guess)
    scale, asymmetry, rotation, shear = result.x

    return scale, asymmetry, rotation, shear

def get_decomposed_affine_matrix_from_bases(input, output):
    """ Fit the affine matrix components from input and output matrices A and B """
    # This util function is used to quickly estimate the needed affine transformation for scan positions
    # If we know the lattice constant and angle between lattice vectors, then we can easily correct the scale, asymmetry, and shear
    # The global rotation of the object is NOT defined by lattice constant/angle so we still need to compare with the actual CBED
    # Typical usage of this function is to first construct A by measuring the lattice vectors of a reconstructed object suffers from affine transformation
    # Then estimate ideal lattice vectors with prior knowledge (lattice constant and angle)
    # Lastly we use this function to estimate the needed F such that B = F @ A
    
    from scipy.optimize import minimize

    def objective(params, A, B):
        scale, asymmetry, rotation, shear = params
        F = compose_affine_matrix(scale, asymmetry, rotation, shear)
        return np.linalg.norm(B - F @ A)

    initial_guess = [1, 0, 0, 0]  # Initial guess for scale, asymmetry, rotation, shear
    result = minimize(objective, initial_guess, args=(input, output), method='L-BFGS-B')
    
    if result.success:
        (scale, asymmetry, rotation, shear) = result.x
        return (scale, asymmetry, rotation, shear)
    else:
        raise ValueError("Optimization failed")

def select_scan_indices(N_scan_slow, N_scan_fast, subscan_slow=None, subscan_fast=None, mode='full', verbose=True):
    
    N_scans = N_scan_slow * N_scan_fast
    vprint(f"Selecting indices with the '{mode}' mode ", verbose=verbose)
    # Generate flattened indices for the entire FOV
    if mode == 'full':
        indices = np.arange(N_scans)
        return indices

    # Set default values for subscan params
    if subscan_slow is None and subscan_fast is None:
        vprint("Subscan params are not provided, setting subscans to default as half of the total scan for both directions", verbose=verbose)
        subscan_slow = N_scan_slow//2
        subscan_fast = N_scan_fast//2
        
    # Generate flattened indices for the center rectangular region
    if mode == 'center':
        vprint(f"Choosing subscan with {(subscan_slow, subscan_fast)}", verbose=verbose) 
        start_row = (N_scan_slow - subscan_slow) // 2
        end_row = start_row + subscan_slow
        start_col = (N_scan_fast - subscan_fast) // 2
        end_col = start_col + subscan_fast
        indices = np.array([row * N_scan_fast + col for row in range(start_row, end_row) for col in range(start_col, end_col)])

    # Generate flattened indices for the entire FOV with sub-sampled indices
    elif mode == 'sub':
        vprint(f"Choosing subscan with {(subscan_slow, subscan_fast)}", verbose=verbose) 
        full_indices = np.arange(N_scans).reshape(N_scan_slow, N_scan_fast)
        subscan_slow_id = np.linspace(0, N_scan_slow-1, num=subscan_slow, dtype=int)
        subscan_fast_id = np.linspace(0, N_scan_fast-1, num=subscan_fast, dtype=int)
        slow_grid, fast_grid = np.meshgrid(subscan_slow_id, subscan_fast_id, indexing='ij')
        indices = full_indices[slow_grid, fast_grid].reshape(-1)

    else:
        raise KeyError(f"Indices selection mode {mode} not implemented, please use either 'full', 'center', or 'sub'")   
        
    return indices

def make_save_dict(output_path, model, params, optimizer, niter, indices, batch_losses):
    ''' Make a dict to save relevant paramerers '''
    
    avg_losses = {name: np.mean(values) for name, values in batch_losses.items()}
    avg_iter_t = np.mean(model.iter_times)
    
    # While it might seem redundant to save bothe `params` and lots of `model_attributes`,    
    # one should note that `params` only stores the initial value from params files,
    # the actual values used for reconstuction such as N_scan_slow, N_scan_fast, dx, dk, Npix, N_scans could be different from initial value due to the meas_crop, meas_resample
    # the model behavior and learning rates could also be different from the initial params dict if the user
    # run the reconstuction with manually modified `model_params` in the detailed walkthrough notebook
    
    # Postprocess the opt_probe back to complex view
    optimizable_tensors = {}
    for name, tensor in model.optimizable_tensors.items():
        optimizable_tensors[name] = tensor.detach().clone()
        if name == 'probe':
            optimizable_tensors['probe'] = model.get_complex_probe_view().detach().clone()
        
    save_dict = {
                'output_path'           : output_path,
                'optimizable_tensors'   : optimizable_tensors,
                'optim_state_dict'      : optimizer.state_dict() if 'optim_state' in params['recon_params']['save_result'] else None,
                'params'                : params, 
                'model_attributes': # Have to do this explicit saving because I want specific fields but don't want the enitre model with grids and other redundant info
                    {'detector_blur_std': model.detector_blur_std,
                     'obj_preblur_std'  : model.obj_preblur_std,
                     'start_iter'       : model.start_iter,
                     'lr_params'        : model.lr_params,
                     'omode_occu'       : model.omode_occu,
                     'H'                : model.H,
                     'N_scan_slow'      : model.N_scan_slow,
                     'N_scan_fast'      : model.N_scan_fast,
                     'crop_pos'         : model.crop_pos,
                     'slice_thickness'  : model.slice_thickness,
                     'dx'               : model.dx,
                     'dk'               : model.dk,
                     'scan_affine'      : model.scan_affine,
                     'tilt_obj'         : model.tilt_obj,
                     'shift_probes'     : model.shift_probes,
                     'probe_int_sum'    : model.probe_int_sum
                     },
                'loss_iters'            : model.loss_iters,
                'iter_times'            : model.iter_times,
                'dz_iters'              : model.dz_iters,
                'avg_iter_t'            : avg_iter_t,
                'niter'                 : niter,
                'indices'               : indices,
                'batch_losses'          : batch_losses,
                'avg_losses'            : avg_losses
                }
    
    return save_dict

def make_output_folder(output_dir, indices, exp_params, recon_params, model, constraint_params, loss_params, recon_dir_affixes=['lr', 'constraint', 'model', 'loss', 'init'], verbose=True):
    ''' Generate the output folder given indices, recon_params, model, constraint_params, and loss_params '''
    
    # # Example
    # NITER        = 50
    # INDICES_MODE = 'full' #'full', 'center', 'sub'
    # BATCH_SIZE   = 128
    # GROUP_MODE   = 'random' #'random', 'sparse', 'compact'

    # output_dir   = 'output/STO'
    # postfix      = ''

    # pos          = model.crop_pos.cpu().numpy()
    # indices      = select_scan_indices(exp_params['N_scan_slow'], exp_params['N_scan_slow'], subscan_slow=None, subscan_fast=None, mode=INDICES_MODE)
    # batches      = make_batches(indices, pos, BATCH_SIZE, mode=GROUP_MODE)
    # recon_params = make_recon_params_dict(NITER, INDICES_MODE, BATCH_SIZE, GROUP_MODE)
    # output_path  = make_output_folder(output_dir, indices, recon_params, model, constraint_params, postfix)
    
    output_path  = output_dir
    illumination = exp_params['illumination_type']
    meas_flipT   = exp_params['meas_flipT']
    indices_mode = recon_params['INDICES_MODE'].get('mode')
    group_mode   = recon_params['GROUP_MODE']
    batch_size   = recon_params['BATCH_SIZE'].get('size') * recon_params['BATCH_SIZE'].get('grad_accumulation') # Affix the effective batch size
    prefix_date  = recon_params['prefix_date']
    prefix       = recon_params['prefix']
    postfix      = recon_params['postfix']
    pmode        = model.get_complex_probe_view().size(0)
    dp_size      = model.get_complex_probe_view().size(-1)
    obj_shape    = model.opt_objp.shape
    probe_lr     = format(model.lr_params['probe'],            '.0e').replace("e-0", "e-") if model.lr_params['probe'] !=0 else 0
    objp_lr      = format(model.lr_params['objp'],             '.0e').replace("e-0", "e-") if model.lr_params['objp'] !=0 else 0
    obja_lr      = format(model.lr_params['obja'],             '.0e').replace("e-0", "e-") if model.lr_params['obja'] !=0 else 0
    tilt_lr      = format(model.lr_params['obj_tilts'],        '.0e').replace("e-0", "e-") if model.lr_params['obj_tilts'] !=0 else 0
    dz_lr        = format(model.lr_params['slice_thickness'],  '.0e').replace("e-0", "e-") if model.lr_params['slice_thickness'] !=0 else 0
    pos_lr       = format(model.lr_params['probe_pos_shifts'], '.0e').replace("e-0", "e-") if model.lr_params['probe_pos_shifts'] !=0 else 0
    scan_affine  = model.scan_affine if model.scan_affine is not None else None
    init_tilts   = model.opt_obj_tilts.detach().cpu().numpy()
    optimizer_str   = model.optimizer_params['name']
    start_iter_dict = model.start_iter

    # Preprocess prefix and postfix
    prefix  = prefix + '_' if prefix  != '' else ''
    postfix = '_'+ postfix if postfix != '' else ''
    if prefix_date:
        prefix = get_date() + '_' + prefix 
    
    # Setup basic params   
    output_path  = output_dir + "/" + prefix + f"{indices_mode}_N{len(indices)}_dp{dp_size}"
    
    # Attach meas flipping
    if meas_flipT is not None:
        output_path = output_path + '_flipT' + ''.join(str(x) for x in meas_flipT)
    
    # Attach recon mode and pmode 
    output_path += f"_{group_mode}{batch_size}_p{pmode}"
    
    # Attach obj shape and dz
    output_path += f"_{obj_shape[0]}obj_{obj_shape[1]}slice"
    if obj_shape[1] != 1:
        slice_thickness = model.slice_thickness.detach().cpu().numpy().round(2) # This is the initialized slice thickness
        output_path += f"_dz{slice_thickness:.3g}"
    
    # Attach optimizer name (optional)
    if 'optimizer' in recon_dir_affixes:
        output_path += f"_{optimizer_str}"
    
    # Attach start_iter (optional)
    if 'start_iter' in recon_dir_affixes:
        if start_iter_dict['probe'] is not None and start_iter_dict['probe'] > 1:
            output_path += f"_ps{start_iter_dict['probe']}"
        if start_iter_dict['obja'] is not None and start_iter_dict['obja'] > 1:
            output_path += f"_oas{start_iter_dict['obja']}"
        if start_iter_dict['objp'] is not None and start_iter_dict['objp'] > 1:
            output_path += f"_ops{start_iter_dict['objp']}"
        if start_iter_dict['probe_pos_shifts'] is not None and start_iter_dict['probe_pos_shifts'] > 1:
            output_path += f"_ss{start_iter_dict['probe_pos_shifts']}"
        if start_iter_dict['obj_tilts'] is not None and start_iter_dict['obj_tilts'] > 1:
            output_path += f"_ts{start_iter_dict['obj_tilts']}"
        if start_iter_dict['slice_thickness'] is not None and start_iter_dict['slice_thickness'] > 1:
            output_path += f"_dzs{start_iter_dict['slice_thickness']}"
    
    # Attach learning rate (optional)
    if 'lr' in recon_dir_affixes:
        if probe_lr != 0:
            output_path += f"_plr{probe_lr}"
        if obja_lr != 0:
            output_path += f"_oalr{obja_lr}"
        if objp_lr != 0:
            output_path += f"_oplr{objp_lr}"
        if pos_lr != 0:
            output_path += f"_slr{pos_lr}" 
        if tilt_lr != 0:
            output_path += f"_tlr{tilt_lr}"
        if dz_lr != 0:
            output_path += f"_dzlr{dz_lr}"
            
    # Attach model params (optional)
    if 'model' in recon_dir_affixes:    
        if model.obj_preblur_std is not None and model.obj_preblur_std != 0:
            output_path += f"_opreb{model.obj_preblur_std}"
            
        if model.detector_blur_std is not None and model.detector_blur_std != 0:
            output_path += f"_dpblur{model.detector_blur_std}"
    
    # Attach constraint params (optional)
    if 'constraint' in recon_dir_affixes:
        if constraint_params['kr_filter']['freq'] is not None:
            obj_type = constraint_params['kr_filter']['obj_type']
            kr_str = {'both': 'kr', 'amplitude': 'kra', 'phase': 'krp'}.get(obj_type)
            radius = constraint_params['kr_filter']['radius']
            output_path += f"_{kr_str}f{radius}"
        
        if constraint_params['kz_filter']['freq'] is not None:
            obj_type = constraint_params['kz_filter']['obj_type']
            kz_str = {'both': 'kz', 'amplitude': 'kza', 'phase': 'kzp'}.get(obj_type)
            beta = constraint_params['kz_filter']['beta']
            output_path += f"_{kz_str}f{beta}"
            
        if constraint_params['obj_rblur']['freq'] is not None and constraint_params['obj_rblur']['std'] != 0:
            obj_type = constraint_params['obj_rblur']['obj_type']
            obj_str = {'both': 'o', 'amplitude': 'oa', 'phase': 'op'}.get(obj_type)
            output_path += f"_{obj_str}rblur{constraint_params['obj_rblur']['std']}"

        if constraint_params['obj_zblur']['freq'] is not None and constraint_params['obj_zblur']['std'] != 0:
            obj_type = constraint_params['obj_zblur']['obj_type']
            obj_str = {'both': 'o', 'amplitude': 'oa', 'phase': 'op'}.get(obj_type)
            output_path += f"_{obj_str}zblur{constraint_params['obj_zblur']['std']}"
        
        if constraint_params['complex_ratio']['freq'] is not None:
            obj_type = constraint_params['complex_ratio']['obj_type']
            obj_str = {'both': 'o', 'amplitude': 'oa', 'phase': 'op'}.get(obj_type)
            alpha1 = round(constraint_params['complex_ratio']['alpha1'],2)
            alpha2 = round(constraint_params['complex_ratio']['alpha2'],2)
            output_path += f"_{obj_str}cplx{alpha1}_{alpha2}"
        
        if constraint_params['mirrored_amp']['freq'] is not None:
            scale = round(constraint_params['mirrored_amp']['scale'],2)
            power = round(constraint_params['mirrored_amp']['power'],2)
            output_path += f"_mamp{scale}_{power}"
        
        if constraint_params['obja_thresh']['freq'] is not None:
            output_path += f"_oathr{round(constraint_params['obja_thresh']['thresh'][0],2)}"
        
        if constraint_params['objp_postiv']['freq'] is not None:
            mode  = constraint_params['objp_postiv'].get('mode', 'clip_neg')
            mode_str = 's' if mode == 'subtract_min' else 'c'
            relax = constraint_params['objp_postiv']['relax']
            relax_str = '' if relax == 0 else f'{round(relax,2)}'
            output_path += f"_opos{mode_str}{relax_str}"
        
        if constraint_params['tilt_smooth']['freq'] is not None:
            output_path += f"_tsm{round(constraint_params['tilt_smooth']['std'],2)}"
            
        if constraint_params['probe_mask_k']['freq'] is not None:
            output_path += f"_pmk{round(constraint_params['probe_mask_k']['radius'],2)}"

    # Attach loss params (optional)
    if 'loss' in recon_dir_affixes:    
        if loss_params['loss_single']['state']:
            output_path += f"_sng{round(loss_params['loss_single']['weight'],2)}"

        if loss_params['loss_poissn']['state']:
            output_path += f"_psn{round(loss_params['loss_poissn']['weight'],2)}"

        if loss_params['loss_pacbed']['state']:
            output_path += f"_pcb{round(loss_params['loss_pacbed']['weight'],2)}"
        
        if loss_params['loss_sparse']['state']:
            output_path += f"_spr{round(loss_params['loss_sparse']['weight'],2)}"

        if loss_params['loss_simlar']['state']:
            output_path += f"_sml{round(loss_params['loss_simlar']['weight'],2)}"

    # # Attach init params (optional)
    if 'init' in recon_dir_affixes:
        if illumination == 'electron':
            init_conv_angle = exp_params['conv_angle']
            init_defocus    = exp_params['defocus']
            init_c3    = exp_params['c3']
            init_c5    = exp_params['c5']
            output_path += f"_ca{init_conv_angle:.3g}"
            output_path += f"_df{init_defocus:.3g}"
            if init_c3 != 0:
                output_path += f"_c3{format(init_c3, '.0e')}"
            if init_c5 != 0:
                output_path += f"_c5{format(init_c5, '.0e')}"
        elif illumination =='xray':
            init_Ls = exp_params['Ls']
            output_path += f"_Ls{init_Ls* 1e9:.0f}"
        else:
            raise KeyError(f"exp_params['illumination_type'] = {illumination} not implemented yet, please use either 'electron' or 'xray'!")
            
    if scan_affine is not None:
        affine_str = '_'.join(f'{x:.2g}' for x in scan_affine)
        output_path += f"_aff{affine_str}"
    
    if np.any(init_tilts):
        tilts_str = '_'.join(f'{x:.2g}' for x in init_tilts.ravel())
        output_path += f"_tilt{tilts_str}"
    
    output_path += postfix
    
    os.makedirs(output_path, exist_ok=True)
    vprint(f"output_path = '{output_path}' is generated!", verbose=verbose)
    return output_path

def copy_params_to_dir(params_path, output_dir, verbose=True):
    import shutil

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.basename(params_path)
    output_path = os.path.join(output_dir, file_name)
    shutil.copy2(params_path, output_path)
    vprint(" ")
    vprint(f"### Successfully copy '{file_name}' to '{output_dir}' ###", verbose=verbose)
    
def make_batches(indices, pos, batch_size, mode='random', verbose=True):
    ''' Make batches from input indices '''
    # Input:
    #   indices: int, (Ns,) array. indices could be a subset of all indices.
    #   pos: int/float (N,2) array. Always pass in the full positions.
    #   batch_size: int. The number of indices of each mini-batch
    #   mode: str. Choose between 'random', 'compact', or 'sparse' grouping.
    # Output:
    #   batches: A list of `num_batch` arrays, or [batch0, batch1, ...]
    # Note:
    #   The actual batch size would only be "close" if it's not divisible by len(indices) for 'random' grouping
    #   For 'compact' or 'sparse', it's generally fluctuating around the specified batch size
    #   'sparse' can be quite slow for large scan positions (like 256x256 takes more than 10min, and 128x128 takes more than 1min on a CPU)
    #   PtychoShelves automatically switches to 'random' for len(pos) > 1e3 and relying on the random statistics 
    #   To check the correctness of each grouping, you may visualize the pos
    #   Also we want to make sure we're not missing any indices, so we can do:
    #
    #   flatten_indices = np.concatenate(batches)
    #   flatten_indices.sort()
    #   indices.sort()
    #   all(flatten_indices == indices)

    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as e:
        missing_package = str(e).split()[-1]
        vprint(f"### {missing_package} is not available, group mode set to 'random'. 'scikit-learn' is needed for 'sparse' and 'compact' ###")
        mode = 'random'
        
    if len(indices) > len(pos):
        raise ValueError(f"len(indices) = '{len(indices)}' is larger than total number of probe positions ({len(pos)}), check your indices generation params")
    
    if indices.max() > len(pos):
        raise ValueError(f"Maximum index '{indices.max()}' is larger than total number of probe positions ({len(pos)}), check your indices generation params")

    num_batch = len(indices) // batch_size   
    t_start = time()
    if mode == 'random':
        rng = np.random.default_rng()
        shuffled_indices = rng.permutation(indices)           # This will make a shuffled copy    
        random_batches = np.array_split(shuffled_indices, num_batch)
        vprint(f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec", verbose=verbose)
        return random_batches
        
    else: # Either 'compact' or 'sparse'
        # Choose the selected pos from indices
        pos_s = pos[indices]
        # Kmeans for clustering
        kmeans = MiniBatchKMeans(init="k-means++", n_init=10, n_clusters=num_batch, max_iter=10, batch_size=3072)
        kmeans.fit(pos_s)
        labels = kmeans.labels_
        
        # Separate data points into groups
        compact_batches = []
        for batch_idx in range(num_batch):
            batch_indices_s = np.where(labels == batch_idx)[0]
            compact_batches.append(indices[batch_indices_s])

        if mode == 'compact':
            vprint(f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec", verbose=verbose)
            return compact_batches

        else: # 'sparse' mode
            sparse_indices = indices.copy() # Make a deep copy of indices so that we may pop elements from sparse_indices later
            
            # Initialize the list to store groups
            sparse_batches = []
            
            # Calculate the centroid for each compact group as initial start for sparse groups
            # The idea is the centroids of each compact group are naturally sparse
            centroids = np.array([np.mean(pos[cbatch], axis=0) for cbatch in compact_batches])
            pairwise_distances = cdist(pos, pos) # Calculate the dist for ALL pos can keep the absolute index and skip the conversion between indexing
            
            used_indices = [] # This list stores the indices used for initialization of the sparse groups
            # Find the indices closest to the centroids of compact groups, these indices are the initial point for each sparse group
            for batch_idx in range(num_batch):
                distances = np.linalg.norm(pos_s - centroids[batch_idx], axis=1) # Note that this distances is only for selected pos (pos_s = pos[indices])
                closest_idx_s = np.argmin(distances) # closest_idx_s is the position of min distances
                closest_idx = indices[closest_idx_s] # closest_idx is the actual index that is closest to the centroid
                sparse_batches.append([closest_idx])
                used_indices.append(closest_idx_s)
            sparse_indices = np.delete(sparse_indices, used_indices) # Delete the used_indices after the entire loop, this helps keep indexing correct and consistent
            # Deleting elements in a loop would make indexing very challenging
            
            # Iterate through remaining points
            for idx in sparse_indices:
                min_distances = []
                # Iterate through groups
                for batch_idx in range(num_batch):
                    distances = pairwise_distances[sparse_batches[batch_idx], idx]
                    min_distances.append(np.min(distances))
                
                max_group_index = np.argmax(min_distances)

                # Add the point to the group with the farthest minimal distance
                sparse_batches[max_group_index].append(idx)
            
            # Final check because this procedure is fairly complicated
            flatten_indices = np.concatenate(sparse_batches)
            flatten_indices.sort()
            indices.sort()
            assert all(flatten_indices == indices), "Sorry, something went wrong with the sparse grouping, please try 'random' for now"
            vprint(f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec", verbose=verbose)
            
            # Final process to make batches a list of arrays
            sparse_batches = [np.array(batch) for batch in sparse_batches]
            return sparse_batches

def parse_hypertune_params_to_str(hypertune_params):
    
    hypertune_str = ''
    for key, value in hypertune_params.items():
        if key[-2:].lower() == "lr":
            hypertune_str += f"_{key}_{value:.1e}"
        elif isinstance(value, (int, float)):
            hypertune_str += f"_{key}_{value:.3g}"
        else:
            hypertune_str += f"_{key}_{value}"
    
    return hypertune_str

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

def save_results(output_path, model, params, optimizer, niter, indices, batch_losses, collate_str=''):
    
    save_result_list = params['recon_params'].get('save_result', ['model', 'obj', 'probe'])
    result_modes = params['recon_params'].get('result_modes')
    iter_str = '_iter' + str(niter).zfill(4)
    
    if 'model' in save_result_list:
        save_dict = make_save_dict(output_path, model, params, optimizer, niter, indices, batch_losses)
        torch.save(save_dict, os.path.join(output_path, f"model{collate_str}{iter_str}.pt"))
    probe      = model.get_complex_probe_view() 
    probe_amp  = probe.reshape(-1, probe.size(-1)).t().abs().detach().cpu().numpy()
    probe_prop = model.get_propagated_probe([0]).permute(0,2,1,3)
    shape      = probe_prop.shape
    prop_p_amp = probe_prop.reshape(shape[0]*shape[1], shape[2]*shape[3]).abs().detach().cpu().numpy()
    objp       = model.opt_objp.detach().cpu().numpy()
    obja       = model.opt_obja.detach().cpu().numpy()
    # omode_occu = model.omode_occu # Currently not used but we'll need it when omode_occu != 'uniform'
    omode      = model.opt_objp.size(0)
    zslice     = model.opt_objp.size(1)
    crop_pos   = model.crop_pos[indices].detach().cpu().numpy() + np.array(probe.shape[-2:])//2
    y_min, y_max = crop_pos[:,0].min(), crop_pos[:,0].max()
    x_min, x_max = crop_pos[:,1].min(), crop_pos[:,1].max()
    
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
            imwrite(os.path.join(output_path, f"probe_amp{bit_str}{collate_str}{iter_str}.tif"), normalize_by_bit_depth(probe_amp, bit))
        if 'probe_prop' in save_result_list:
            imwrite(os.path.join(output_path, f"probe_prop_amp{bit_str}{collate_str}{iter_str}.tif"), normalize_by_bit_depth(prop_p_amp, bit))
        for fov in result_modes['FOV']:
            if fov == 'crop':
                fov_str = '_crop'
                objp_crop = objp[:, :, y_min-1:y_max, x_min-1:x_max]
                obja_crop = obja[:, :, y_min-1:y_max, x_min-1:x_max]
            elif fov == 'full':
                fov_str = ''
                objp_crop = objp
                obja_crop = obja
            else:
                fov_str = ''
                objp_crop = objp
                obja_crop = obja
                
            postfix_str = fov_str + bit_str + collate_str + iter_str
                
            if any(keyword in save_result_list for keyword in ['obj', 'objp', 'object']):
                # TODO: For omode_occu != 'uniform', we should do a weighted sum across omode instead
                
                for dim in result_modes['obj_dim']:
                    
                    if omode == 1 and zslice == 1:
                        if dim == 2: 
                            imwrite(os.path.join(output_path, f"objp{postfix_str}.tif"),              normalize_by_bit_depth(objp_crop[0,0], bit))
                    elif omode == 1 and zslice > 1:
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"objp_zstack{postfix_str}.tif"),       normalize_by_bit_depth(objp_crop[0,:], bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"objp_zsum{postfix_str}.tif"),         normalize_by_bit_depth(objp_crop[0,:].sum(0), bit))
                    elif omode > 1 and zslice == 1:
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"objp_ostack{postfix_str}.tif"),       normalize_by_bit_depth(objp_crop[:,0], bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"objp_omean{postfix_str}.tif"),        normalize_by_bit_depth(objp_crop[:,0].mean(0), bit))
                            imwrite(os.path.join(output_path, f"objp_ostd{postfix_str}.tif"),         normalize_by_bit_depth(objp_crop[:,0].std(0), bit))
                    else:
                        if dim == 4:
                            imwrite(os.path.join(output_path, f"objp_4D{postfix_str}.tif"),           normalize_by_bit_depth(objp_crop[:,:], bit))
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"objp_ostack_zsum{postfix_str}.tif"),  normalize_by_bit_depth(objp_crop[:,:].sum(1), bit))
                            imwrite(os.path.join(output_path, f"objp_omean_zstack{postfix_str}.tif"), normalize_by_bit_depth(objp_crop[:,:].mean(0), bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"objp_omean_zsum{postfix_str}.tif"),   normalize_by_bit_depth(objp_crop[:,:].mean(0).sum(0), bit))
                            
            if any(keyword in save_result_list for keyword in ['obja']):
                # TODO: For omode_occu != 'uniform', we should do a weighted sum across omode instead
                
                for dim in result_modes['obj_dim']:
                    
                    if omode == 1 and zslice == 1:
                        if dim == 2: 
                            imwrite(os.path.join(output_path, f"obja{postfix_str}.tif"),              normalize_by_bit_depth(obja_crop[0,0], bit))
                    elif omode == 1 and zslice > 1:
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"obja_zstack{postfix_str}.tif"),       normalize_by_bit_depth(obja_crop[0,:], bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"obja_zmean{postfix_str}.tif"),         normalize_by_bit_depth(obja_crop[0,:].mean(0), bit))
                            imwrite(os.path.join(output_path, f"obja_zprod{postfix_str}.tif"),         normalize_by_bit_depth(obja_crop[0,:].prod(0), bit))
                    elif omode > 1 and zslice == 1:
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"obja_ostack{postfix_str}.tif"),       normalize_by_bit_depth(obja_crop[:,0], bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"obja_omean{postfix_str}.tif"),        normalize_by_bit_depth(obja_crop[:,0].mean(0), bit))
                            imwrite(os.path.join(output_path, f"obja_ostd{postfix_str}.tif"),         normalize_by_bit_depth(obja_crop[:,0].std(0), bit))
                    else:
                        if dim == 4:
                            imwrite(os.path.join(output_path, f"obja_4D{postfix_str}.tif"),           normalize_by_bit_depth(obja_crop[:,:], bit))
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"obja_ostack_zmean{postfix_str}.tif"),  normalize_by_bit_depth(obja_crop[:,:].mean(1), bit))
                            imwrite(os.path.join(output_path, f"obja_ostack_zprod{postfix_str}.tif"),  normalize_by_bit_depth(obja_crop[:,:].prod(1), bit))
                            imwrite(os.path.join(output_path, f"obja_omean_zstack{postfix_str}.tif"), normalize_by_bit_depth(obja_crop[:,:].mean(0), bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"obja_omean_zmean{postfix_str}.tif"),   normalize_by_bit_depth(obja_crop[:,:].mean(0).mean(0), bit))
                            imwrite(os.path.join(output_path, f"obja_omean_zprod{postfix_str}.tif"),   normalize_by_bit_depth(obja_crop[:,:].mean(0).prod(0), bit))

def imshift_single(img, shift, grid):
    """
    Generates a single shifted image from a single input image (..., Ny,Nx) with arbitray leading dimensions.
    
    This function shifts a complex/real-valued input image by applying phase shifts in the Fourier domain,
    achieving subpixel shifts in both x and y directions.

    Inputs:
        img (torch.Tensor): The input image to be shifted. 
                            img could be either a mixed-state complex probe (pmode, Ny, Nx) complex64 tensor, 
                            or a mixed-state pseudo-complex object stack (2,omode,Nz,Ny,Nx) float32 tensor.
        shift (torch.Tensor): The shift to be applied to the image. It should be a (2,) tensor and as (shift_y, shift_x).
        grid (torch.Tensor): The k-space grid used for computing the shifts in the Fourier domain. It should be a tensor with shape=(2, Ny, Nx),
                             where Ny and Nx are the height and width of the images, respectively. Note that the grid is normalized so the value spans
                             from 0 to 1

    Outputs:
        shifted_img (torch.Tensor): The shifted image (..., Ny, Nx),

    Note:
        - The shifts are in unit of pixel. For example, a shift of (0.5, 0.5) will shift the image by half a pixel in both y and x directions, positive is down/right-ward.
        - The function utilizes the fast Fourier transform (FFT) to perform the shifting operation efficiently.
        - Make sure to convert the input image and shifts tensor to the desired device before passing them to this function.
        - The fft2 and fftshifts are all applied on the last 2 dimensions, therefore it's only shifting along y and x directions
        - tensor[None, ...] would add an extra dimension at 0, so *[None]*ndim means unwrapping a list of ndim None as [None, None, ...]
        - The img is automatically broadcast to (Nb, *img.shape), so if a batch of images are passed in, each image would be shifted independently
    """
    
    assert img.shape[-2:] == grid.shape[-2:], f"Found incompatible dimensions. img.shape[-2:] = {img.shape[-2:]} while grid.shape[-2:] = {grid.shape[-2:]}"
    
    ndim = img.ndim                                                                   # Get the total img ndim so that the shift is dimension-indepent
    shift = shift[(...,) + (None,) * (ndim-1)]                                        # Expand shifts to (2,1,1,...) so shift.ndim = ndim+1. It was written as `shifts = shifts[..., *[None]*(ndim-1)]` for Python 3.11 or above with better readability
    grid = grid[(slice(None),) + (None,) * (ndim - 2) + (...,)]                       # Expand grid to (2,1,1,...,Ny,Nx) so grid.ndim = ndim+1 It was written as `grid = grid[:,*[None]*(ndim-2), ...]` for Python 3.11 or above with better readability
    shift_y, shift_x = shift[0], shift[1]                                             # shift_y, shift_x are (1,1,...) with ndim singletons, so the shift_y.ndim = ndim
    ky, kx = grid[0], grid[1]                                                         # ky, kx are (1,1,...,Ny,Nx) with ndim-2 singletons, so the ky.ndim = ndim
    w = torch.exp(-(2j * torch.pi) * (shift_x * kx + shift_y * ky))                   # w = (1,1,...,Ny,Nx) so w.ndim = ndim. w is at the center.
    shifted_img = ifft2(ifftshift2(fftshift2(fft2(img)) * w))                         # For real-valued input, take shifted_img.real. 
    
    # Note that for imshift, it's better to keep fft2(img) than fft2(ifftshift2(img))
    # While fft2(img).angle() might seem serrated, it's indeed better to keep it as is, which is essentially setting the center as the origin for FFT.
    
    return shifted_img

def imshift_batch(img, shifts, grid):
    """
    Generates a batch of shifted images from a single input image (..., Ny,Nx) with arbitray leading dimensions.
    
    This function shifts a complex/real-valued input image by applying phase shifts in the Fourier domain,
    achieving subpixel shifts in both x and y directions.

    Inputs:
        img (torch.Tensor): The input image to be shifted. 
                            img could be either a mixed-state complex probe (pmode, Ny, Nx) complex64 tensor, 
                            or a mixed-state pseudo-complex object stack (2,omode,Nz,Ny,Nx) float32 tensor.
        shifts (torch.Tensor): The shifts to be applied to the image. It should be a (Nb,2) tensor and each slice as (shift_y, shift_x).
        grid (torch.Tensor): The k-space grid used for computing the shifts in the Fourier domain. It should be a tensor with shape=(2, Ny, Nx),
                             where Ny and Nx are the height and width of the images, respectively. Note that the grid is normalized so the value spans
                             from 0 to 1

    Outputs:
        shifted_img (torch.Tensor): The batch of shifted images. It has an extra dimension than the input image, i.e., shape=(Nb, ..., Ny, Nx),
                                    where Nb is the number of samples in the input batch.

    Note:
        - The shifts are in unit of pixel. For example, a shift of (0.5, 0.5) will shift the image by half a pixel in both y and x directions, positive is down/right-ward.
        - The function utilizes the fast Fourier transform (FFT) to perform the shifting operation efficiently.
        - Make sure to convert the input image and shifts tensor to the desired device before passing them to this function.
        - The fft2 and fftshifts are all applied on the last 2 dimensions, therefore it's only shifting along y and x directions
        - tensor[None, ...] would add an extra dimension at 0, so *[None]*ndim means unwrapping a list of ndim None as [None, None, ...]
        - The img is automatically broadcast to (Nb, *img.shape), so if a batch of images are passed in, each image would be shifted independently
    """
    
    assert img.shape[-2:] == grid.shape[-2:], f"Found incompatible dimensions. img.shape[-2:] = {img.shape[-2:]} while grid.shape[-2:] = {grid.shape[-2:]}"
    
    ndim = img.ndim                                                                   # Get the total img ndim so that the shift is dimension-indepent
    shifts = shifts[(...,) + (None,) * ndim]                                          # Expand shifts to (Nb,2,1,1,...) so shifts.ndim = ndim+2. It was written as `shifts = shifts[..., *[None]*ndim]` for Python 3.11 or above with better readability
    grid = grid[(slice(None),) + (None,) * (ndim - 1) + (...,)]                       # Expand grid to (2,1,1,...,Ny,Nx) so grid.ndim = ndim+2. It was written as `grid = grid[:,*[None]*(ndim-1), ...]` for Python 3.11 or above with better readability
    shift_y, shift_x = shifts[:, 0], shifts[:, 1]                                     # shift_y, shift_x are (Nb,1,1,...) with ndim singletons, so the shift_y.ndim = ndim+1
    ky, kx = grid[0], grid[1]                                                         # ky, kx are (1,1,...,Ny,Nx) with ndim-2 singletons, so the ky.ndim = ndim+1
    w = torch.exp(-(2j * torch.pi) * (shift_x * kx + shift_y * ky))                   # w = (Nb, 1,1,...,Ny,Nx) so w.ndim = ndim+1. w is at the center.
    shifted_img = ifft2(ifftshift2(fftshift2(fft2(img)) * w))                         # For real-valued input, take shifted_img.real
    
    # Note that for imshift, it's better to keep fft2(img) than fft2(ifftshift2(img))
    # While fft2(img).angle() might seem serrated, it's indeed better to keep it as is, which is essentially setting the center as the origin for FFT.
    
    return shifted_img

def near_field_evolution(Npix_shape, dx, dz, lambd):
    """ Fresnel propagator """
    #  Translated and simplified from Yi's fold_slice Matlab implementation into numPy by Chia-Hao Lee

    ygrid = (np.arange(-Npix_shape[0] // 2, Npix_shape[0] // 2) + 0.5) / Npix_shape[0]
    xgrid = (np.arange(-Npix_shape[1] // 2, Npix_shape[1] // 2) + 0.5) / Npix_shape[1]

    # Standard ASM
    k  = 2 * np.pi / lambd
    ky = 2 * np.pi * ygrid / dx
    kx = 2 * np.pi * xgrid / dx
    Ky, Kx = np.meshgrid(ky, kx, indexing="ij")
    H = np.fft.ifftshift(np.exp(1j * dz * np.sqrt(k ** 2 - Kx ** 2 - Ky ** 2))) # H has zero frequency at the corner in k-space

    return H

def near_field_evolution_torch(Npix_shape, dx, dz, lambd, device='cuda'):
    """Fresnel propagator in PyTorch"""
    # This is for testing and demonstration purpose and is never called in PtyRAD
    # The actual optimizable Fresnel propagator is directly built inside "PtychoAD.get_propagators"
    dx    = dx.to(device)
    dz    = dz.to(device)
    lambd = lambd.to(device)

    ygrid = (torch.arange(-Npix_shape[0] // 2, Npix_shape[0] // 2, device=device) + 0.5) / Npix_shape[0]
    xgrid = (torch.arange(-Npix_shape[1] // 2, Npix_shape[1] // 2, device=device) + 0.5) / Npix_shape[1]

    # Standard ASM
    k  = 2 * torch.pi / lambd
    ky = 2 * torch.pi * ygrid / dx
    kx = 2 * torch.pi * xgrid / dx
    Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
    H = torch.fft.ifftshift(torch.exp(1j * dz * torch.sqrt(k ** 2 - Kx ** 2 - Ky ** 2))) # H has zero frequency at the corner in k-space

    return H

def add_tilts_to_propagator(propagator, tilts, dz, dk):
    """ Add small crystal tilts to a single propagator """
    # After PtyRAD v0.1.0-beta3.0 this is not called anymore for efficiency
    # The tilting is directly calculated in `PtychoAD.get_propagators`
    
    # Ref: https://abtem.readthedocs.io/en/latest/user_guide/walkthrough/multislice.html
    # tilts angle should be less than 1 deg (17 mrad)
    # tilt-induced phase shift = exp(2pi*i*dz*(kx*tan(tx)+ky*tan(ty)), note that k in 1/Ang
    # This is a vectorized function that apply crystall tilts to a single propagator
    # If tilts.ndim=1, then it'll return a single propagator with (1,Y,X)
    # Note that the propagator is corner-centered at k-space
    
    # Create grid of coordinates
    device         = propagator.device
    (ny, nx)       = propagator.shape[-2:]
    grid_y, grid_x = torch.meshgrid(fftfreq(ny, 1 / ny, device=device), fftfreq(nx, 1 / nx, device=device), indexing='ij')
    kx             = grid_x * dk # dk in 1/Ang
    ky             = grid_y * dk
    
    tilts_y        = tilts[:,0,None,None] / 1e3 #mrad, tilts_y = (N,Y,X)
    tilts_x        = tilts[:,1,None,None] / 1e3
    phase_shift    = 2 * torch.pi * dz * (ky * torch.tan(tilts_y) + kx * torch.tan(tilts_x)) 
    propagators    = propagator * torch.exp(1j*phase_shift)
    
    return propagators

def test_loss_fn(model, indices, loss_fn):
    """ Print loss values for each term for convenient weight tuning """
    # model: PtychoAD model
    # indices: array-like indices indicating which probe position to evaluate
    # measurements: 4D-STEM data that's already passed to DEVICE
    # loss_fn: loss function object created from CombinedLoss
    
    with torch.no_grad():
        model_CBEDs, objp_patches = model(indices)
        measured_CBEDs = model.get_measurements(indices)
        _, losses = loss_fn(model_CBEDs, measured_CBEDs, objp_patches, model.omode_occu)

        # Print loss_name and loss_value with padding
        for loss_name, loss_value in zip(loss_fn.loss_params.keys(), losses):
            print(f"{loss_name.ljust(11)}: {loss_value.detach().cpu().numpy():.8f}")
    return

def test_constraint_fn(test_model, constraint_fn, plot_forward_pass):
    """ Test run of the constraint_fn """
    # Note that this would directly modify the model so we need to make a test one

    indices = np.random.randint(0,len(test_model.measurements),2)
    
    constraint_fn(test_model, niter=1) 
    if plot_forward_pass is not None:
        plot_forward_pass(test_model, indices, 0.5)
    del test_model
    return
    
def kv2wavelength(acceleration_voltage):
    # Physical Constants
    PLANCKS = 6.62607015E-34 # m^2*kg / s
    REST_MASS_E = 9.1093837015E-31 # kg
    CHARGE_E = 1.602176634E-19 # coulomb 
    SPEED_OF_LIGHT = 299792458 # m/s

    # Useful constants in EM unit 
    hc = PLANCKS * SPEED_OF_LIGHT / CHARGE_E*1E-3*1E10 # 12.398 keV-Ang, h*c
    REST_ENERGY_E = REST_MASS_E*SPEED_OF_LIGHT**2/CHARGE_E*1E-3 # 511 keV, m0c^2
    
    wavelength = hc/np.sqrt((2*REST_ENERGY_E + acceleration_voltage)*acceleration_voltage) # Angstrom, lambda = hc/sqrt((2*m0c^2 + e*V)*e*V))

    return wavelength

def get_default_probe_simu_params(exp_params):
    illumination_type = exp_params['illumination_type']
    if illumination_type == 'electron':
        probe_simu_params = {
                        ## Basic params
                        "kv"             : exp_params['kv'],
                        "conv_angle"     : exp_params['conv_angle'],
                        "Npix"           : exp_params['Npix'],
                        "dx"             : exp_params['dx_spec'], # dx = 1/(dk*Npix) #angstrom
                        "pmodes"         : exp_params['pmode_max'], # These pmodes specific entries might be used in `make_mixed_probe` during initialization
                        "pmode_init_pows": exp_params['pmode_init_pows'],
                        ## Aberration coefficients
                        "df"             : exp_params['defocus'], #first-order aberration (defocus) in angstrom, positive defocus here refers to actual underfocus or weaker lens strength following Kirkland's notation
                        "c3"             : exp_params['c3'] , #third-order spherical aberration in angstrom
                        "c5"             : exp_params['c5'], #fifth-order spherical aberration in angstrom
                        "c7":0, #seventh-order spherical aberration in angstrom
                        "f_a2":0, #twofold astigmatism in angstrom
                        "f_a3":0, #threefold astigmatism in angstrom
                        "f_c3":0, #coma in angstrom
                        "theta_a2":0, #azimuthal orientation in radian
                        "theta_a3":0, #azimuthal orientation in radian
                        "theta_c3":0, #azimuthal orientation in radian
                        "shifts":[0,0], #shift probe center in angstrom
                        }
    elif illumination_type == 'xray':
        probe_simu_params = {
                        ## Basic params
                        "beam_energy"    : exp_params['energy'],
                        "Npix"           : exp_params['Npix'],
                        "dx"             : exp_params['dx_spec'],
                        "pmodes"         : exp_params['pmode_max'], # These pmodes specific entries might be used in `make_mixed_probe` during initialization
                        "pmode_init_pows": exp_params['pmode_init_pows'],
                        "Ls"             : exp_params['Ls'],
                        "Rn"             : exp_params['Rn'],
                        "dRn"            : exp_params['dRn'],
                        "D_FZP"          : exp_params['D_FZP'],
                        "D_H"            : exp_params['D_H'],
        }
    else:
        raise KeyError(f"exp_params['illumination_type'] = {illumination_type} not implemented yet, please use either 'electron' or 'xray'!")
    return probe_simu_params

def make_stem_probe(params_dict, verbose=True):
    # MAKE_TEM_PROBE Generate probe functions produced by object lens in 
    # transmission electron microscope.
    # Written by Yi Jiang based on Eq.(2.10) in Advanced Computing in Electron 
    # Microscopy (2nd edition) by Dr.Kirkland
    # Implemented and slightly modified in python by Chia-Hao Lee
 
    # Outputs:
        #  probe: complex probe functions at real space (sample plane)
    # Inputs: 
        #  params_dict: probe parameters and other settings
    
    from numpy.fft import fftfreq, fftshift, ifft2, ifftshift
    
    ## Basic params
    voltage     = float(params_dict["kv"])         # Ang
    conv_angle  = float(params_dict["conv_angle"]) # mrad
    Npix        = int  (params_dict["Npix"])       # Number of pixel of thr detector/probe
    dx          = float(params_dict["dx"])         # px size in Angstrom
    ## Aberration coefficients
    df          = float(params_dict["df"]) #first-order aberration (defocus) in angstrom
    c3          = float(params_dict["c3"]) #third-order spherical aberration in angstrom
    c5          = float(params_dict["c5"]) #fifth-order spherical aberration in angstrom
    c7          = float(params_dict["c7"]) #seventh-order spherical aberration in angstrom
    f_a2        = float(params_dict["f_a2"]) #twofold astigmatism in angstrom
    f_a3        = float(params_dict["f_a3"]) #threefold astigmatism in angstrom
    f_c3        = float(params_dict["f_c3"]) #coma in angstrom
    theta_a2    = float(params_dict["theta_a2"]) #azimuthal orientation in radian
    theta_a3    = float(params_dict["theta_a3"]) #azimuthal orientation in radian
    theta_c3    = float(params_dict["theta_c3"]) #azimuthal orientation in radian
    shifts      = params_dict["shifts"] #shift probe center in angstrom
    
    # Calculate some variables
    wavelength = 12.398/np.sqrt((2*511.0+voltage)*voltage) #angstrom
    k_cutoff = conv_angle/1e3/wavelength
    dk = 1/(dx*Npix)

    vprint("Start simulating STEM probe", verbose=verbose)
    
    # Make k space sampling and probe forming aperture
    kx = fftshift(fftfreq(Npix, 1/Npix))
    # kx = np.linspace(-np.floor(Npix/2),np.ceil(Npix/2)-1,Npix)
    kX,kY = np.meshgrid(kx,kx, indexing='xy')

    kX = kX*dk
    kY = kY*dk
    kR = np.sqrt(kX**2+kY**2)
    theta = np.arctan2(kY,kX)
    mask = (kR<=k_cutoff).astype('bool') 
    
    # Adding aberration one-by-one, the aberrations modify the flat phase (imagine a flat wavefront at aperture plane) with some polynomial perturbations
    # The aberrated phase is called chi(k), probe forming aperture is placed here to select the relatively flat phase region to form desired real space probe
    # Note that chi(k) is real-valued function with unit as radian, it's also not limited between -pi,pi. Think of phase shift as time delay might help.
    
    chi = -np.pi*wavelength*kR**2*df
    if c3!=0: 
        chi += np.pi/2*c3*wavelength**3*kR**4
    if c5!=0: 
        chi += np.pi/3*c5*wavelength**5*kR**6
    if c7!=0: 
        chi += np.pi/4*c7*wavelength**7*kR**8
    if f_a2!=0: 
        chi += np.pi*f_a2*wavelength*kR**2*np.sin(2*(theta-theta_a2))
    if f_a3!=0: 
        chi += 2*np.pi/3*f_a3*wavelength**2*kR**3*np.sin(3*(theta-theta_a3))
    if f_c3!=0: 
        chi += 2*np.pi/3*f_c3*wavelength**2*kR**3*np.sin(theta-theta_c3)

    psi = np.exp(-1j*chi)*np.exp(-2*np.pi*1j*shifts[0]*kX)*np.exp(-2*np.pi*1j*shifts[1]*kY)
    probe = mask*psi # It's now the masked wave function at the aperture plane
    probe = fftshift(ifft2(ifftshift(probe))) # Propagate the wave function from aperture to the sample plane. 
    probe = probe/np.sqrt(np.sum((np.abs(probe))**2)) # Normalize the probe so sum(abs(probe)^2) = 1

    if verbose:
        # Print some useful values
        vprint(f'kv          = {voltage} kV')    
        vprint(f'wavelength  = {wavelength:.4f} Ang')
        vprint(f'conv_angle  = {conv_angle} mrad')
        vprint(f'Npix        = {Npix} px')
        vprint(f'dk          = {dk:.4f} Ang^-1')
        vprint(f'kMax        = {(Npix*dk/2):.4f} Ang^-1')
        vprint(f'alpha_max   = {(Npix*dk/2*wavelength*1000):.4f} mrad')
        vprint(f'dx          = {dx:.4f} Ang, Nyquist-limited dmin = 2*dx = {2*dx:.4f} Ang')
        vprint(f'Rayleigh-limited resolution  = {(0.61*wavelength/conv_angle*1e3):.4f} Ang (0.61*lambda/alpha for focused probe )')
        vprint(f'Real space probe extent = {dx*Npix:.4f} Ang')
    
    return probe

def make_fzp_probe(params_dict, verbose=True):
    """
    Generates a Fresnel zone plate probe with internal Fresnel propagation for x-ray ptychography simulations.

    Parameters:
        N (int): Number of pixels.
        lambda_ (float): Wavelength.
        dx (float): Pixel size (in meters) in the sample plane.
        Ls (float): Distance (in meters) from the focal plane to the sample.
        Rn (float): Radius of outermost zone (in meters).
        dRn (float): Width of outermost zone (in meters).
        D_FZP (float): Diameter of pinhole.
        D_H (float): Diameter of the central beamstop (in meters).

    Returns:
        ndarray: Calculated probe field in the sample plane.
    """
    N        = int(params_dict['Npix'])
    energy   = int(params_dict['beam_energy'])
    dx       = params_dict['dx']
    Ls       = params_dict['Ls']
    Rn       = params_dict['Rn']
    dRn      = params_dict['dRn']
    D_FZP    = params_dict['D_FZP']
    D_H      = params_dict['D_H']

    lambda_ = 1.23984193e-9 / energy
    fl = 2 * Rn * dRn / lambda_  # focal length corresponding to central wavelength

    vprint("Start simulating FZP probe", verbose=verbose)

    dx_fzp = lambda_ * fl / N / dx  # pixel size in the FZP plane

    # Coordinate in the FZP plane
    lx_fzp = np.linspace(-dx_fzp * N / 2, dx_fzp * N / 2, N)
    x_fzp, y_fzp = np.meshgrid(lx_fzp, lx_fzp)

    
    T = np.exp(-1j * 2 * np.pi / lambda_ * (x_fzp**2 + y_fzp**2) / (2 * fl))
    C = (np.sqrt(x_fzp**2 + y_fzp**2) <= (D_FZP / 2)).astype(np.float64)  # circular function of FZP
    H = (np.sqrt(x_fzp**2 + y_fzp**2) >= (D_H / 2)).astype(np.float64)  # central block

    
    IN = C * T * H
    M, N = IN.shape
    k = 2 * np.pi / lambda_

    # Coordinate grid for input plane
    lx = np.linspace(-dx_fzp * M / 2, dx_fzp * M / 2, M)
    x, y = np.meshgrid(lx, lx)

    # Coordinate grid for output plane
    fc = 1 / dx_fzp
    fu = lambda_ * (fl + Ls) * fc
    lu = np.fft.ifftshift(np.linspace(-fu / 2, fu / 2, M))
    u, v = np.meshgrid(lu, lu)

    z = fl + Ls
    if z > 0:
        # Propagation in the positive z direction
        pf = np.exp(1j * k * z) * np.exp(1j * k * (u**2 + v**2) / (2 * z))
        kern = IN * np.exp(1j * k * (x**2 + y**2) / (2 * z))
        
        kerntemp = np.fft.fftshift(kern)
        cgh = np.fft.fft2(kerntemp)
        probe = np.fft.fftshift(cgh * pf)
    else:
        # Propagation in the negative z direction (or backward propagation)
        z = abs(z)
        pf = np.exp(1j * k * z) * np.exp(1j * k * (x**2 + y**2) / (2 * z))
        cgh = np.fft.ifft2(np.fft.ifftshift(IN) / np.exp(1j * k * (u**2 + v**2) / (2 * z)))
        probe = np.fft.fftshift(cgh) / pf

    return probe

def make_mixed_probe(probe, pmodes, pmode_init_pows, verbose=True):
    ''' Make a mixed state probe from a single state probe '''
    # Input:
    #   probe: (Ny,Nx) complex array
    #   pmodes: number of incoherent probe modes, scaler int
    #   pmode_init_pows: Integrated intensity of modes. List of a value (e.g. [0.02]) or a couple values for the first few modes. sum(pmode_init_pows) must < 1. 
    # Output:
    #   mixed_probe: A mixed state probe with (pmode,Ny,Nx)
       
    # Prepare a mixed-state probe `mixed_probe`
    vprint(f"Start making mixed-state STEM probe with {pmodes} incoherent probe modes", verbose=verbose)
    M = np.ceil(pmodes**0.5)-1
    N = np.ceil(pmodes/(M+1))-1
    mixed_probe = hermite_like(probe, M,N)[:pmodes]
    
    # Normalize each pmode
    pmode_pows = np.zeros(pmodes)
    for ii in range(1,pmodes):
        if ii<np.size(pmode_init_pows):
            pmode_pows[ii] = pmode_init_pows[ii-1]
        else:
            pmode_pows[ii] = pmode_init_pows[-1]
    if sum(pmode_pows)>1:
        raise ValueError('Modes total power exceeds 1, check pmode_init_pows')
    else:
        pmode_pows[0] = 1-sum(pmode_pows)

    mixed_probe = mixed_probe * np.sqrt(pmode_pows)[:,None,None]
    vprint(f"Relative power of probe modes = {pmode_pows}", verbose=verbose)
    return mixed_probe

def hermite_like(fundam, M, N):
    # %HERMITE_LIKE
    # % Receives a probe and maximum x and y order M N. Based on the given probe
    # % and multiplying by a Hermitian function new modes are computed. The modes
    # % are then orthonormalized.
    
    # Input:
    #   fundam: base function
    #   X,Y: centered meshgrid for the base function
    #   M,N: order of the hermite_list basis
    # Output:
    #   H: 
    # Note:
    #   This function is a python implementation of `ptycho\+core\hermite_like.m` from PtychoShelves with some modification
    #   Most indexings arr converted from Matlab (start from 1) to Python (start from 0)
    #   The X, Y meshgrid are moved into the funciton
    #   The H is modified into (pmode, Ny, Nx) to be consistent with ptyrad
    #   Note that H would output (M+1)*(N+1) modes, which could be a bit more than the specified pmode
    
    
    # Initialize i/o
    M = M.astype('int')
    N = N.astype('int')
    m = np.arange(M+1)
    n = np.arange(N+1)
    H = np.zeros(((M+1)*(N+1), fundam.shape[-2], fundam.shape[-1]), dtype=fundam.dtype)
      
    # Create meshgrid
    rows, cols = fundam.shape[-2:]
    x = np.arange(cols) - cols / 2
    y = np.arange(rows) - rows / 2
    X, Y = np.meshgrid(x, y)
    
    cenx = np.sum(X * np.abs(fundam)**2) / np.sum(np.abs(fundam)**2)
    ceny = np.sum(Y * np.abs(fundam)**2) / np.sum(np.abs(fundam)**2)
    varx = np.sum((X - cenx)**2 * np.abs(fundam)**2) / np.sum(np.abs(fundam)**2)
    vary = np.sum((Y - ceny)**2 * np.abs(fundam)**2) / np.sum(np.abs(fundam)**2)

    counter = 0
    
    # Create basis
    for nii in n:
        for mii in m:
            auxfunc = ((X - cenx)**mii) * ((Y - ceny)**nii) * fundam
            if counter == 0:
                auxfunc = auxfunc / np.sqrt(np.sum(np.abs(auxfunc)**2))
            else:
                auxfunc = auxfunc * np.exp(-((X - cenx)**2 / (2*varx)) - ((Y - ceny)**2 / (2*vary)))
                auxfunc = auxfunc / np.sqrt(np.sum(np.abs(auxfunc)**2))

            # Now make it orthogonal to the previous ones
            for ii in range(counter): # The other ones
                auxfunc = auxfunc - np.dot(H[ii].reshape(-1), np.conj(auxfunc).reshape(-1)) * H[ii]

            # Normalize each mode so that their intensities sum to 1
            auxfunc = auxfunc / np.sqrt(np.sum(np.abs(auxfunc)**2))
            H[counter] = auxfunc
            counter += 1

    return H

def check_modes_ortho(tensor, atol = 2e-5):
    ''' Check if the modes in tensor (Nmodes, []) is orthogonal to each other'''
    # The easiest way to check orthogonality is to calculate the dot product of their 1D vector views
    # Orthogonal vectors would have dot product equals to 0 (Note that `orthonormal` also requires they have unit length)
    # Note that due to the floating point precision, we should set a reasonable tolerance w.r.t 0.
    
    print(f"Input tensor has shape {tensor.shape} and dtype {tensor.dtype}")
    for i in range(tensor.shape[0]):
        for j in range(i + 1, tensor.shape[0]):
            dot_product = torch.dot(tensor[i].view(-1), tensor[j].view(-1))
            if torch.allclose(dot_product, torch.tensor(0., dtype=dot_product.dtype, device=dot_product.device), atol=atol):
                print(f"Modes {i} and {j} are orthogonal with abs(dot) = {dot_product.abs().detach().cpu().numpy()}")
            else:
                print(f"Modes {i} and {j} are not orthogonal with abs(dot) = {dot_product.abs().detach().cpu().numpy()}")

def get_center_of_mass(image, corner_centered=False):
    """ Finds and returns the center of mass of an real-valued 2/3D tensor """
    # The expected input shape can be either (Ny, Nx) or (N, Ny, Nx)
    # The output center_y and center_x will be either (N,) or a scaler tensor
    # Note that for even-number sized arr (like [128,128]), even it's uniformly ones, the "center" would be between pixels like [63.5,63.5]
    # Note that the `corner_centered` flag idea is adapted from py4DSTEM, which is quite handy when we have corner-centered probe or CBED
    # https://github.com/py4dstem/py4DSTEM/blob/dev/py4DSTEM/process/utils/utils.py
    
    ndim = image.ndim
    assert ndim in [2, 3], f"image.ndim must be either 2 or 3, we've got {ndim}"
    
    # Create grid of coordinates
    device = image.device
    (ny, nx) = image.shape[-2:]

    if corner_centered:
        grid_y, grid_x = torch.meshgrid(fftfreq(ny, 1 / ny, device=device), fftfreq(nx, 1 / nx, device=device), indexing='ij')
    else:
        grid_y, grid_x = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing='ij')
    
    # Compute total intensity
    total_intensity = torch.sum(image, dim = (-2,-1)).mean()
    
    # Compute weighted sum of x and y coordinates
    center_y = torch.sum(grid_y * image, dim = (-2,-1)) / total_intensity
    center_x = torch.sum(grid_x * image, dim = (-2,-1)) / total_intensity
    
    return center_y, center_x

def get_blob_size(dx, blob, output='d90', plot_profile=False, verbose=True):
    import matplotlib.pyplot as plt
    """ Get the probe / blob size

    Args:
        dx (float): px size in Ang
        blob (array): the probe/blob image, note that we assume the input is already directly measurable and no squaring is needed, centered, and background free
        plot_profile (bool): Flag for plotting the profile or not 

    Returns:
        D50*dx: D50 in Ang
        D90*dx: D90 in Ang
        radius_rms*dx: RMS radius in Ang
        radial_profile: radially averaged profile
        radial_sum: radial profile without normalizing by the ring area
        fig: Line profile figure
    """
    def get_radial_profile(data, center):
        # The radial intensity is calculated up to the corners
        # So len(radialprofile) will be len(data)/sqrt(2)
        # The bin width is set to be the same with original data spacing (dr = dx)
        y, x = np.indices((data.shape))
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / nr
        radial_sum = tbin
        return radial_profile, radial_sum

    radial_profile, radial_sum = get_radial_profile(blob, (len(blob)//2, len(blob)//2))
    #print("sum(radial_sum) = %.5f " %(np.sum(radial_sum)))

    # Calculate the rms radius, in px
    x = np.arange(len(radial_profile))
    radius_rms = np.sqrt(np.sum(x**2*radial_profile*x)/np.sum(radial_profile*x))

    # Calculate FWHM
    
    HWHM = np.max(np.where((radial_profile / radial_profile.max()) >=0.5))
    
    # Calculate D50, D90
    cum_sum = np.cumsum(radial_sum)

    # R50, 90 without normalization
    R50 = np.min(np.where(cum_sum>=0.50*np.sum(radial_sum))[0])
    R90 = np.min(np.where(cum_sum>=0.90*np.sum(radial_sum))[0])
    R99 = np.min(np.where(cum_sum>=0.99*np.sum(radial_sum))[0])
    R995 = np.min(np.where(cum_sum>=0.995*np.sum(radial_sum))[0])
    R999 = np.min(np.where(cum_sum>=0.999*np.sum(radial_sum))[0])

    D50  = (2*R50+1)
    D90  = (2*R90+1)
    D99  = (2*R99+1)
    D995 = (2*R995+1)
    D999 = (2*R999+1)
    FWHM = (2*HWHM+1)

    if plot_profile:
        
        num_ticks = 11
        x = dx*np.arange(len(radial_profile))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title("Radially averaged profile")
        plt.margins(x=0, y=0)
        ax.plot(x, radial_profile/np.max(radial_profile), label='Radially averaged profile')
        #plt.plot(x, cum_sum, 'k--', label='Integrated current')
        plt.vlines(x=R50*dx, ymin=0, ymax=1, color="tab:orange", linestyle=":", label='R50') #Draw vertical lines at the data coordinate, in this case would be Ang.
        plt.vlines(x=R90*dx, ymin=0, ymax=1, color="tab:red", linestyle=":", label='R90')
        plt.vlines(x=HWHM*dx, ymin=0, ymax=1, color="tab:blue", linestyle=":", label='FWHM')
        plt.vlines(x=radius_rms*dx, ymin=0, ymax=1, color="tab:green", linestyle=":", label='Radius_RMS')
        plt.xticks(np.arange(num_ticks)*np.round(len(radial_profile)*dx/num_ticks, decimals = 1-int(np.floor(np.log10(len(radial_profile)*dx)))))
        ax.set_xlabel(r"Distance from blob center ($\AA$)")
        ax.set_ylabel("Normalized intensity")
        plt.legend()
        plt.show()

    if output == 'd50':
        out = D50*dx
    elif output =='d90':
        out =  D90*dx
    elif output =='d99':
        out =  D99*dx
    elif output =='d995':
        out =  D995*dx
    elif output =='d999':
        out =  D999*dx
    elif output =='radius_rms':
        out =  radius_rms*dx
    elif output =='FWHM':
        out =  FWHM*dx
    elif output =='radial_profile':
        out =  radial_profile
    elif output =='radial_sum':
        out =  radial_sum
    elif output =='fig':
        out =  fig
    else:
        raise KeyError(f"output ={output} not implemented!")
    
    if output not in ['radial_profile', 'radial_sum', 'fig'] and verbose:
        vprint(f'{output} = {out/dx:.3f} px or {out:.3f} Ang')
    return out

def make_sigmoid_mask(Npix, relative_radius=2/3, relative_width=0.2):
    ''' Make a mask from circular sigmoid function '''    
    # relative_radius = 0.67 # This is the relative Nyquist frequency where the sigmoid = 0.5
    # relative_width  = 0.2 # This is the relative width (compared to full image) that y drops from 1 to 0
    
    def scaled_sigmoid(x, offset=0, scale=1):
        # If scale =  1, y drops from 1 to 0 between (-0.5,0.5), or effectively 1 px
        # If scale = 10, it takes roughly 10 px for y to drop from 1 to 0
        scaled_sigmoid = 1 / (1 + torch.exp((x-offset)/scale*10))
        return scaled_sigmoid
    
    ky = torch.linspace(-floor(Npix/2),ceil(Npix/2)-1,Npix)
    kx = torch.linspace(-floor(Npix/2),ceil(Npix/2)-1,Npix)
    grid_ky, grid_kx = torch.meshgrid(ky, kx, indexing='ij')
    kR = torch.sqrt(grid_ky**2+grid_kx**2) # centered already
    sigmoid_mask = scaled_sigmoid(kR, offset=Npix/2*relative_radius, scale=relative_width*Npix)
    
    return sigmoid_mask

def get_rbf(meas, thresh=0.5):
    """ Utility function that returns an estimate of the radius of rbf from CBEDs """
    # meas: 3D array of (N,ky,kx) so that we can take an average 
    # thresh: 0.5 for FWHM, 0.1 for Full-width at 10th maximum
    dp      = meas.sum(0)
    line    = dp.max(0)
    indices = np.where(line > line.max()*thresh)[0]
    rbf     = 0.5*(indices[-1]-indices[0]) # Return rbf in px
    return rbf

def get_local_obj_tilts(pos, objp, dx, slice_thickness, slice_indices, blob_params, window_size=9):
    """ Estimate the local obj tilts from relative atomic column shifts """
    # objp (Nz, Ny, Nx)
    # pos: probe position at integer px sites, (N,2)

    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from scipy.ndimage import center_of_mass
    from scipy.optimize import curve_fit
    from skimage.feature import blob_log

    # Choose the 2 slices from objp and detect blobs from the top slice
    slice_t, slice_b = slice_indices
    height = (slice_b - slice_t)*slice_thickness
    print(f"The height difference between slices {(slice_t, slice_b)} is {height:.2f} Ang")

    target_stack = objp[[slice_t,slice_b]]
    blobs = blob_log(target_stack[0], **blob_params)
    print(f"Found {len(blobs)} blobs with mean radius of {1.414*blobs.mean(0)[-1]:.2f} px or {dx*1.414*blobs.mean(0)[-1]:.2f} Ang")
    
    # Plot the detected blobs
    fig, ax = plt.subplots(figsize=(18,16))
    ax.imshow(target_stack[0])
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, linewidth=2, fill=False)
        ax.add_patch(c)
    plt.show()
    
    # Get the CoM of each atomic column for both top and bottom slices
    row_start = np.uint32(blobs[:,0]-window_size//2)
    row_end   = np.uint32(blobs[:,0]+window_size//2+1)
    col_start = np.uint32(blobs[:,1]-window_size//2)
    col_end   = np.uint32(blobs[:,1]+window_size//2+1)
    coord_t   = np.zeros((len(blobs),2))
    coord_b   = np.zeros((len(blobs),2))

    for i in range(len(blobs)):
        crop_img_t = target_stack[0][row_start[i]:row_end[i], col_start[i]:col_end[i]]
        crop_img_b = target_stack[1][row_start[i]:row_end[i], col_start[i]:col_end[i]]
        coord_t[i] = center_of_mass(crop_img_t) + blobs[i,:-1] - window_size//2
        coord_b[i] = center_of_mass(crop_img_b) + blobs[i,:-1] - window_size//2
    shift_vecs = coord_b - coord_t # This is the needed tilt to correct the obj tilt so it's pointing from top to bottom

    # Plot the detected CoM
    fig, axs = plt.subplots(1,2, figsize=(8,4))
    im0 = axs[0].imshow(crop_img_t)
    im1 = axs[1].imshow(crop_img_b)
    axs[0].set_title(f"crop_img_t \n {coord_t[-1].round(2)}")
    axs[1].set_title(f"crop_img_b \n {coord_b[-1].round(2)}")
    fig.colorbar(im0, shrink=0.7)
    fig.colorbar(im1, shrink=0.7)
    plt.show()
    
    # Plot the tilt vectors
    X = coord_t[:,1]
    Y = coord_t[:,0]
    U = shift_vecs[:,1]
    V = shift_vecs[:,0]
    M = np.arctan(np.hypot(U,V)*dx/height)*1e3

    fig, ax = plt.subplots(figsize=(16,12))
    plt.title("Needed local object tilts", fontsize=16)
    ax.imshow(target_stack[0], cmap='gray')
    q = ax.quiver(X, Y, U, V, M, pivot='mid', angles='xy', scale_units='xy')
    cbar = fig.colorbar(q, shrink=0.75)
    cbar.ax.set_ylabel('mrad')
    plt.show()
    
    # Interpolate tilt_y, tilt_x map
    tilt_y = np.arctan(V*dx/height)*1e3
    tilt_x = np.arctan(U*dx/height)*1e3

    xnew, ynew= np.mgrid[0:target_stack.shape[-2]:1, 0:target_stack.shape[-1]:1]
    tilt_y_interp = griddata(np.stack([Y,X], -1), tilt_y ,(xnew, ynew), method='cubic')
    tilt_x_interp = griddata(np.stack([Y,X], -1), tilt_x ,(xnew, ynew), method='cubic')

    fig, axs = plt.subplots(1,2, figsize=(12,6))
    im0=axs[0].imshow(tilt_y_interp)
    im1=axs[1].imshow(tilt_x_interp)
    axs[0].set_title("tilt_y_interp")
    axs[1].set_title("tilt_x_interp")
    cbar0 = fig.colorbar(im0, shrink=0.7)
    cbar0.ax.set_ylabel('mrad')
    cbar1 = fig.colorbar(im1, shrink=0.7)
    cbar1.ax.set_ylabel('mrad')
    plt.show()
    
    # Use curve_fit to extrapolate to the entire FOV
    def surface_fn(t, a1, b1, c1, d):
        y,x = t
        return  a1*x + b1*y + c1*x*y + d

    xdata = np.vstack((Y,X))
    ydata_tilt_y = tilt_y
    ydata_tilt_x = tilt_x
    popt_tilt_y, _ = curve_fit(surface_fn, xdata, ydata_tilt_y)
    popt_tilt_x, _ = curve_fit(surface_fn, xdata, ydata_tilt_x)
    
    # Implanting griddata interpolated values into the fitted background
    surface_tilt_y = surface_fn(np.stack((ynew,xnew)), *popt_tilt_y)
    surface_tilt_x = surface_fn(np.stack((ynew,xnew)), *popt_tilt_x)

    mask_tilt_y = ~np.isnan(tilt_y_interp)
    surface_tilt_y[mask_tilt_y] = tilt_y_interp[mask_tilt_y]
    mask_tilt_x = ~np.isnan(tilt_x_interp)
    surface_tilt_x[mask_tilt_x] = tilt_x_interp[mask_tilt_x]

    fig, axs = plt.subplots(1,2, figsize=(12,6))
    im0=axs[0].imshow(surface_tilt_y)
    im1=axs[1].imshow(surface_tilt_x)
    axs[0].set_title("surface_tilt_y")
    axs[1].set_title("surface_tilt_x")
    cbar0 = fig.colorbar(im0, shrink=0.7)
    cbar0.ax.set_ylabel('mrad')
    cbar1 = fig.colorbar(im1, shrink=0.7)
    cbar1.ax.set_ylabel('mrad')
    plt.show()
    
    # Sample the surface with our probe position
    tilt_ys = surface_tilt_y[pos[:,0], pos[:,1]]
    tilt_xs = surface_tilt_x[pos[:,0], pos[:,1]]
    obj_tilts = np.stack([tilt_ys, tilt_xs], axis=-1)

    fig, axs = plt.subplots(1,2, figsize=(12,4))
    im0=axs[0].scatter(x=pos[:,1], y=pos[:,0], c=obj_tilts[:,0])
    im1=axs[1].scatter(x=pos[:,1], y=pos[:,0], c=obj_tilts[:,1])
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()

    axs[0].set_title("tilt_ys")
    axs[1].set_title("tilt_xs")
    cbar0 = fig.colorbar(im0, shrink=0.7)
    cbar0.ax.set_ylabel('mrad')
    cbar1 = fig.colorbar(im1, shrink=0.7)
    cbar1.ax.set_ylabel('mrad')
    plt.show()

    return obj_tilts

def add_const_phase_shift(cplx, phase_shift):
    """ Add a constant phase shift to the complex input """
    # This is a handy function to demonstrate that adding a constant phase shift to complex function has no effect to its physical properties
    # For example, even we add a constant phase shift to the probe, it has no effect to the resulting CBED pattern
    # If phase_shift is not a scalar but phase_shift.ndim = 1 as [phi0, phi1, phi2.....], it'll be applied to each mode of the cplx (N, Y, X)
    
    # torch.ones_like will copy the dtype so we make it a float before making the abs and angle
    ones = torch.ones_like(cplx.abs())
    phi = torch.polar(abs=ones, angle=ones * phase_shift[:,None,None])
    return cplx*phi

def get_gaussian1d(size, std, norm=False):
    from scipy.signal.windows import gaussian as gaussian1d

    k = gaussian1d(size, std)
    if norm:
        k /= k.sum()
    return k

def gaussian_blur_1d(tensor, kernel_size=5, sigma=0.5):
    # Note that the F.con1d does not have `padding_mode`, so it's default to be 0 padding, which is not ideal for obja
    # tensor_blur = F.conv1d(input=tensor.reshape(-1, 1, tensor.size(-1)), weight=k1d, padding='same').view(*tensor.shape)

    dtype  = tensor.dtype
    device = tensor.device 
    k = torch.from_numpy(get_gaussian1d(kernel_size, sigma, norm=True)).type(dtype).to(device)
    k1d = k.view(1, 1, -1)
    
    gaussian1d = torch.nn.Conv1d(1,1,kernel_size,padding='same', bias=False, padding_mode='replicate')
    gaussian1d.weight = torch.nn.Parameter(k1d)
    tensor_blur = gaussian1d(tensor.reshape(-1, 1, tensor.size(-1))).view(*tensor.shape)
    return tensor_blur

def fftshift2(x):
    """ A wrapper over torch.fft.fftshift for the last 2 dims """
    # Note that fftshift and ifftshift are only equivalent when N = even 
    return torch.fft.fftshift(x, dim=(-2,-1))  

def ifftshift2(x):
    """ A wrapper over torch.fft.ifftshift for the last 2 dims"""
    # Note that fftshift and ifftshift are only equivalent when N = even 
    return torch.fft.ifftshift(x, dim=(-2,-1))  

def mfft2(im):
    # Periodic Artifact Reduction in Fourier Transforms of Full Field Atomic Resolution Images
    # https://doi.org/10.1017/S1431927614014639
    rows, cols = im.shape
    
    # Compute boundary conditions
    s = np.zeros_like(im)
    s[0, :] = im[0, :] - im[rows-1, :]
    s[rows-1, :] = -s[0, :]
    s[:, 0] += im[:, 0] - im[:, cols-1]
    s[:, cols-1] -= im[:, 0] - im[:, cols-1]

    # Create grid for computing Poisson solution
    cx, cy = np.meshgrid(2 * np.pi * np.arange(cols) / cols, 
                          2 * np.pi * np.arange(rows) / rows)

    # Generate smooth component from Poisson Eq with boundary condition
    D = 2 * (2 - np.cos(cx) - np.cos(cy))
    D[0, 0] = np.inf  # Enforce zero mean & handle division by zero
    S = np.fft.fft2(s) / D

    P = np.fft.fft2(im) - S  # FFT of periodic component
    return P, S

def center_crop(image, crop_height, crop_width, offset = (0,0)):
    """
    Center crops a 2D or 3D array (e.g., an image).

    Args:
        image (numpy.ndarray): The input array to crop. Can be 2D (H, W) or 3D (H, W, C).
        crop_height (int): The desired height of the crop.
        crop_width (int): The desired width of the crop.

    Returns:
        numpy.ndarray: The cropped image.
    """
    if len(image.shape) not in [2, 3]:
        raise ValueError("Input image must be a 2D or 3D array.")

    height, width = image.shape[-2:]

    if crop_height > height or crop_width > width:
        raise ValueError("Crop size must be smaller than the input image size.")

    start_y = (height - crop_height) // 2 + offset[0]
    start_x = (width - crop_width) // 2 + offset[0]

    return image[..., start_y:start_y + crop_height, start_x:start_x + crop_width]

def exponential_decay(r, a, b):
    return a * np.exp(-b * r)

def power_law(r, a, b):
    return a * r**-b

def create_one_hot_mask(image, percentile):
    threshold = np.percentile(image, percentile)
    mask = image <= threshold
    vprint(f"Using percentile = {percentile:.2f}% to create an one-hot mask for measurements amplitude background fitting")
    radius_px = np.sqrt(np.abs(1-mask).sum() / np.pi)
    radius_r  = radius_px / (len(mask)//2)
    vprint(f"The mask has roughly {radius_px:.2f} px in radius, or {radius_r:.2f} of the distance from center to edge of the image")
    return mask.astype(int)

def fit_background(image, mask, fit_type='exp'):
    from scipy.optimize import curve_fit

    y, x = np.indices(image.shape)
    center = np.array(image.shape) // 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2) + 1e-10
    
    masked_r = r[mask == 1]
    masked_image = image[mask == 1]
    
    if fit_type == 'exp':
        initial_guess = [np.max(masked_image), 0.1]  # [a_guess, b_guess]
        bounds = ([0, 0], [np.inf, np.inf])  # a > 0, b > 0
        popt, _ = curve_fit(exponential_decay, masked_r, masked_image, p0=initial_guess, bounds=bounds,maxfev=10000)
        vprint(f"Fitted a = {popt[0]:.4f}, b = {popt[1]:.4f} for exponential decay: y = a*exp(-b*r)")
    elif fit_type == 'power':
        initial_guess = [np.max(masked_image), 1]  # [a_guess, b_guess]
        bounds = ([0, 0], [np.inf, np.inf])  # a > 0, b > 0
        popt, _ = curve_fit(power_law, masked_r, masked_image, p0=initial_guess, bounds=bounds, maxfev=10000)
        vprint(f"Fitted a = {popt[0]:.4f}, b = {popt[1]:.4f} for power law decay: y = a*r^-b")
    else:
        raise ValueError("fit_type must be 'exp' or 'power'")
    
    return popt

###################################### ARCHIVE ##################################################

def cplx_from_np(a, cplx_type='amp_phase', ndim = -1):
    """ Transform a complex numpy array in a "pseudo-complex" tensor"""
    # a: Input complex np array
    # ndim: The axis that stacks the real/imag or amp/phase part
    # cplx_type: "real_imag" or "amp_phase"
    # return: pseuso-complex array shaped (...,2)
    
    if cplx_type == "real_imag":
        return torch.stack([torch.from_numpy(a).real, torch.from_numpy(a).imag], ndim)
    elif cplx_type == "amp_phase":
        return torch.stack([torch.from_numpy(a).abs(), torch.from_numpy(a).angle()], ndim)
    else:
        warnings.warn("cplx_type {} not implemented. Defaulting to 'amp_phase'.".format(cplx_type))
        return torch.stack([torch.from_numpy(a).abs(), torch.from_numpy(a).angle()], ndim)

def complex_object_interp3d(complex_object, zoom_factors, z_axis, use_np_or_cp='np'):
    """
    Interpolate a 3D complex object while preserving multiscattering behavior.

    Parameters:
    - complex_object (ndarray): Input complex object with shape (z, y, x).
    - zoom_factors (tuple): Tuple of zoom factors for (z, y, x).
    = z_axis: int indicating the z-axis posiiton
    - use_np_or_cp (str): Specify the library to use, 'np' for NumPy or 'cp' for CuPy.

    Returns:
    ndarray: Interpolated complex object with the same dtype as the input.

    Notes:
    - Amplitude and phase are treated separately as they obey different conservation laws.
    - Phase shift for multiple z-slices is additive, ensuring the sum of all z-slices remains the same.
    - Amplitude between each z-slice is multiplicative. Linear interpolation of log(amplitude) is performed
      while maintaining the conservation law.
    - The phase of the object should be unwrapped and smooth.
    - If possible, use cupy for 40x faster speed (I got 1 sec vs 40 sec for 320*320*420 target size in a one-shot calculation on my Quadro P5000)

    Example:
    >>> complex_object = np.random.rand(10, 10, 10) + 1j * np.random.rand(10, 10, 10)
    >>> zoom_factors = (2, 2, 1.5)
    >>> result = complex_object_interp3d(complex_object, zoom_factors, use_np_or_cp='np')
    """
    
    if use_np_or_cp == 'cp':
        import cupy as xp
        from cupyx.scipy import ndimage
        complex_object = xp.array(complex_object)
    else:
        import numpy as xp
        from scipy import ndimage
    
    if zoom_factors == (1,1,1):
        print(f"No interpolation is needed, returning original object with shape = {complex_object.shape}.")
        return complex_object

    else:
        obj_dtype = complex_object.dtype
        obj_a = xp.abs(complex_object)
        obj_p = xp.angle(complex_object)
        
        obj_a_interp = xp.exp(ndimage.zoom(xp.log(obj_a), zoom_factors) / zoom_factors[z_axis])
        obj_p_interp = ndimage.zoom(obj_p, zoom_factors) / zoom_factors[z_axis]
        
        complex_object_interp3d = obj_a_interp * xp.exp(obj_p_interp*1j)
        print(f"The object shape is interpolated to {complex_object_interp3d.shape}.")
        return complex_object_interp3d.astype(obj_dtype)

def Fresnel_propagator(probe, dz_distances, lambd, extent):
    # Positive z_distance is adding more overfocus, or letting the probe to forward propagate more
    
    # Example usage
    # dfs = np.linspace(0,200,100)
    # prop_probes = Fresnel_propagator(probe_data, dfs, lambd, extent)
    # print(f"probe_data.shape = {probe_data.shape}, prop_probes.shape = {prop_probes.shape}")
    # print(f"sum(abs(probe)**2) = {np.sum(np.abs(probe_data)**2)}, \nsum(abs(prop_probes)**2) = {np.sum(np.abs(prop_probes)**2, axis=(-3,-2,-1))}")
    
    
    # dfs = [-3,-2,-1,0]
    # prop_probes = Fresnel_propagator(probe_data, dfs, lambd, extent)
    # print(f"probe_data.shape = {probe_data.shape}, prop_probes.shape = {prop_probes.shape}")
    # print(f"sum(abs(probe)**2) = {np.sum(np.abs(probe_data)**2)}, \nsum(abs(prop_probes)**2) = {np.sum(np.abs(prop_probes)**2, axis=(-3,-2,-1))}")

    # plt.figure()
    # plt.title("probe int x-z")
    # plt.imshow(np.abs(prop_probes[:,0,prop_probes.shape[-2]//2,:])**2, aspect=10)
    # plt.yticks(np.arange(0, prop_probes.shape[0]), dfs)
    # plt.ylabel('Ang along z')
    # plt.colorbar()
    # plt.show()
    
    from numpy.fft import fft2, fftshift, ifft2, ifftshift
    
    prop_probes = np.zeros((len(dz_distances), *probe.shape)).astype(probe.dtype)
    for i, z_distance in enumerate(dz_distances):
        _, H, _, _ = near_field_evolution(probe.shape[-2:], z_distance, lambd, extent, use_ASM_only=True, use_np_or_cp='np') # H is corner-centered at k-space
        prop_probes[i] = fftshift(ifft2(H * fft2(ifftshift(probe, axes=(-2,-1)))), axes=(-2,-1))
    
    return prop_probes