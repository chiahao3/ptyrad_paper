# Python script to run py4DSTEM
# Updated by Chia-Hao Lee on 2025.06.08

import argparse
import gc

import cupy as cp
from py4DSTEM.process.phase.utils_CHL import (
    load_yml_params,
    print_system_info,
    py4DSTEM_ptycho_solver,
)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run py4DSTEM", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_path", type=str, required=True)
    args = parser.parse_args()
    
    
    for round_idx in [1]:
        for batch in [16]:
            for pmode in [12]:
                for slice in [6]:
                    try:
                        
                        print_system_info()
                        params = load_yml_params(args.params_path)
                        
                        # Run py4DSTEM_ptycho_solver
                        print(f"Running (round_idx, batch, pmode, slice) = {(round_idx, batch, pmode, slice)}")
                        
                        # params['recon_params']['output_dir'] += f'_r{str(round_idx)}/'
                        params['recon_params']['BATCH_SIZE'] = batch
                        params['exp_params']['pmode_max'] = pmode
                        params['exp_params']['Nlayer'] = slice
                        params['exp_params']['slice_thickness'] = round(12/slice, 2)
                        
                        py4DSTEM_ptycho_solver(params)
                        
                    except Exception as e:
                        print(f"An error occurred for (round, batch, pmode, slice) = {(round_idx, batch, pmode, slice)}: {e}")
                        
                    finally:
                        cp.cuda.Device(0).synchronize()
                        cp.get_default_memory_pool().free_all_blocks()  # Clear GPU memory
                        gc.collect()
                    