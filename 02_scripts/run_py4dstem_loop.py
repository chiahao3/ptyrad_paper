# Python script to run py4DSTEM
# Updated by Chia-Hao Lee on 2024.12.02

import argparse
import cupy as cp

from py4DSTEM.process.phase.utils_CHL import print_system_info, load_yml_params, py4DSTEM_ptycho_solver

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run py4DSTEM", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_path", type=str, required=True)
    args = parser.parse_args()
    
    print_system_info()
    
    
    for batch in [1024,256,64,16]:#[16,32,64,128,256,512,1024]:
        for pmode in [6]: #1,3,6,12
            for update_step_size in [0.5]:
                try:
                    # Run py4DSTEM_ptycho_solver
                    params = load_yml_params(args.params_path)
                    
                    print(f"Running batch = {batch}, pmode = {pmode}, update_step_size = {update_step_size}")
                    params['exp_params']['pmode_max'] = pmode
                    params['recon_params']['BATCH_SIZE'] = batch
                    params['recon_params']['update_step_size'] = update_step_size
                    py4DSTEM_ptycho_solver(params)
                except cp.cuda.memory.OutOfMemoryError as e:
                    print(f"Skipped batch={batch}, pmode={pmode}, update_step_size = {update_step_size} due to OOM.")
                    cp.get_default_memory_pool().free_all_blocks()  # Clear GPU memory
                except Exception as e:
                    print(f"An error occurred for batch={batch}, pmode={pmode}: {e}")