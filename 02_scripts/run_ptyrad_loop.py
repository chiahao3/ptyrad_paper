# Python script to run PtyRAD
# Updated by Chia-Hao Lee on 2025.06.04

import argparse

from ptyrad.load import load_params
from ptyrad.reconstruction import PtyRADSolver
from ptyrad.utils import CustomLogger, print_system_info, set_accelerator, set_gpu_device, vprint

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run PtyRAD", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--gpuid", type=str, required=False, default="0", help="GPU ID to use ('acc', 'cpu', or an integer)")
    parser.add_argument("--jobid", type=int, required=False, default=0, help="Unique identifier for hypertune mode with multiple GPU workers")
    args = parser.parse_args()
    
    for batch in [4, 1, 1024, 256, 64, 16]:
        try:
            # Setup CustomLogger
            logger = CustomLogger(
                log_file='ptyrad_log.txt',
                log_dir='auto',
                prefix_date=True,
                prefix_jobid=args.jobid,
                append_to_file=True,
                show_timestamp=True
            )            
            # Set up accelerator for multiGPU/mixed-precision setting, note that thess has no effect when we launch it with just `python <script>`
            accelerator = set_accelerator()
                
            print_system_info()
            params = load_params(args.params_path)
            device = set_gpu_device(args.gpuid)
                            
            # Run ptyrad_ptycho_solver
            vprint(f"Running batch = {batch}")
            
            params['recon_params']['BATCH_SIZE']['size'] = batch
            ptycho_solver = PtyRADSolver(params, device=device, acc=accelerator, logger=logger)
            ptycho_solver.run()
            
        except Exception as e:
            vprint(f"An error occurred for batch = {batch}: {e}")


