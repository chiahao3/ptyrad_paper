# Python script to run PtyRAD
# Updated by Chia-Hao Lee on 2025.06.08

import argparse
import gc

import torch
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
    
    for round_idx in range(1, 6):
        for batch in [1024, 512, 256, 128, 64, 32, 16]:
            for pmode in [1, 3, 6, 12]:
                for slice in [1, 3, 6]:
                    for df in [0]: #[-20, -15, -10, -5, 0, 5, 10, 15, 20]:
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
                            vprint(f"Running (round_idx, batch, pmode, slice, df) = {(round_idx, batch, pmode, slice, df)}")
                            
                            params['recon_params']['output_dir'] += f'_r{str(round_idx)}/'
                            params['recon_params']['BATCH_SIZE']['size'] = batch
                            params['init_params']['probe_pmode_max'] = pmode
                            params['init_params']['obj_Nlayer'] = slice
                            params['init_params']['obj_slice_thickness'] = round(12/slice, 2)
                            params['init_params']['probe_defocus'] = df
                            
                            ptycho_solver = PtyRADSolver(params, device=device, acc=accelerator, logger=logger)
                            ptycho_solver.run()
                            
                        except Exception as e:
                            vprint(f"An error occurred for (round, batch, pmode, slice, df) = {(round_idx, batch, pmode, slice, df)}: {e}")

                        finally:
                            del ptycho_solver  # Clear model
                            torch.cuda.empty_cache()
                            gc.collect()

