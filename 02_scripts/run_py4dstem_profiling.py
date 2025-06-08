# Python script to run py4DSTEM
# Updated by Chia-Hao Lee on 2025.06.04

import argparse
from py4DSTEM.process.phase.utils_CHL import print_system_info, load_yml_params, py4DSTEM_ptycho_solver

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Run py4DSTEM", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_path", type=str, required=True)
    args = parser.parse_args()
    
    print_system_info()
    
    # Run py4DSTEM_ptycho_solver
    params = load_yml_params(args.params_path)
    py4DSTEM_ptycho_solver(params)