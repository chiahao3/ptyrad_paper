% Matlab function to run PtyShv
% Updated by Chia-Hao Lee on 2025.06.06

function run_ptyshv_profiling(params_path)

    % Add the ptyshv_utils folder to the MATLAB path
    % Note that this script should be placed in the same dir with /ptyshv_utils/
    utils_path = fullfile(fileparts(mfilename('fullpath')), 'ptyshv_utils');
    addpath(utils_path);

    % Load Params file, currently only support json
    params = load_params(params_path);

    % Call core function
    ptyshv_solver(params);

end