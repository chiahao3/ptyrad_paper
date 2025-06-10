% Matlab function to run PtyShv
% Updated by Chia-Hao Lee on 2025.06.08

function run_ptyshv_loop(params_path)

    % Add the ptyshv_utils folder to the MATLAB path
    % Note that this script should be placed in the same dir with /ptyshv_utils/
    utils_path = fullfile(fileparts(mfilename('fullpath')), 'ptyshv_utils');
    addpath(utils_path);

    for round_idx = 1:5
        for batch = [1024, 512, 256, 128, 64, 32, 16]
            for pmode = [1, 3, 6, 12]
                for slice = [1, 3, 6]

                    try
                        % Load Params file, currently only support json
                        params = load_params(params_path);

                        fprintf('Running (round_idx, batch, pmode, slice) = (%d, %d, %d, %d)\n', round_idx, batch, pmode, slice);
                        params.output_dir = [params.output_dir, sprintf('_r%d/', round_idx)];
                        params.grouping = batch;
                        params.Nprobe = pmode;
                        params.Nlayers = slice;
                        params.delta_z = 12/slice;

                        % Call core function
                        ptyshv_solver(params);
                    catch ME
                        fprintf('An error occurred for (round_idx, batch, pmode, slice) = (%d, %d, %d, %d)\n', round_idx, batch, pmode, slice);
                        fprintf('Error message: %s\n', ME.message);
                    end

                    % Optional: clear GPU memory to prevent OOM on next run
                    try
                        gpuDevice([]);  % Reset GPU
                    catch
                        % If no GPU or GPU already reset, ignore
                    end

                end
            end
        end
    end
end