function data = load_measurements(file_path, key)

    [~, ~, ext] = fileparts(file_path);
    switch lower(ext)
        case '.mat'
            data = load_mat(file_path, key);
        case {'.h5', '.hdf5'}
            data = load_hdf5(file_path, key);
        case {'.tif', '.tiff', '.png'}
            data = load_tiff(file_path);  % key ignored
        case '.raw' % Specifically for .raw from EMPAD1
            data = load_raw(file_path);
        otherwise
            error('Unsupported file extension: %s', ext);
    end
end