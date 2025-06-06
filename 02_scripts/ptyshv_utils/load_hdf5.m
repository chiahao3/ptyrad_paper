function data = load_hdf5(file_path, dataset_path)
    fallback_keys = {'/meas', '/cbed', '/dp', '/data', '/dataset', '/ds'};

    if nargin >= 2 && ~isempty(dataset_path)
        data = h5read(file_path, dataset_path);
        return;
    end

    info = h5info(file_path);
    dataset_names = {info.Datasets.Name};

    for k = 1:length(fallback_keys)
        name = erase(fallback_keys{k}, '/');  % Strip slash for matching
        if any(strcmp(name, dataset_names))
            data = h5read(file_path, ['/' name]);
            return;
        end
    end

    % Fallback to first dataset
    if ~isempty(info.Datasets)
        data = h5read(file_path, ['/' info.Datasets(1).Name]);
        warning('No known dataset matched. Using first dataset: %s', info.Datasets(1).Name);
    else
        error('No datasets found in HDF5 file.');
    end
end