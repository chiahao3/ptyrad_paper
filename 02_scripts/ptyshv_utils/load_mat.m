function data = load_mat(file_path, var_name)
    fallback_keys = {'meas', 'cbed', 'dp', 'data', 'dataset', 'ds'};

    if nargin >= 2 && ~isempty(var_name)
        loaded = load(file_path, var_name);
        if isfield(loaded, var_name)
            data = loaded.(var_name);
            return;
        else
            error('Variable "%s" not found in MAT file.', var_name);
        end
    end

    % Load all variables and scan for common keys
    loaded = load(file_path);
    for k = 1:length(fallback_keys)
        key = fallback_keys{k};
        if isfield(loaded, key)
            data = loaded.(key);
            return;
        end
    end

    % If no match, return everything
    warning('No known key matched. Returning full struct.');
    data = loaded;
end