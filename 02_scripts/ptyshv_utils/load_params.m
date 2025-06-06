function params = load_params(params_path)

    % Check file path
    if ~isfile(params_path)
        error("Params file not found: %s", params_path);
    end

    fid = fopen(params_path);
    raw = fread(fid, '*char')';
    fclose(fid);
    params = jsondecode(raw);
end