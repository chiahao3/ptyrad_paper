function data = load_raw(file_path)
    fid = fopen(file_path, 'rb');
    if fid == -1
        error('Cannot open raw file: %s', file_path);
    end
    data = fread(fid, 'float32');
    data = reshape(data, 128,130,[]); % Ndpx, Ndpy, Nscans
    data = data(1:128, 1:128, :); % Crop the 2 rows
    fclose(fid);
end