function data = load_tiff(file_path)
    info = imfinfo(file_path);
    num_images = numel(info);

    % Read each frame (could be 2D or 3D stack per slice)
    sample = imread(file_path, 1);
    dims = size(sample);
    data = zeros([dims, num_images], class(sample));

    for k = 1:num_images
        data(:, :, :, k) = imread(file_path, k);
    end

    % Squeeze out unused singleton dimensions
    data = squeeze(data);
end