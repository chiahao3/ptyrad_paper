function path = normalize_path(path)
% This normalize all the path separater to /, even on Windows
    path = strrep(path, '\', '/');
end