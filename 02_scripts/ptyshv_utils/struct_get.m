function val = struct_get(s, key, default)
    if isfield(s, key)
        val = s.(key);
    else
        val = default;
    end
end