function entropy = crossEntropy(x, z)
    ndata = size(x, 2);
    entropy = -sum(sum(x .* log(z) + (1 - x) .* log(1 - z))) / ndata;
end