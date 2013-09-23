function [M, labels] = loaddata(path, features)
	addpath ..;
	load(path);

	M = [];
    
    for i = 1 : length(features)
        M = [M data.(features{i})];
    end

	M = sigmoid(M');

	labels = data.labels;
end

function sigm = sigmoid(x)
      sigm = 1 ./ (1 + exp(-x));
end
