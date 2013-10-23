function [M, labels] = loaddata(path, features)
	addpath ..;
	load(path);

	M = [];
    
    for i = 1 : length(features)
        f = data.(features{i});
        M = [M f];
    end

    M    = M';
    %  zero-mean and unit-variance
    m    = mean(M,2);
    stdm = std(M,0, 2);
    M    = (M - repmat(m,1,size(M,2))) ./ repmat(stdm,1, size(M,2));
    % rescaling
%     M    = M - min(M(:));
%     M    = M ./ max(M(:));
    M    = sigmoid(M);

	labels = data.labels;
end
