function [M, labels] = loaddata(path, featureName)
	addpath ..;
	load(path);

	M = [];

	if any(strfind(featureName, 'VOLUME'))
		M = data.VOLUME;
	end

	if any(strfind(featureName, 'CONVEXITY'))
		M = [M  data.VOLUME];
	end

	if any(strfind(featureName, 'WAVELET'))
		M = [M  data.WAVELET];
	end

	if any(strfind(featureName, 'CMRGLC'))
		M = [M  data.CMRGLC];
	end

	if any(strfind(featureName, 'SOLIDITY'))
		M = [M  data.SOLIDITY];
	end

	if any(strfind(featureName, 'CCV'))
		M = [M  data.CCV];
	end

	M = sigmoid(M');

	labels = data.labels;
end

function sigm = sigmoid(x)
      sigm = 1 ./ (1 + exp(-x));
end
