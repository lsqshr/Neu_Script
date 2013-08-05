function [M, labels] = load_feature(feature_name)
addpath ../
load biodata;
M = [];

if any(strfind(feature_name, 'VOLUME'))
	M = data.VOLUME;
end

if any(strfind(feature_name, 'CONVEXITY'))
	M = [M  data.VOLUME];
end

if any(strfind(feature_name, 'WAVELET'))
	M = [M  data.WAVELET];
end

if any(strfind(feature_name, 'CMRGLC'))
	M = [M  data.CMRGLC];
end

if any(strfind(feature_name, 'SOLIDITY'))
	M = [M  data.SOLIDITY];
end

if any(strfind(feature_name, 'CCV'))
	M = [M  data.CCV];
end

M = M';

labels = data.labels;