% try increasing the hiddenSize 
  %data = sampleIMAGES('IMAGES.mat', ndata, patchsize);
addpath ../sparseAutoencoder
addpath ../dataset/loader

[data, labels] = loaddata('../dataset/biodata.mat', ['VOLUME', 'SOLIDITY', 'CONVEXITY']);
upper = 512;
lower = 256;
lacc = zeros(1, upper - lower);
for hiddenSize = lower : upper 
	model = bioSparseTrain(hiddenSize, data, 0.05, 0.0001, 3, 400, false, false);
	lacc(hiddenSize - lower + 1) = softmax(10, model); % using 10 fold to compute the accuracy.
end

save('hiddenSize_256_512.mat', 'lacc2');
	

