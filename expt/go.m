% try increasing the hiddenSize 
  %instances = sampleIMAGES('IMAGES.mat', ninstance, patchsize);
addpath ../sparse_autoencoder
addpath ../dataset/loader

[instances, labels] = loaddata('../dataset/biodata.mat', ['VOLUME', 'SOLIDITY', 'CONVEXITY']);
upper = 512
lower = 256;
lacc = zeros(1, upper - lower);
for hiddenSize = lower : upper 
	model = bio_sparse_train(hiddenSize, instances, 0.05, 0.0001, 3, 400, false, false);
	lacc(hiddenSize - lower + 1) = softmax(10, model); % using 10 fold to compute the accuracy.
	disp({'hiddenSize', hiddenSize});
end

save('hiddenSize_256_512.mat', 'lacc2');
	

