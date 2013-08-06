function deep_train(lhidden)
	% lhidden: the list of number of each hidden layer units

	% train each auto encoder one at a time
	% store the parameters in a matrix T, 
	% use the hidden features as the inputs for the following layer
	% the parameters will be fine tuned using softmax
	addpath ../expt/
	addpath ../dataset/loader;
	addpath ../sparse_autoencoder;
	[instances, labels] = loaddata('../dataset/biodata.mat', ['VOLUME', 'SOLIDITY', 'CONVEXITY']);
	W = cell(size(lhidden));
	b = cell(size(lhidden));
	T = cell(size(lhidden));

	for i = 1 : length(lhidden)
		if i == 1
			inputs = instances;
		else
			inputs = model.hiddenFeatures;
		end

		hiddenSize = lhidden(i);
		visibleSize = size(inputs, 1);

		model = bio_sparse_train(hiddenSize, inputs, 0.05, 0.0001, 3, 400, false, false);

		% extract the W(1) and b(1) from the theta vector
		% W{i} = reshape(model.theta(1 : hiddenSize * visibleSize), hiddenSize, visibleSize); 
		% b{i} = model.theta(2 * hiddenSize * visibleSize + 1 : 2 * hiddenSize * visibleSize + hiddenSize);
		T{i} = model.theta;
	end

	% train softmax parameters using the features from the last unsupervised layer
	[acc, softmax_model] = softmax(1, model); % when set nfold to 1, no test data is splitted disp 'softmax train done';

	% concat the whole network from each layer together 
	% fine-tune!!!
	softmax_theta = softmax_model.optTheta;

	% grab random data from the initial dataset and feedforward the whole network
	% and observe the accuracy
end



