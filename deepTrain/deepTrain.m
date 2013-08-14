function [features, model] = deepTrain(hiddenSize, data, LAMBDA, BETA, sparsityParam, DEBUG)
    

	% lhidden: the list of number of each hidden layer units
    lhidden = [hiddenSize];
	% train each auto encoder one at a time
	% store the parameters in a matrix T, 
	% use the hidden features as the data for the following layer
	% the parameters will be fine tuned using softmax
	addpath ../expt/;
	addpath ../sparseAutoencoder;
    addpath ../softmax/;

	model = bioSparseTrain(hiddenSize, data, ...
        sparsityParam, LAMBDA, BETA, 400, DEBUG, false);

	% train softmax parameters using the features from the last unsupervised layer
	disp 'start to train the softmax using a{n - 1}'
    
	% when set nfold to 1, no test data is splitted disp 'softmax train done';
	
    
	[W, b] = extractParam(model.theta, lhidden, size(data, 1));
	% grab random data from the initial dataset and feedforward the whole network
	% and observe the accuracy
	% for the unsupervised neural network(sparse) we need to
	% feedforward all the data before the backpropagatlion
	[features, ~, ~] = feedforward(data, W, b);
	model.hiddenFeatures = features;
end
