function deepTrain(lhidden, datasetName)

LAMBDA = 0.0001;
BETA = 3;
% LAMBDA = 0;
% BETA = 0;
sparsityParam = 0.05;


	% lhidden: the list of number of each hidden layer units

	% train each auto encoder one at a time
	% store the parameters in a matrix T, 
	% use the hidden features as the inputs for the following layer
	% the parameters will be fine tuned using softmax
	addpath ../expt/
	addpath ../sparseAutoencoder;
	if strcmp(datasetName, 'bio')
		addpath ../dataset/loader;
		[data, labels] = loaddata('../dataset/biodata.mat', ['VOLUME', 'SOLIDITY', 'CONVEXITY']);
		numClasses = 4;
	elseif strcmp(datasetName, 'MNIST')
		addpath ../dataset/MNIST
		data = loadMNISTImages('train-images.idx3-ubyte');
		labels = loadMNISTLabels('train-labels.idx1-ubyte');
		data = data(:,1 : 1000);
		labels = labels(1: 1000);
		labels(labels==0) = 10; % Remap 0 to 10
		numClasses = 10;
	end

	T = cell(size(lhidden));

	for i = 1 : length(lhidden)
		if i == 1
			inputs = data;
		else
			inputs = model.hiddenFeatures;
		end

		hiddenSize = lhidden(i);
		model = bioSparseTrain(hiddenSize, inputs, sparsityParam, LAMBDA, BETA, 400, false, false);

		T{i} = model.theta;
	end

	% train softmax parameters using the features from the last unsupervised layer
	disp 'start to train the softmax using a{n - 1}'
    softmaxModel.numClasses = numClasses;
	% when set nfold to 1, no test data is splitted disp 'softmax train done';
	[~, softmaxModel] = softmax(1, model, LAMBDA, labels, softmaxModel, false); 

	% concat the whole network from each layer together for funetune
	% fine tune: using the result we derived from the softmax regression to adjust parameters
	% Use minFunc to minimize the function
	addpath ../sparseAutoencoder/minFunc/;
	options.Method = 'lbfgs'; 
	options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 

	thetaW = [];
	thetaB = [];
	% gather all the relevant theta into a vector
	for i = 1 : length(T)
		hiddenSize = lhidden(i);

		if i == 1	
			visibleSize = size(data, 1);
		else
			visibleSize = lhidden(i - 1);
		end

		W = T{i}(1 : hiddenSize * visibleSize);
		b = T{i}(2 * hiddenSize * visibleSize + 1 :...
						 2 * hiddenSize * visibleSize + hiddenSize);

		thetaW = [thetaW ; W];
		thetaB = [thetaB ; b];
	end

	opttheta = [thetaW ; thetaB];
	disp 'fine-tuning';
	[opttheta, ~] = minFunc( @(x) finetune(x, softmaxModel, ...
							lhidden, sparsityParam, LAMBDA, BETA, data, labels), ...
		                    opttheta, options);


	model.theta = opttheta;
	[W, b] = extractParam(opttheta, lhidden, size(data, 1));
	% grab random data from the initial dataset and feedforward the whole network
	% and observe the accuracy
	% for the unsupervised neural network(sparse) we need to
	% feedforward all the data before the backpropagatlion
	[y, ~, ~] = feedforward(data, W, b);

	model.hiddenFeatures = y;
    softmax(10, model, LAMBDA, labels, softmaxModel, true); 
end
