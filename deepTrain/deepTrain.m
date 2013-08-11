function deepTrain(lhidden)

	LAMBDA = 0.0001;
	BETA = 3;
	sparsityParam = 0.05;

	% lhidden: the list of number of each hidden layer units

	% train each auto encoder one at a time
	% store the parameters in a matrix T, 
	% use the hidden features as the inputs for the following layer
	% the parameters will be fine tuned using softmax
	addpath ../expt/
	addpath ../dataset/loader;
	addpath ../sparseAutoencoder;
	[data, labels] = loaddata('../dataset/biodata.mat', ['VOLUME', 'SOLIDITY', 'CONVEXITY']);
	W = cell(size(lhidden));
	b = cell(size(lhidden));
	T = cell(size(lhidden));

	for i = 1 : length(lhidden)
		if i == 1
			inputs = data;
		else
			inputs = model.hiddenFeatures;
		end

		hiddenSize = lhidden(i);
		visibleSize = size(inputs, 1);

		model = bioSparseTrain(hiddenSize, inputs, sparsityParam, LAMBDA, BETA, 400, false, false);

		T{i} = model.theta;
	end

	% train softmax parameters using the features from the last unsupervised layer
	disp 'start to train the softmax using a{n - 1}';
	% when set nfold to 1, no test data is splitted disp 'softmax train done';
	[acc, softmaxModel] = softmax(1, model, 4, LAMBDA, labels, 0, false); 

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

	theta = [thetaW ; thetaB];
	lenUnsTheta = length(theta);
	disp 'fine-tuning';
	[opttheta, cost] = minFunc( @(x) finetune(x, lenUnsTheta, softmaxModel, size(data, 1),...
									 lhidden, sparsityParam, LAMBDA, BETA, data, labels), ...
	  	                             theta, options);


	model.theta = opttheta;
	[W, b] = extractParam(theta, lhidden, size(data, 1));
	% grab random data from the initial dataset and feedforward the whole network
	% and observe the accuracy
	% for the unsupervised neural network(sparse) we need to
	% feedforward all the data before the backpropagatlion
	[cost, a, hp] = preFeedforward(W, b, data, LAMBDA, sparsityParam, ...
								   BETA, data, @feedforward, @distance, false, false);

	model.hiddenFeatures = a{end};
	[acc, softmaxModel] = softmax(10, model, 4, LAMBDA, labels, softmaxModel, true); 
end
