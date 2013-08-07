function deepTrain(lhidden)
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

		model = bioSparseTrain(hiddenSize, inputs, 0.05, 0.0001, 3, 400, false, false);

		% extract the W(1) and b(1) from the theta vector
		% W{i} = reshape(model.theta(1 : hiddenSize * visibleSize), hiddenSize, visibleSize); 
		% b{i} = model.theta(2 * hiddenSize * visibleSize + 1 : 2 * hiddenSize * visibleSize + hiddenSize);
		T{i} = model.theta;
	end

	% train softmax parameters using the features from the last unsupervised layer
	[acc, softmaxModel] = softmax(1, model); % when set nfold to 1, no test data is splitted disp 'softmax train done';

	% concat the whole network from each layer together 
	% fine-tune!!!
	%finetune(T, softmaxModel, size(data, 1),...
	%		 lhidden, 0.05, 0.0001, 3, data, labels);

	%  Use minFunc to minimize the function
	addpath ../sparseAutoencoder/minFunc/
	options.Method = 'lbfgs'; 
	options.maxIter = 200;	  % Maximum number of iterations of L-BFGS to run 

	thetaW = [];
	thetaB = [];
	% gather all the relevant theta into a vector
	for i = 1 : length(T)
		hiddenSize = lhidden(i);

		if i == 1	
			visibleSize = size(data, 1);
		else
			visibleSize = lhidden{i - 1};
		end

		W = T{i}(1 : hiddenSize * visibleSize);
		b = T{i}(2 * hiddenSize * visibleSize + 1 :...
						 2 * hiddenSize * visibleSize + hiddenSize);
		thetaW = [thetaW ; W];
		thetaB = [thetaB ; b];
	end

	theta = [thetaW ; thetaB];
	lenUnsTheta = length(theta);

	[opttheta, cost] = minFunc( @(x) finetune(x, lenUnsTheta, softmaxModel, size(data, 1),...
									 lhidden, 0.05, 0.0001, 3, data, labels), ...
	  	                             theta, options);

	% grab random data from the initial dataset and feedforward the whole network
	% and observe the accuracy
end



