function [cost, grad] = finetune(theta, lenUnsTheta, softmaxModel,...
								 visibleSize, lhidden, LAMBDA,...
								 sparsityParam, BETA, data, labels)

	addpath '../sparseAutoencoder/';

	[W, b] = extractParam(theta, lhidden, size(data, 1));

	ndatas = size(data, 2);
	nlayer = length(W) + 1;

	% for the unsupervised neural network(sparse) we need to
	% feedforward all the data before the backpropagatlion
	[cost, a, hp] = deepPreFeedforward(W, b, data,...
									 LAMBDA, sparsityParam, ...
									 BETA, labels, softmaxModel);

    % use backpropagation to get two partial derivatives
    [dW, db] = backpropagation(labels', W, b, a,...
	     hp, BETA, sparsityParam,...
	      @(hypothesis, labels) softmaxDeriv(...
	      				lhidden(length(lhidden)), softmaxModel.optTheta, softmaxModel.numClasses,...
	      				a{length(lhidden) + 1}, hypothesis,labels));

	Wgrads = cell(1, nlayer - 1);
	bgrads = cell(1, nlayer - 1);

	gradW = [];
	gradb = [];
	for l = 1 : nlayer - 1
	    Wgrads{l} = dW{l} / ndatas +...
		     	    (LAMBDA * W{l}) ; % the paritial derivative of W
	    bgrads{l} = db{l} / ndatas;
	    gradW = [gradW ; Wgrads{l}(:)];
	    gradb = [gradb ; bgrads{l}(:)];
	end
	grad = [gradW ; gradb];
end

function dJ = softmaxDeriv(inputSize ,theta, numClasses,...
							 data, hypothesis, labels)
	ndatas = size(data, 2);
	groundTruth = full(sparse(labels, 1:ndatas, 1));

	% compute cost(theta)
	% compute hTheta(x), vectorized
	M = exp(theta * data);

	% tried to avoid overflow by adding the following line, while it creates -Inf sometimes
	%M = bsxfun(@minus, M, median(M));

	hTheta = bsxfun(@rdivide, M, sum(M));
		groundTruth = full(sparse(labels, 1:ndatas, 1));

	dJ = theta' * (groundTruth - hTheta);
end