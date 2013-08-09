function [cost, grad] = finetune(theta, lenUnsTheta, softmaxModel,...
								 visibleSize, lhidden, LAMBDA,...
								 sparsityParam, BETA, data, labels)

	addpath '../sparseAutoencoder/';

	% get the number of digits in theta is for W
	s = 0;
	for i = 1 : length(lhidden)
		if i == 1
			s = s + size(data, 1) * lhidden(i);
		else
			s = s + lhidden(i) * lhidden(i - 1);
		end
	end

	bstart = s + 1;
	bTheta = theta(bstart : end);

	for i = 1 : length(lhidden) 
		hiddenSize = lhidden(i);

		if i == 1	
			visibleSize = size(data, 1);
		else
			visibleSize = lhidden(i - 1);
		end

		W{i} = reshape(theta((hiddenSize * visibleSize) * (i - 1) + 1 :...
								 hiddenSize * visibleSize * i),...
								 hiddenSize, visibleSize);
		b{i} = bTheta((i - 1) * hiddenSize + 1 : ...
					  i * hiddenSize);
	end

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