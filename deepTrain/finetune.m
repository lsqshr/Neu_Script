function [cost, grad] = finetune(theta, softmaxModel,...
								 lhidden, LAMBDA,...
								 sparsityParam, BETA, data, labels)

	addpath ../sparseAutoencoder/;

	[W, b] = extractParam(theta, lhidden, size(data, 1));

	ndatas = size(data, 2);
	nlayer = length(W) + 2;

	% for the unsupervised neural network(sparse) we need to
	% feedforward all the data before the backpropagatlion
	[cost, a, hp] = deepPreFeedforward(W, b, data,...
									 LAMBDA, sparsityParam, ...
									 BETA, labels, softmaxModel);

    % use backpropagation to get two partial derivatives
    [dW, db] = backpropagation(labels', W, a,...
	     hp, 0, 0,@(hypothesis, labels) softmaxDeriv(softmaxModel.optTheta,hypothesis, labels));

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

function dJ = softmaxDeriv(theta, hypothesis, labels)
	% compute cost(theta)
	% compute hTheta(x), vectorized
    labels = full(sparse(labels, 1 : length(labels, 1), 1));
	M = exp(theta * hypothesis);
    hypothesis = bsxfun(@rdivide, M, sum(M));
    dJ = theta' * (labels - hypothesis);
end