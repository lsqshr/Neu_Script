function finetune(unsTheta, softmaxModel, visibleSize, lhidden, ...
                                             LAMBDA, sparsityParam, BETA, data, labels)

	addpath '../sparse_autoencoder/';

	for i = 1 : length(unsTheta)
		hiddenSize = lhidden{i};
		if i == 1	
			visibleSize = size(data, 1);
		else
			visibleSize = lhidden{i - 1};
		end
		W{i} = reshape(unsTheta{i}(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
		b{i} = unsTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
	end

	ninstance = size(data, 2);
	nlayer = length(W) + 1;

	% for the unsupervised neural network(sparse) we need to
	% feedforward all the instances before the backpropagatlion
	[cost, a, hp] = deep_pre_feedforward(W, b, data,...
		 LAMBDA, sparsityParam, BETA, labels, softmaxModel);

    % use backpropagation to get two partial derivatives
    [dW, db] = backpropagation(data, W, b, a, hp, BETA, sparsityParam, @softmaxDeriv);

	Wgrads = cell(1, nlayer - 1);
	bgrads = cell(1, nlayer - 1);

	for l = 1 : nlayer - 1
	    Wgrads{l} = dW{l} / ninstance + (LAMBDA * W{l}) ; % the paritial derivative of W
	    bgrads{l} = db{l} / ninstance;
	end

	%-------------------------------------------------------------------
	% After computing the cost and gradient, we will convert the gradients back
	% to a vector format (suitable for minFunc).  Specifically, we will unroll
	% your gradient matrices into a vector.

	grad = [Wgrads{1}(:) ; Wgrads{2}(:) ; bgrads{1}(:) ; bgrads{2}(:)];

end

function dJ = softmaxDeriv(labels, theta, numClasses, inputSize )
	theta = reshape(theta, numClasses, inputSize);
	ninstances = size(data, 2);
	groundTruth = full(sparse(labels, 1:ninstances, 1));

	% compute cost(theta)
	% compute h_theta(x), vectorized
	M = exp(theta * data);

	% tried to avoid overflow by adding the following line, while it creates -Inf sometimes
	%M = bsxfun(@minus, M, median(M));

	h_theta = bsxfun(@rdivide, M, sum(M));
		groundTruth = full(sparse(labels, 1:ninstances, 1));

	dJ = theta' * (groundTruth - h_theta)
end