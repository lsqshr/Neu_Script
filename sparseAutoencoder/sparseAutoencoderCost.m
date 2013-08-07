function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             LAMBDA, sparsityParam, BETA, data)

	W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
	W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
	b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
	b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

	W{1} = W1;
	W{2} = W2;
	b{1} = b1;
	b{2} = b2;

	ninstance = size(data, 2);
	nlayer = length(W) + 1;

	% for the unsupervised neural network(sparse) we need to
	% feedforward all the instances before the backpropagation
	[cost, a, hp] = preFeedforward(W, b, data, LAMBDA, sparsityParam, BETA, data, @feedforward, @distance);

    % use backpropagation to get two partial derivatives
    [dW, db] = backpropagation(data, W, b, a, hp, BETA, sparsityParam, @(labels, hypothesis)outputLayerCost(labels, hypothesis));

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

	%disp([size(Wgrads{1}(:)),size(Wgrads{2}(:)),size(bgrads{1}(:)),size(bgrads{2}(:))]);
	grad = [Wgrads{1}(:) ; Wgrads{2}(:) ; bgrads{1}(:) ; bgrads{2}(:)];

end
    
function errterm = outputLayerCost(hypothesis, labels)
	sigPrime = (hypothesis .* (1 - hypothesis));
    errterm = -(labels - hypothesis) .* sigPrime;
end