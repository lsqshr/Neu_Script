function [cost,grad] = sparseAutoencoderCost(theta,data, options)

	W{1} = reshape(theta(1:options.hiddenSize*options.visibleSize),...
        options.hiddenSize, options.visibleSize);
	W{2} = reshape(theta(options.hiddenSize*options.visibleSize+1:2*options.hiddenSize*options.visibleSize),...
        options.visibleSize, options.hiddenSize);
	b{1} = theta(2*options.hiddenSize*options.visibleSize+1:2*options.hiddenSize*options.visibleSize+options.hiddenSize);
	b{2} = theta(2*options.hiddenSize*options.visibleSize+options.hiddenSize+1:end);

	ndata = size(data, 2);
	nlayer = length(W) + 1;

    if options.memorySave == true
        [cost, ~, hp] = preFeedforward(W, b, data, options.LAMBDA, options.sparsityParam, ...
            options.BETA, data, @feedforward, options.lossFunc, false, true);
        [dW, db] = options.memorySaveBackpropagation(data, W, b, hp, options.BETA, options.sparsityParam, ...
            @(labels, hypothesis)outputLayerCost(labels, hypothesis));
    else
        % for the unsupervised neural network(sparse) we need to
        % feedforward all the data before the backpropagation
        [cost, a, hp] = preFeedforward(W, b, data, options.LAMBDA, options.sparsityParam, ...
            options.BETA, data, @feedforward, options.lossFunc, false, true);
        
        % use backpropagation to get two partial derivatives
        [dW, db] = backpropagation(data, W, a, hp, options.BETA, options.sparsityParam, ...
            @(labels, hypothesis)outputLayerCost(labels, hypothesis));
    end

	Wgrads = cell(1, nlayer - 1);
	bgrads = cell(1, nlayer - 1);

	for l = 1 : nlayer - 1
	    Wgrads{l} = dW{l} / ndata + (options.LAMBDA * W{l}) ; % the paritial derivative of W
	    bgrads{l} = db{l} / ndata;
	end

	%-------------------------------------------------------------------
	% After computing the cost and gradient, we will convert the gradients back
	% to a vector format (suitable for minFunc).  Specifically, we will unroll
	% your gradient matrices into a vector.

	grad = [Wgrads{1}(:) ; Wgrads{2}(:) ; bgrads{1}(:) ; bgrads{2}(:)];

end
    
function errterm = outputLayerCost(labels, hypothesis)
    errterm = labels - hypothesis;
end