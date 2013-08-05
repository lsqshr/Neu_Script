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
	[cost, a, hp] = pre_feedforward(W, b, data, LAMBDA, sparsityParam, BETA);

    % use backpropagation to get two partial derivatives
    [dW, db] = backpropagation(data, W, b, a, hp, BETA, sparsityParam);

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


function diver = KL(p, pj)
    diver = p * log(p ./ pj) + (1 - p) * log((1 - p) ./ (1 - pj));
end
    
    
function [dW, db] = backpropagation(labels, W, b, a, hp, BETA, sparsityParam)
	% The labels here are the initial data in unsupervised machine

	nlayer = length(W) + 1;
	ninstance = size(labels, 2);
	hypothesis = a{nlayer};

	% calc the error terms
	sig_prime = (hypothesis .* (1 - hypothesis));
	errterms = cell(nlayer, 1);
	errterms{nlayer} = -(labels - hypothesis) .* sig_prime;

	for l = (nlayer - 1): -1 : 2
	    sig_prime = (a{l} .* (1 - a{l}));
	    sparsity_delta = -(sparsityParam ./ hp{l}) + (1 - sparsityParam) ./ (1 - hp{l});
	    errterms{l} = ((W{l}' * errterms{l + 1})  + BETA * repmat(sparsity_delta, 1, ninstance)) .* sig_prime;
	end

	dW = cell(nlayer - 1, 1);
	db = cell(nlayer - 1, 1);
	for l = 1 : nlayer - 1
	    % calc two partial dervatives
	    dW{l} = errterms{l + 1} * a{l}';
	    db{l} = sum(errterms{l + 1}, 2);
	end

end


% feedforward through all the instances to compute the average
% activations and gives out the current SEC
function [cost, a, hp] = pre_feedforward(W, b, data, LAMBDA, p, BETA)
    nlayer = length(W) + 1;
    ninstance = size(data, 2);

    % do feedforward on all of the instances and get the cost
    ninstance = size(data, 2);
    nlayer = length(W) + 1;
    cost = 0;

    % feedforward m instances by vectorization
    [hypothesis, a, hp] = feedforward(data, W, b);

    cost = (sum(0.5 * sum((data - hypothesis) .^ 2))) / ninstance;

    % compute J(W,b)
   	s = 0;
   	for l = 1: nlayer - 1 
   		sq = W{l}(:) .^ 2;
	    s = s + sum(sq);
   	end 
    cost = cost + 0.5 * LAMBDA * s;

    % sum up all the KL(p||^pj) to compute Jsparse
    s = 0;
    for l = 2 : nlayer - 1
        s = s + sum(KL(p, hp{l}));
    end

    cost = cost + BETA * s;
end
