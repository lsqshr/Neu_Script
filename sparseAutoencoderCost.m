function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             LAMBDA, sparsityParam, BETA, data)

	% visibleSize: the number of input units (probably 64) 
	% hiddenSize: the number of hidden units (probably 25) 
	% LAMBDA: weight decay parameter
	% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
	%                           notes by the greek alphabet rho, which looks like a lower-case "p").
	% BETA: weight of sparsity penalty term
	% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
	  
	% The input theta is a vector (because minFunc expects the parameters to be a vector). 
	% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
	% follows the notation convention of the lecture notes. 

	W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
	W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
	b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
	b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

	W{1} = W1;
	W{2} = W2;
	b{1} = b1;
	b{2} = b2;

	% Cost and gradient variables (your code needs to compute these values). 
	% Here, we initialize them to zeros. 
	cost = 0;
	W1grad = zeros(size(W1)); 
	W2grad = zeros(size(W2));
	b1grad = zeros(size(b1)); 
	b2grad = zeros(size(b2));

	%% ---------- YOUR CODE HERE --------------------------------------
	%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
	%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
	%
	% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
	% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
	% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
	% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
	% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
	% [(1/m) \Delta W^{(1)} + \LAMBDA W^{(1)}] in the last block of pseudo-code in Section 2.2 
	% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
	% 
	% Stated differently, if we were using batch gradient descent to optimize the parameters,
	% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
	% 

	ninstance = size(data, 2);
	nlayer = length(W) + 1;

	% init two clusters of increment
	delta_W = cell(nlayer - 1, 1);
	delta_b = cell(nlayer - 1, 1);

	% init two clusters of increment
	for l = 1 : nlayer - 1
	    delta_W{l} = zeros(size(W{l}));
	    delta_b{l} = zeros(size(b{l}));
	end

	% init W and b

	%for l = 1 : nlayer - 1
	%    % init W as gaussian distribution with a range in [-(6/(n_in + n_out + 1))^-2, (6/(n_in + n_out + 1))^-2]
	%    if l == 1
	%        n_in = 0;
	%    else 
	%        n_in = self.l_nele(l - 1);
	%    end
	%    n_out = self.l_nele(l + 1);
	%    highbound = -((6/(n_in + n_out + 1))^(0.5));
	%    lowbound = -highbound;
	%    self.W{l} = lowbound + (highbound - lowbound) .* udmatrix;
	%   self.b{l} = zeros(self.l_nele(l + 1), 1);
	%end

	% for the unsupervised neural network(sparse) we need to
	% feedforward all the instances before the backpropagation
	[cost, hp] = pre_feedforward(W, b, data, LAMBDA, sparsityParam, BETA);
	    
	for m = 1 : ninstance
	    % use backpropagation to get two partial derivatives
	    [dW, db] = backpropagation(data(:, m), W, b, hp, BETA, sparsityParam);
	    
	    % update delta_W and delta_b
	    for l = 1 : nlayer - 1
	        delta_W{l} = delta_W{l} + dW{l};
	        delta_b{l} = delta_b{l} + db{l};
	    end

	end

	Wgrads = cell(1, nlayer - 1);
	bgrads = cell(1, nlayer - 1);

	for l = 1 : nlayer - 1
	    Wgrads{l} = (1 / ninstance * delta_W{l}) + LAMBDA * W{l} ; % the paritial derivative of W
	    bgrads{l} = (1 / ninstance * delta_b{l});
	end

	%-------------------------------------------------------------------
	% After computing the cost and gradient, we will convert the gradients back
	% to a vector format (suitable for minFunc).  Specifically, we will unroll
	% your gradient matrices into a vector.

	grad = [Wgrads{1}(:) ; Wgrads{2}(:) ; bgrads{1}(:) ; bgrads{2}(:)];

	end


	function sigm = sigmoid(x)
	  
	    sigm = 1 ./ (1 + exp(-x));
	end


	function diver = KL(p, pj)
            diver = p * log(p ./ pj) + (1 - p) * log((1 - p) ./ (1 - pj));
    end


	function [dW, db] = backpropagation(instance, W, b, hp, BETA, sparsityParam)
	linstance = size(instance,1);
	nlayer = length(W) + 1;

	% in case of the big dataset, im doing the feedforward twice
	% (have done it before backpropagations to compute the average activations)
	% the hp returned here will be disposed
	[hypothesis, a, disposed_hp] = feedforward(instance, W, b, []);

	% calc the error terms
	d_sigmoid = (hypothesis .* (1 - hypothesis));
	errterms = cell(nlayer, 1);
	errterms{nlayer} = -(instance - hypothesis) .* d_sigmoid;

	for l = (nlayer - 1): -1 : 2
	    d_sigmoid = (a{l} .* (1 - a{l}));
	    errterms{l} = (W{l}' * errterms{l + 1}) .* d_sigmoid + ...
	            BETA * (-(sparsityParam ./ hp{l}) + (1 - sparsityParam) ./ (1 - hp{l}));
	end

	dW = cell(nlayer - 1, 1);
	db = cell(nlayer - 1, 1);
	for l = 1 : nlayer - 1
	    % calc two partial dervatives
	    dW{l} = errterms{l + 1} * a{l}';
	    db{l} = errterms{l + 1};
	end

end


function [y, a, hp] =  feedforward(inputs, W, b, hp)
	% This method support nlayer feedforward
	% W: a cell list of weight matrices
	% b: a cell list of bias term vectors
	% hp: the sparsity parameter, a cell of $nlayer vectors
    nlayer = length(W) + 1;
    z = cell(nlayer, 1);
    a = cell(nlayer, 1);
    z{1} = inputs; % the first layer is just the inputs
    a{1} = inputs; 

    % in case we do not need to 
    if isempty(hp)
		hp = cell(0);
	end

    for l = 1 : nlayer - 1
        z{l + 1} = W{l} * a{l} + b{l};
        a{l + 1} = sigmoid(z{l + 1});
	    if ~isempty(hp) 
	        hp{l + 1} = increment_p(a{l + 1}, hp{l + 1});
	    end
    end
    
     y = a{nlayer};
end

function hp = increment_p(a, hp)
    % args: 
    % a vector of the activation of each unit in this layer
    % l the index of the current layer
    if isempty(hp) 
        hp = a;
    else
        hp = hp + a;
    end
end

% feedforward through all the instances to compute the average
% activations and gives out the current SEC
function [cost, hp] = pre_feedforward(W, b, data, LAMBDA, p, BETA)
    nlayer = length(W) + 1;
    ninstance = size(data, 2);
    hp = cell(1, nlayer);

    % do one feedforward on all of the instances and get the cost
    % the sparsity parameter will be fillt up in self.JWb()
    [cost, hp] = JWb(W,b, data, LAMBDA);

    % cal the average for each vector a(x)
    for l = 1 : nlayer
        hp{l} = hp{l} / ninstance;
    end

    % sum up all the KL(p||^pj) to compute Jsparse
    s = 0;
    for l = 2 : nlayer - 1
        s = s + sum(KL(p, hp{l}));
    end

    cost = cost + BETA * s;
end

% get the J(W,b), not called in the supervised machine
function [cost, hp] = JWb(W, b, data, LAMBDA)
    ninstance = size(data, 2);
    nlayer = length(W) + 1;
   	hp = cell(1, nlayer);
    cost = 0;
    for m = 1 : ninstance
        instance = data(:, m);
        linstance = size(instance,1);
        [hypothesis, a, hp] = feedforward(instance, W, b, hp);
        label = instance;
        cost = cost + 0.5 * sum((label - hypothesis) .^ 2);
    end

    % compute J(W,b)
    s = 0;
    for l = nlayer - 1
        s = s + sum(sum(W{l} .^ 2));
    end

    cost = cost / ninstance + LAMBDA * s;
end