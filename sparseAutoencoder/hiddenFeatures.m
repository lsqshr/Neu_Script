function hiddenFeatures = hiddenFeatures(theta, hiddenSize, visibleSize, data, LAMBDA, BETA)
	W{1} = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
	W{2} = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
	b{1} = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
	b{2} = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

    nlayer = length(W) + 1;
    ninstances = size(data, 2);

    % do feedforward on all of the instances and get the cost
    ninstances = size(data, 2);
    nlayer = length(W) + 1;
    cost = 0;

    % feedforward m instances by vectorization
    [~, a, ~] = feedforward(data, W, b);

    hiddenFeatures = a{2};
end
