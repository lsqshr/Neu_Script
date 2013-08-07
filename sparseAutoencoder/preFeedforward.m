% feedforward through all the instances to compute the average
% activations and gives out the current SEC
function [cost, a, hp] = preFeedforward(W, b, data, LAMBDA, p, BETA, labels, feedfun, costfun)
    nlayer = length(W) + 1;
    ninstances = size(data, 2);

    % do feedforward on all of the instances and get the cost
    ninstances = size(data, 2);
    nlayer = length(W) + 1;
    cost = 0;

    % feedforward m instances by vectorization
    [hypothesis, a, hp] = feedfun(data, W, b);

    cost = costfun(labels, hypothesis);

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

function diver = KL(p, pj)
    diver = p * log(p ./ pj) + (1 - p) * log((1 - p) ./ (1 - pj));
end
    
