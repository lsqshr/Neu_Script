function [cost, a, hp] = deepPreFeedforward(W, b, suTheta,data, LAMBDA, p, BETA, labels, softmaxModel)
	[cost, a, hp] = preFeedforward(W, b, data, LAMBDA, p, BETA, labels,...
								 @deepFeedforward, @softCost);
	% use the activations of last unsupervised layer to predict
	% add the results of the classification probablities to the end
	cost = cost + 0.5 * LAMBDA * sum(suTheta .^ 2);
end