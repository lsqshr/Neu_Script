function [cost, a, hp] = deepPreFeedforward(W, b, data, LAMBDA, p, BETA, labels, softmaxModel)
	[cost, a, hp] = preFeedforward(W, b, data, LAMBDA, p, BETA, labels,...
								 @(data, W, b)deepFeedforward(data, W, b, ...
								 	softmaxModel), @softCost);
	% use the activations of last unsupervised layer to predict
	% add the results of the classification probablities to the end
	cost = cost + 0.5 * LAMBDA * sum(softmaxModel.optTheta .^ 2);
end