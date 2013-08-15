function [cost, a, hp] = deepPreFeedforward(W, b, data, LAMBDA, p, BETA, labels, softmaxModel)
	l = 1 : size(labels, 1);
	labels = (full(sparse(l, labels, 1.0)))';
	[cost, a, hp] = preFeedforward(W, b, data, LAMBDA, p, BETA, labels,...
								 @(data, W, b)deepFeedforward(data, W, b, ...
								 	softmaxModel), @distance, true, true);
	% use the activations of last unsupervised layer to predict
	% add the results of the classification probablities to the end
end