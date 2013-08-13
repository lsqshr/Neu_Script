function [hypothesis, a, hp] = deepFeedforward(data, W, b, softmaxModel)
	addpath ../sparseAutoencoder
	[hypothesis, a, hp] = feedforward(data, W, b);
	[condP, ~] = softmaxPredict(softmaxModel, hypothesis);
	a{length(a) + 1} = condP;
	hypothesis = condP;
end