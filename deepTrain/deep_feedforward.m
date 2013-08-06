function hypothesis = deep_feedforward(data, W, b, softmaxModel)
	addpath ../sparse_autoencoder
	[hypothesis , ~, ~] = feedforward(data, W, b);
	hypothesis = softmaxPredict(softmaxModel, hypothesis);
end