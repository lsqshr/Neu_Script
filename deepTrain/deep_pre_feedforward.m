function [cost, a, hp] = deep_pre_feedforward(W, b, suTheta,data, LAMBDA, p, BETA, labels, softmaxModel)
	[cost, a, hp] = pre_feedforward(W, b, data, LAMBDA, p, BETA, labels, @deep_pre_feedforward);
	cost = cost + 0.5 * LAMBDA * sum(suTheta .^ 2);
end