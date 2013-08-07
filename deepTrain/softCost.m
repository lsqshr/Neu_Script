function cost = softCost(condP, labels)
	ninstances = length(labels);
	groundTruth = full(sparse(labels, 1:ninstances, 1));
	cost = (- 1 / ninstances) * sum(sum(groundTruth * condP));
end