function cost = softCost(condP, labels)
	ndatas = length(labels);
	groundTruth = full(sparse(labels, 1:ndatas, 1));
	cost = (- 1 / ndatas) * sum(sum(groundTruth * condP));
end