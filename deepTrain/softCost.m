function cost = softCost(condP, labels)
	ninstancess = length(labels);
	groundTruth = full(sparse(labels, 1:ninstancess, 1));
	cost = (- 1 / ninstancess) * sum(sum(groundTruth * condP));
end