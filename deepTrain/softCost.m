function cost = softCost(labels, condP)
	ndatas = length(labels);
	groundTruth = full(sparse(labels, 1:ndatas, 1));
	% disp(size(condP));
	% disp(size(groundTruth));
	cost = (- 1 / ndatas) * sum(sum(groundTruth .* log(condP)));
end