function cost = squaredError(labels, hypothesis)
	ndata = size(labels, 2);
    cost = (sum(0.5 * sum((labels - hypothesis) .^ 2))) / ndata;
end
