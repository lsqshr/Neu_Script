function cost = distance(labels, hypothesis)
	ndata = length(labels);
    cost = (sum(0.5 * sum((labels - hypothesis) .^ 2))) / ndata;
end
