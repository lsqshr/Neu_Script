function cost = distance(labels, hypothesis)
	ninstance = length(labels);
    cost = (sum(0.5 * sum((labels - hypothesis) .^ 2))) / ninstance;
end
