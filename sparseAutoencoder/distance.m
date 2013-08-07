function cost = distance(labels, hypothesis)
	ninstances = length(labels);
    cost = (sum(0.5 * sum((labels - hypothesis) .^ 2))) / ninstances;
end
