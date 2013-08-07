function deriv = softmaxDeriv(ninstances, theta, hypothesis, labels)

ninstances = size(data, 2);

groundTruth = full(sparse(labels, 1:ninstances, 1));

% compute gradients
prob = groundTruth - hypothesis;
deriv = theta' * prob(:)
end
