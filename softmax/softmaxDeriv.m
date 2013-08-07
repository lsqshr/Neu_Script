function deriv = softmaxDeriv(ndatas, theta, hypothesis, labels)

ndatas = size(data, 2);

groundTruth = full(sparse(labels, 1:ndatas, 1));

% compute gradients
prob = groundTruth - hypothesis;
deriv = theta' * prob(:)
end
