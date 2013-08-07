function deriv = softmaxDeriv(ninstancess, theta, hypothesis, labels)

ninstancess = size(data, 2);

groundTruth = full(sparse(labels, 1:ninstancess, 1));

% compute gradients
prob = groundTruth - hypothesis;
deriv = theta' * prob(:)
end
