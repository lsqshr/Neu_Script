function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

ninstances = size(data, 2);

groundTruth = full(sparse(labels, 1:ninstances, 1));

cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% compute cost(theta)
% compute h_theta(x), vectorized
e_matrix = exp(theta * data);
e_sum = sum(e_matrix);
h_theta= repmat((1 ./ e_sum), numClasses, 1) .* e_matrix;
log_h = log(h_theta);
cost = -(1 / ninstances) * sum(sum(groundTruth .* log_h));
penalty = (0.5 * lambda) * sum(sum(theta .^ 2));
cost = cost + penalty;

% compute gradients
prob = groundTruth - h_theta;
s = 0;
for i = 1 : ninstances
	s = s + sumtimes(data(:, i), prob(:, i));
end

thetagrad =  s' / -ninstances + lambda * theta;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

% this function is designed to buttress the bsxfun for gradients computing
% multiply a with every element in b
function r = sumtimes(a, b)
	r = bsxfun(@times, a, b');
end