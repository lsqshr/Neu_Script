function [cost, grad] = finetune(theta, softmaxModel,...
								 lhidden, LAMBDASM, LAMBDA, noiseRatio,...
								 data, labels, LOSSMODE)
    LAMBDA = 0; % try not using LAMBDA
	% split the softTheta from the long one
    softTheta = theta(1 : numel(softmaxModel.optTheta));
    softmaxModel.optTheta = reshape(softTheta, softmaxModel.numClasses, lhidden(end));
    
    theta = theta(numel(softmaxModel.optTheta) + 1 : end);
	[W, b] = extractParam(theta, lhidden, size(data, 1));

	ndata = size(data, 2);
	nlayer = length(W) + 1;

    if strcmp('squared', LOSSMODE)
        lossFunc = @squaredError;
    else
        lossFunc = @crossEntropy;
    end
    
	% for the unsupervised neural network(sparse) we need to
	% feedforward all the data before the backpropagatlion
    noiseRatio = 10000;
	[~, a, hp] = deepPreFeedforward(W, b, data,...
									 LAMBDA, 0, ...
									 0, noiseRatio, labels, softmaxModel, lossFunc);

    % use backpropagation to get two partial derivatives
    [dW, db] = softBackpropagation(labels', W, a,...
	     hp, 0, 0,@(labels, hypothesis) softmaxDeriv(softmaxModel.optTheta,...
         labels, hypothesis));

	Wgrads = cell(1, nlayer - 1);
	bgrads = cell(1, nlayer - 1);

	gradW = [];
	gradb = [];
    for l = 1 : nlayer - 1
        Wgrads{l} = dW{l} / ndata +...
            (LAMBDA * W{l}) ; % the paritial derivative of W
        bgrads{l} = db{l} / ndata;
        
        gradW = [gradW ; Wgrads{l}(:)];
        gradb = [gradb ; bgrads{l}(:)];
    end
    
    %% calc the gradient of softTheta
    M = size(data, 2);
    groundTruth = full(sparse(labels, 1:M, 1));
    cost = -1/ndata * sum(groundTruth(:) .* log(a{nlayer + 1}(:))) + LAMBDASM/2 * sum(softmaxModel.optTheta(:) .^ 2);
    gradSoft = -1/ndata * (groundTruth - a{nlayer + 1}) * a{nlayer}' + LAMBDASM * softmaxModel.optTheta;

    %% regularize only the softmax weights 
    %cost = cost + LAMBDA * 0.5 * sum(sum((softmaxModel.optTheta) .^ 2));
    
    %% concat the theta together
	grad = [gradSoft(:); gradW ; gradb];
end

function dJ = softmaxDeriv(theta, labels, hypothesis)
	% compute cost(theta)
	% compute hTheta(x), vectorized
    labels = full(sparse(labels, 1 : length(labels), 1));
	dJ = theta' * (labels - hypothesis);
end

% this function is designed to buttress the bsxfun for gradients computing
% multiply a with every element in b
function r = sumtimes(a, b)
	r = bsxfun(@times, a, b');
end