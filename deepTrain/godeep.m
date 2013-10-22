function [acc, classacc, classf1score, sumperf, lperf, softmaxModel] = godeep(lhidden, autoencodertype, data, labels, LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, MEMORYSAVE, lossmode,validation)
% lhidden       : array of the number of neurons in each hidden layer
% data          : The preloaded data in matrix
% labels        : The preloaded labels in matrix
% LAMBDA        : Relative weight of weight decay
% LAMBDASM      : Relative weight of softmax weight decay
% BETA          : The relative weight for sparsity penalty.
% sparsityParam : the target of sparsity.
% noiseRatio    : The noiseRatio for denoising autoencoder. This parameter
%                 takes effect only when 'autoencodertype == denoising'
% MAXITER       : maximum number of iterations for the optimizer(L-BFG). 
% DEBUG         : when this flag is turned on, this function will evaluate the derivatives of backpropagation of both sparse autoencoder and fine-tune stage,
%                 by compare the computed gradients and approximate numerical gradient. The correct difference should be lower than Xe-9.
%                 This implementation can fulfil thhis requirement
% MEMORYSAVE    : When this flag is turned on, the activations obtained from feedforward will not be stored and passed to backpropagation.
%                 Instead, in the backpropagation stage, the activations of each layer will be calculated again.
%                 This flag will slightly optimize the memory effciency, and slow down the whole learning progress.
% lossmode      : two options to measure reconstruction error: 'square',
%                 'square' : mean square error (MSE)
%                 'cross'  : cross-entropy

    % get the number of classes
    numClasses = length(unique(labels));
             
    %% train autoencoders (unsupervised learning)
    features = data;
    T = cell(size(lhidden), 1);
    
    for i = 1 : length(lhidden)
        model = bioSparseTrain(lhidden(i), autoencodertype, features, ...
                               sparsityParam, LAMBDA, BETA, noiseRatio, MAXITER, DEBUG, MEMORYSAVE, lossmode);
        [W, b] = extractParam(model.theta, lhidden(i), size(features, 1));
	
        [features, ~, ~] = feedforward(features, W, b);
        T{i} = model.theta;
        model.hiddenFeatures = features;
    end
    
    %% start supervised learning
    [acc, classacc, classf1score, sumperf, lperf, softmaxModel] = supervised(10, T, ...
                                     numClasses, lhidden, model, LAMBDASM, noiseRatio, MAXITER, labels, lossmode, validation);
end
