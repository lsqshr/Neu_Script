function [acc, classacc, classf1score, sumperf, lperf, softmaxModel] = godeep(lhidden, autoencodertype, data, labels, LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, MEMORYSAVE, lossmode)
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
             
    %% train autoencoders
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
    
    
    
    %% train the softmax model
    softmaxModel.numClasses = numClasses;
    [~, ~, ~, ~, ~, softmaxModel] = softmax(1, model, LAMBDASM, MAXITER, labels, softmaxModel, false);
    
    DEBUG = false;
    
    %% finetune
    opttheta = gofinetune(T, softmaxModel, lhidden, LAMBDASM, LAMBDA, noiseRatio, MAXITER, data, labels, DEBUG, lossmode);
    
    %% restore W and b from finetuned opttheta
    lenSoftTheta = numel(softmaxModel.optTheta);
    softTheta = opttheta(1 : lenSoftTheta);
    softmaxModel.optTheta = reshape(softTheta, softmaxModel.numClasses, lhidden(end));
    theta = opttheta(lenSoftTheta + 1 : end);
	[W, b] = extractParam(theta, lhidden, size(data, 1));
    
    % save W for visualization
    save('W.mat', 'W');
    
    [y, ~, ~] = feedforward(data, W, b);
    
    model.hiddenFeatures = y;
    
    %% evaluate the result using 10 fold
    [acc, classacc, classf1score, sumperf, lperf, softmaxModel] = softmax(10, model, LAMBDASM, MAXITER, labels,...
                                                                      softmaxModel, true);    
    disp(lhidden);
    disp([BETA,LAMBDA, sparsityParam]);
end
