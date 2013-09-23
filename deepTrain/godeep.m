function acc = godeep(lhidden, data, labels, LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, MEMORYSAVE)
% lhidden : array of the number of neurons in each hidden layer
% datasetName : indicate which dataset to load. 
        % there are currently 3 options :  1. bio : the raw preprocessed feature set extracted from 3D brain images;
                                         % 2. Hangyu : further preprocessed feature set extracted from 'bio' with lower dimensions
                                         % 3. MNIST : hand writting standard dataset for testing the learner. The proper implementation should achieve an higher accuracy than approximately 96%.
                                                      % This implementation can achieve almost 100%
% numData : only work for DEBUGING(MNIST dataset)
% LAMBDA : the forgetting parameter(weight decay)
% BETA : the weight for sparsity term. controls the number of 0s in the representation
% sparsityParam : the target of sparsity. It should be a real number closed to 0. Normally we set it to 0.05
% MAXITER : maximum number of iterations for the optimizer(L-BFG). 
% DEBUG: when this flag is turned on, this function will evaluate the derivatives of backpropagation of both sparse autoencoder and fine-tune stage,
%        by compare the computed gradients and approximate numerical gradient. The correct difference should be lower than Xe-9.
%        This implementation can fulfil thhis requirement
% MEMORYSAVE : When this flag is turned on, the activations obtained from feedforward will not be stored and passed to backpropagation.
%              Instead, in the backpropagation stage, the activations of each layer will be calculated again.
%              This flag will slightly optimize the memory effciency, and slow down the whole learning progress.
% SAVEPARAM : NOT IMPLEMENTED*****When this flag is turned on, the parameters will be saved,
%             since we are using L-BFG which is a stochastic process, not guarenteeing
%             the final optimization is the best fit. You can save the
%             optimized parameters and compare the results. Then reuse them
%             for learning by loading the parameters produces the best result in cross-folding validation.

    % get the number of classes
    numClasses = length(unique(labels));
             
    %% train autoencoders
    features = data;
    T = cell(size(lhidden), 1);
    
    for i = 1 : length(lhidden)
        model = bioSparseTrain(lhidden(i), features, ...
                               sparsityParam, LAMBDA, BETA, MAXITER, DEBUG, false, MEMORYSAVE);
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
    opttheta = gofinetune(T, softmaxModel, lhidden, LAMBDASM, LAMBDA, MAXITER, data, labels, DEBUG);
    
    %% restore W and b from finetuned opttheta
    lenSoftTheta = numel(softmaxModel.optTheta);
    softTheta = opttheta(1 : lenSoftTheta);
    softmaxModel.optTheta = reshape(softTheta, softmaxModel.numClasses, lhidden(end));
    theta = opttheta(lenSoftTheta + 1 : end);
	[W, b] = extractParam(theta, lhidden, size(data, 1));
    
    % save W for visualization
    %save('W.mat', 'W');
    
    [y, ~, ~] = feedforward(data, W, b);
    
    model.hiddenFeatures = y;
    
    %% evaluate the result using 10 fold
    [acc, classacc, classf1score, sumperf, lperf, ~] = softmax(10, model, LAMBDASM, MAXITER, labels, softmaxModel, true); 

    disp(lhidden);
    disp([BETA,LAMBDA, sparsityParam]);
    %% store the current achievement
%     result{1} = datasetName;
%     result{2} = acc;
%     result{3} = LAMBDA;
%     result{4} = LAMBDASM;
%     result{5} = sparsityParam;
%     result{6} = BETA;
%     result{7} = MAXITER;
%     result{8} = lhidden;
%     result{9} = clock;
%     result{10} = size(data);
%     result{11} = classacc;
%     result{12} = classf1score;
%     result{13} = sumperf;
%     result{14} = lperf;
    
    %% append the result to the existed list
%     if exist('../dataset/results/results.mat', 'file') == 2
%         results = load('../dataset/results/results.mat');
%         results = results.results;
%         idx = results{end};
%         idx = idx + 1;
%         results{idx} = result;
%         results{end} = idx;
%         save('../dataset/results/results.mat', 'results');
%     else
%         results = cell(10000, 1);
%         results{1} = result;
%         results{end} = 1;
%         save('../dataset/results/results.mat', 'results');
%     end
end
