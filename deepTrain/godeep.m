function [totalf1score, acc, classacc, classf1score, sumperf, lperf, softmaxModel] = godeep(lhidden, autoencodertype, data, labels, LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, MEMORYSAVE, lossmode,validation)
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
    nfold = 10;

    % get the number of classes
    labelNums = unique(labels);
    numClasses = length(labelNums);
    
    %%data partition
    %split data and labels for testing
    ndata = size(data, 2);

    %% find out which labels are going to be classified this time
    % split a subset of input data as test data
    %idx = randperm(ndata);
    idx = 1 : ndata;
    start = 1;
    % split the data into 10 segments by moding
    seg = cell(1,10);

    start = 1;
   
    %% randomly permute the instances of each labels
    for i  = 1 : numClasses
        if i == numClasses
            nextstart = length(idx);
        else
            nextstart = find(labels == labelNums(i + 1), 1 , 'first');
        end
        idx(start : nextstart - 1) = randperm(nextstart - start) + start - 1;
        start = nextstart;
    end

    for i = 1 : nfold
        i_idx = 1 : length(idx);
        i_idx = i_idx(rem(i_idx, nfold) == i - 1);
        seg{i} = idx(i_idx);
    end

    % sum of fold accuracy
    sumacc = 0;
    lperf = cell(nfold, 1);
    % sum of confusion matrix
    sumconf = 0;
    sumperf = [0, 0, 0];
    
    % start cross fold
    for i = 1 : nfold
        %% partition data
        % grab the indices of test data and training data
        testidx = seg{i};
        testData = data(:, testidx);
        testLabels = labels(testidx);
        trainidx = setdiff(idx, testidx);
        trainData = data(:, trainidx);
        trainLabels = labels(trainidx);
        assert(size(trainData, 2) == size(trainLabels, 1));
        assert(size(testData, 2) == size(testLabels, 1));
        
        %% train autoencoders (unsupervised learning)
        tridx    = randperm(size(trainData, 2));
        features = trainData(:, tridx);
        labels   = trainLabels(tridx);
        T = cell(size(lhidden), 1);

        for j = 1 : length(lhidden)
            model = bioSparseTrain(lhidden(j), autoencodertype, features, ...
                                   sparsityParam, LAMBDA, BETA, noiseRatio, MAXITER, DEBUG, MEMORYSAVE, lossmode);
            [W, b] = extractParam(model.theta, lhidden(j), size(features, 1));

            [features, ~, ~] = feedforward(features, W, b);
            T{j} = model.theta;
            model.hiddenFeatures = features;
        end

        %% start supervised learning
        %% train softmax model
        softmaxModel = softmax(features, trainLabels, LAMBDASM, MAXITER);
        
        %% finetune
        opttheta = gofinetune(T, softmaxModel, lhidden, LAMBDASM, noiseRatio, MAXITER, trainData, trainLabels, DEBUG, lossmode);
        
        %% restore W and b from finetuned opttheta
        lenSoftTheta = numel(softmaxModel.optTheta);
        softTheta = opttheta(1 : lenSoftTheta);
        softmaxModel.optTheta = reshape(softTheta, softmaxModel.numClasses, lhidden(end));
        theta = opttheta(lenSoftTheta + 1 : end);
        [W, b] = extractParam(theta, lhidden, size(trainData, 1));
        
        %% use the finetuned unsupervised parameters to get new extracted meaningful features
        [y, ~, ~] = feedforward(testData, W, b);
    
        [~, pred] = softmaxPredict(softmaxModel, y);

        acc = mean(testLabels(:) == pred(:));
        fprintf('Accuracy: %0.3f%%\n', acc * 100);
        sumacc = sumacc + acc;
        
        % add up the confusion matrix
        sumconf = sumconf + confusionmat(testLabels(:), pred(:));
        
        perf = classperf(testLabels(:), pred(:));
        sumperf = sumperf + [perf.CorrectRate, perf.Sensitivity, perf.Specificity];
        
        lperf{i} = perf;
        
        if validation == true %% when using validation, we only do it with one iteration
            nfold = 1;
            break;
        end
    end
    
    %% calc the performance
    disp({nfold 'fold accuracy:'});
    acc = sumacc / nfold;
    disp(acc);

    % compute the averaged confusion matrix and get the presicion of
    % each class

    conf = sumconf ./ nfold;
    classrecall = (diag(conf) ./ sum(conf, 2))';
    classacc = diag(conf)' ./ sum(conf, 1);
    classacc(isnan(classacc)) = 0;
    classf1score = 2 * classacc .* classrecall ./ (classacc + classrecall);
    totalf1score = mean(classf1score);
    disp('confmat:');
    disp(conf);
    disp('accuracy for each class:');
    disp(classacc);
    disp('recall for each class:');
    disp(classrecall);
    disp('f1 score for each class:');
    disp(classf1score);
    disp('total f1score:');

    sumperf = sumperf ./ nfold;
    disp('classification performance: ')
    disp(sumperf);
end
