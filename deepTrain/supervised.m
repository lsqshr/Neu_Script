function [acc, classacc, classf1score, sumperf, lperf, softmaxModel] = supervised(nfold, T, numClasses, lhidden, model, LAMBDASM, noiseRatio, MAXITER, labels, lossmode, validation)
    %split data and labels for testing
    inputData = model.hiddenFeatures;
    ndata = size(inputData, 2);

    %% find out which labels are going to be classified this time
    labelNums  = unique(labels);
    % split a subset of input data as test data
    %idx = randperm(ndata);
    idx = 1 : ndata;

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

    for i = 1 : nfold
        % grab the indices of test data and training data
        testidx = seg{i};
        testData = inputData(:, testidx);
        testLabels = labels(testidx);
        trainidx = setdiff(idx, testidx);
        trainData = inputData(:, trainidx);
        trainLabels = labels(trainidx);
        assert(size(trainData, 2) == size(trainLabels, 1));
        assert(size(testData, 2) == size(testLabels, 1));

        %% train softmax model
        softmaxModel = softmax(trainData, trainLabels, LAMBDASM, MAXITER);
        
        DEBUG = false;
        
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