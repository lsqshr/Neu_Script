function godeep(lhidden, datasetName, numData)
    LAMBDA = 3e-3;
    LAMBDASM = 1e-4;
    BETA = 3;
    sparsityParam = 0.1;
    DEBUG = false;	
    MAXITER = 400;
    
    %% load data
    if strcmp(datasetName, 'bio')
		addpath ../dataset/loader;
		[data, labels] = loaddata('../dataset/biodata.mat', ['VOLUME', 'SOLIDITY', 'CONVEXITY']);
		numClasses = 4;
	elseif strcmp(datasetName, 'MNIST')
		addpath ../dataset/MNIST
		data = loadMNISTImages('train-images.idx3-ubyte');
		labels = loadMNISTLabels('train-labels.idx1-ubyte');
		data = data(:,1 : numData);
		labels = labels(1: numData);
		labels(labels==0) = 10; % Remap 0 to 10
		numClasses = 10;
    end

    %% train autoencoders
    features = data;
    T = cell(size(lhidden), 1);
    
    for i = 1 : length(lhidden)
        model = bioSparseTrain(lhidden(i), features, ...
                               sparsityParam, LAMBDA, BETA, MAXITER, DEBUG, false);
        [W, b] = extractParam(model.theta, lhidden(i), size(features, 1));
	
        [features, ~, ~] = feedforward(features, W, b);
        T{i} = model.theta;
        model.hiddenFeatures = features;
    end
    
    %% train the softmax model
    softmaxModel.numClasses = numClasses;
    [~, softmaxModel] = softmax(1, model, LAMBDASM, labels, softmaxModel, false);
    
    DEBUG = false;
    
    %% finetune
    opttheta = gofinetune(T, softmaxModel, lhidden, LAMBDASM, LAMBDA, data, labels, DEBUG);
    
    %% restore W and b from finetuned opttheta
    lenSoftTheta = numel(softmaxModel.optTheta);
    softTheta = opttheta(1 : lenSoftTheta);
    softmaxModel.optTheta = reshape(softTheta, softmaxModel.numClasses, lhidden(end));
    theta = opttheta(lenSoftTheta + 1 : end);
	[W, b] = extractParam(theta, lhidden, size(data, 1));
    
    [y, ~, ~] = feedforward(data, W, b);
    
    model.hiddenFeatures = y;
    
    %% evaluate the result using 10 fold
    acc = softmax(10, model, LAMBDASM, labels, softmaxModel, true); 
    
    disp(lhidden);
    disp([BETA,LAMBDA, sparsityParam]);
    %% store the current achievement
    result{1} = datasetName;
    result{2} = acc;
    result{3} = LAMBDA;
    result{4} = LAMBDASM;
    result{5} = sparsityParam;
    result{6} = BETA;
    result{7} = MAXITER;
    result{8} = lhidden;
    result{9} = clock;
    result{10} = size(data);
    
    %% append the result to the existed list
    if exist('../dataset/results/results.mat', 'file') == 2
        results = load('../dataset/results/results.mat');
        results = results.results;
        idx = results{end};
        idx = idx + 1;
        results{idx} = result;
        results{end} = idx;
        save('../dataset/results/results.mat', 'results');
    else
        results = cell(10000, 1);
        results{1} = result;
        results{end} = 1;
        save('../dataset/results/results.mat', 'results');
    end
end
