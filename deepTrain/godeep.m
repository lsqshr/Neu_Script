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
    softmax(10, model, LAMBDASM, labels, softmaxModel, true); 
   
end
