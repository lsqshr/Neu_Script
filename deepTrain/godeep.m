function godeep(lhidden, datasetName, numData)
    LAMBDA = 0.0001;
    BETA = 3;
    % LAMBDA = 0;
    % BETA = 0;
    sparsityParam = 0.05;
    DEBUG = false;	
    
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

    features = data;
    T = cell(size(lhidden));
    for i = 1 : length(lhidden)
        [features, model] = deepTrain(lhidden(i), features, LAMBDA, BETA, sparsityParam, DEBUG);
        T{i} = model.theta;
        
    end
    softmaxModel.numClasses = numClasses;
    % train the softmax model
    [~, softmaxModel] = softmax(1, model, LAMBDA, labels, softmaxModel, false);

    %% finetune
    opttheta = gofinetune(T, softmaxModel, lhidden, sparsityParam, LAMBDA, BETA, data, labels);
    
    %% restore W and b from finetuned opttheta
    [W, b] = extractParam(opttheta, lhidden, size(data, 1));
    [y, ~, ~] = feedforward(data, W, b);
    
    model.hiddenFeatures = y;
    
    %% evaluate the result using 10 fold
    softmax(10, model, LAMBDA, labels, softmaxModel, true); 
   
end
