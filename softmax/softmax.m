function softmaxModel = softmax(trainData, trainLabels, LAMBDASM, MAXITER)
numClasses = length(unique(trainLabels));
inputSize = size(trainData, 1); % Size of input vector         
options.maxIter = MAXITER;
disp({'training using ', size(trainData, 2) , ' instances'});
softmaxModel = softmaxTrain(inputSize, numClasses, LAMBDASM, ...
                            trainData, trainLabels, options);
end