function [condP , pred] = softmaxPredict(softmaxModel, data)

    % softmaxModel - model trained using softmaxTrain


    % Unroll the parameters from theta
    theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix

    M = exp(theta * data);
    condP = bsxfun(@rdivide, M, sum(M));
    [~, pred] = max(condP);

end

