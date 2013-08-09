function [acc, softmaxModel] = softmax(nfold, model, numClasses, lambda, labels)
    addpath ../softmax/
    addpath ../dataset/
    load biodata;

    %split data and labels for testing
    inputData = model.hiddenFeatures;
    inputSize = size(inputData, 1); % Size of input vector 
    ndata = size(inputData, 2);

    % split a subset of input data as test data
    ntest = round(ndata / nfold);
    idx = randperm(ndata);

    sumacc = 0;
    for i = 1 : nfold
        segsize = round(ndata / nfold);
        % grab the indices of test data and training data
        testidx = [];

        % when 1 fold, we just train without spliting the data
        if nfold ~= 1
            if i == nfold
                testidx = idx((i - 1) * segsize + 1 : end);
            else
                testidx = idx((i - 1) * segsize + 1 : i * segsize);
            end
            testData = inputData(:, testidx);
            testLabels = labels(testidx);
            trainidx = setdiff(idx, testidx);
            trainData = inputData(:, trainidx);
            trainLabels = labels(trainidx);
            assert(size(trainData, 2) == size(trainLabels, 1));
            assert(size(testData, 2) == size(testLabels, 1));
        else
            trainData = inputData;
            trainLabels = labels;
            assert(size(trainData, 2) == size(trainLabels, 1));
        end


        DEBUG = false; % Set DEBUG to true when debugging.
        if DEBUG
            inputSize = 8;
            inputData = randn(8, 100);
            labels = randi(10, 100, 1);
        end

        % Randomly initialise theta
        theta = 0.005 * randn(numClasses * inputSize, 1);


        if DEBUG
        %%======================================================================
        %% STEP 2: Implement softmaxCost
        %
        %  Implement softmaxCost in softmaxCost.m. 
        [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, inputData, labels);

        %%======================================================================
        %% STEP 3: Gradient checking
        %
        %  As with any learning algorithm, you should always check that your
        %  gradients are correct before learning the parameters.
        % 

            addpath ../sparseAutoencoder
            numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, ...
                                            inputSize, lambda, trainData, trainLabels), theta);

            % Use this to visually compare the gradients side by side
            disp([numGrad grad]); 

            % Compare numerically computed gradients with those computed analytically
            diff = norm(numGrad-grad)/norm(numGrad+grad);
            disp(diff); 
            % The difference should be small. 
            % In our implementation, these values are usually less than 1e-7.

            % When your gradients are correct, congratulations!
        end
        %%======================================================================
        %% STEP 4: Learning parameters

        options.maxIter = 400;
        disp({'training using ', size(trainData, 2) , ' instances'});
        softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                                    trainData, trainLabels, options);
                                  
        % Although we only use 100 iterations here to train a classifier for the 
        % MNIST data set, in practice, training for more iterations is usually
        % beneficial.

        %%======================================================================
        %% STEP 5: Testing

        if nfold == 1
            continue
        end

        [~, pred] = softmaxPredict(softmaxModel, testData);

        acc = mean(testLabels(:) == pred(:));
        fprintf('Accuracy: %0.3f%%\n', acc * 100);
        sumacc = sumacc + acc;
    end

    if nfold ~= 1
        disp({nfold  'fold accuracy:'});
        acc = sumacc / nfold;
        disp(acc);
    else
        acc = 0;
    end

end