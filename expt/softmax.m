function [acc, softmaxModel] = softmax(nfold, model)
    addpath ../softmax/
    addpath ../dataset/
    %load model;
    load biodata;

    softmaxModel = 0;
    % disp(acc, softmaxModel);

    numClasses = 4;     % Number of classes (MNIST images fall into 10 classes)
    lambda = 1e-4; % Weight decay parameter

    %split data and labels for testing
    inputData = model.hiddenFeatures;
    disp(size(inputData));
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
        if nfold ~= 1
            if i == nfold
                testidx = idx((i - 1) * segsize + 1 : end);
            else
                testidx = idx((i - 1) * segsize + 1 : i * segsize);
            end
            testData = inputData(:, testidx);
            testLabels = labels(testidx);
            trainidx = intersect(idx, testidx);
            trainData = inputData(:, trainidx);
            labels = data.labels;
            trainLabels = labels(trainidx);
        else
            trainData = inputData;
            trainLabels = data.labels;

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

            addpath ../sparse_autoencoder
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
        %
        %  Once you have verified that your gradients are correct, 
        %  you can start training your softmax regression code using softmaxTrain
        %  (which uses minFunc).

        options.maxIter = 400;
        softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                                    inputData, trainLabels, options);
                                  
        % Although we only use 100 iterations here to train a classifier for the 
        % MNIST data set, in practice, training for more iterations is usually
        % beneficial.

        %%======================================================================
        %% STEP 5: Testing
        %
        %  You should now test your model against the test images.
        %  To do this, you will first need to write softmaxPredict
        %  (in softmaxPredict.m), which should return predictions
        %  given a softmax model and the input data.

        %labels(labels==0) = 10; % Remap 0 to 10

        if nfold == 1
            continue
        end

        % You will have to implement softmaxPredict in softmaxPredict.m
        [pred] = softmaxPredict(softmaxModel, testData);

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