function model = bioSparseTrain(hiddenSize, data, sparsityParam, LAMBDA, BETA, MAXITER, DEBUG, MEMORYSAVE, LOSSMODE)
  %%======================================================================
  %% STEP 0: Here we provide the relevant parameters values that will
  %  allow your sparse autoencoder to get good filters; you do not need to 
  %  change the parameters below.
  addpath ../sparseAutoencoder/
  addpath ../dataset/
  addpath ../dataset/loader/

  if DEBUG == true
    hiddenSize = 2;
  end

  %%======================================================================
  %% STEP 1: load data from the dataset into a vector 
  %

  visibleSize = size(data, 1);   % number of input units 

  if DEBUG == true
    disp(size(data));
  end
  %displayNetwork(data(:,randi(size(data,2),200,1)),patchsize);

  %  Obtain random parameters theta
  theta = initializeParameters(hiddenSize, visibleSize);

  %% STEP 3: Gradient Checking
  if DEBUG == true
      [~, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, LAMBDA, ...
                                           sparsityParam, BETA, data, MEMORYSAVE);

      %%======================================================================
      checkNumericalGradient();
      disp('going to compute numerical gradient');

      % check your cost function and derivative calculations
      % for the sparse autoencoder.  
      numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, hiddenSize, LAMBDA, ...
                                                        sparsityParam, BETA, ...
                                                         data, MEMORYSAVE), theta);

      % Compare numerically computed gradients with the ones obtained from backpropagation
      diff = norm(numgrad-grad)/norm(numgrad+grad);
      disp(diff); % Should be small. In our implementation, these values are
                  % usually less than 1e-9.

                  % When you got this working, Congratulations!!! 

  end

  %  Randomly initialize the parameters
  if DEBUG == false

  theta = initializeParameters(hiddenSize, visibleSize);

  %  Use minFunc to minimize the function
  addpath ../sparseAutoencoder/minFunc/
  options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                            % function. Generally, for minFunc to work, you
                            % need a function pointer with two outputs: the
                            % function value and the gradient. In our problem,
                            % sparseAutoencoderCost.m satisfies this.
  options.maxIter = MAXITER;	  % Maximum number of iterations of L-BFGS to run 
  
  encoderoptions.visibleSize   = visibleSize;
  encoderoptions.hiddenSize    = hiddenSize;
  encoderoptions.LAMBDA        = LAMBDA;
  encoderoptions.sparsityParam = sparsityParam;
  encoderoptions.BETA          = BETA;
  encoderoptions.memorySave    = MEMORYSAVE;
  
  if LOSSMODE == 'squared'
    encoderoptions.lossFunc    = @squaredError;
  else
    encoderoptions.lossFunc    = @crossEntropy;
  end
  
  % start optimization
  [opttheta, ~] = minFunc( @(p) sparseAutoencoderCost(p, data, encoderoptions), ...
                                theta, options);

  % use the current cost to run feedforward on every instance
  features = hiddenFeatures(theta, hiddenSize, visibleSize, data, LAMBDA, BETA);
  % store a{2} and theta
  model.hiddenFeatures = features;
  model.theta = theta;
  model.hiddenSize = hiddenSize;
  model.visibleSize = visibleSize;

  % save('../dataset/model.mat', 'model');

  end

end

