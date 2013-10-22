function model = bioSparseTrain(hiddenSize, autoencodertype, data, sparsityParam, LAMBDA, BETA, noiseRatio, MAXITER, DEBUG, MEMORYSAVE, LOSSMODE)
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
  encoderoptions.noiseRatio    = noiseRatio;
  
  if strcmp( 'square', LOSSMODE)
    encoderoptions.lossFunc    = @squaredError;
  elseif strcmp('cross', LOSSMODE)
    encoderoptions.lossFunc    = @crossEntropy;
  else
    assert(1~=1,'The lossmode is not known. Need square or cross');
  end
  
  % start optimization
  [opttheta, ~] = minFunc( @(p) sparseAutoencoderCost(p, autoencodertype, data, encoderoptions), ...
                                theta, options);

  % use the current cost to run feedforward on every instance
  features = hiddenFeatures(opttheta, hiddenSize, visibleSize, data);
  
  % store a{2} and theta
  model.hiddenFeatures = features;
  model.theta = opttheta;
  model.hiddenSize = hiddenSize;
  model.visibleSize = visibleSize;

  % save('../dataset/model.mat', 'model');

  end

end

