function model = bioSparseTrain(hiddenSize, data, sparsityParam, lambda, BETA, MAXITER, DEBUG, VISUAL, MEMORYSAVE)
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
  if DEBUG  
  %%======================================================================
  %% STEP 2: Implement sparseAutoencoderCost

  [~, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
                                       sparsityParam, BETA, data);

  %%======================================================================
  checkNumericalGradient();
  disp('going to compute numerical gradient');

  % check your cost function and derivative calculations
  % for the sparse autoencoder.  
  numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, hiddenSize, lambda, ...
                                                    sparsityParam, BETA, ...
                                                     data, MEMORYSAVE), theta);
d = abs(numgrad - grad);
  for i = 1 : length(numgrad)
    if d(i) > 0.01
        disp(i)
        disp([numgrad(i) grad(i)]);
    end
    
  end
                                        
  % Compare numerically computed gradients with the ones obtained from backpropagation
  diff = norm(numgrad-grad)/norm(numgrad+grad);
  
  disp(diff); % Should be small. In our implementation, these values are
              % usually less than 1e-9.

              % When you got this working, Congratulations!!! 
  pause 
  end

  %%======================================================================
  %% STEP 4: After verifying that your implementation of
  %  sparseAutoencoderCost is correct, You can start training your sparse
  %  autoencoder with minFunc (L-BFGS).

  %  Randomly initialize the parameters


  theta = initializeParameters(hiddenSize, visibleSize);

  %  Use minFunc to minimize the function
  addpath ../sparseAutoencoder/minFunc/
  options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                            % function. Generally, for minFunc to work, you
                            % need a function pointer with two outputs: the
                            % function value and the gradient. In our problem,
                            % sparseAutoencoderCost.m satisfies this.
  options.maxIter = MAXITER;	  % Maximum number of iterations of L-BFGS to run 
  %options.display = 'on';


  [opttheta, ~] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                     visibleSize, hiddenSize, ...
                                     lambda, sparsityParam, ...
                                     BETA, data, MEMORYSAVE), ...
                                theta, options);

  % use the current cost to run feedforward on every instance
  features = hiddenFeatures(opttheta, hiddenSize, visibleSize, data, lambda, BETA);
  % store a{2} and theta to a file
  model.hiddenFeatures = features;
  model.theta = opttheta;
  model.hiddenSize = hiddenSize;
  model.visibleSize = visibleSize;


  %%======================================================================
  %% STEP 5: Visualization 

  if VISUAL == true
  W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
  displayNetwork(W1, 12); 

  print -djpeg weights.jpg   % save the visualization to a file 
  end

end