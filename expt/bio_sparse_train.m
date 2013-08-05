function bio_sparse_train()
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.
clear all;
addpath ../sparse_autoencoder/
addpath ../dataset/
addpath ../dataset/loader/

hiddenSize = 36;    % number of hidden units 
sparsityParam = 0.05;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
            		     %  in the lecture notes). 
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term       
MAXITER = 400;
DEBUG = false;
VISUAL = false;

if DEBUG == true
  hiddenSize = 2;
end

%%======================================================================
%% STEP 1: load data from the dataset into a vector 
%

%instances = sampleIMAGES('IMAGES.mat', ninstance, patchsize);
[instances, labels] = loaddata(['VOLUME', 'SOLIDITY', 'CONVEXITY']);
instances = sigmoid(instances);
visibleSize = size(instances, 1);   % number of input units 

if DEBUG == true
  disp(size(instances));
  %disp(instances);
end
%display_network(instances(:,randi(size(instances,2),200,1)),patchsize);

%  Obtain random parameters theta
theta = initializeParameters(hiddenSize, visibleSize);

%%======================================================================
%% STEP 2: Implement sparseAutoencoderCost

[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, instances);

%%======================================================================
%% STEP 3: Gradient Checking
if DEBUG == true
checkNumericalGradient();
disp('going to compute numerical gradient');

% check your cost function and derivative calculations
% for the sparse autoencoder.  
numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, hiddenSize, lambda, ...
                                                  sparsityParam, beta, ...
                                                   instances), theta);

% Compare numerically computed gradients with the ones obtained from backpropagation
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are
            % usually less than 1e-9.

            % When you got this working, Congratulations!!! 

end

%%======================================================================
%% STEP 4: After verifying that your implementation of
%  sparseAutoencoderCost is correct, You can start training your sparse
%  autoencoder with minFunc (L-BFGS).

%  Randomly initialize the parameters
if DEBUG == false

theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath ../sparse_autoencoder/minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = MAXITER;	  % Maximum number of iterations of L-BFGS to run 
%options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, instances), ...
                              theta, options);

% use the current cost to run feedforward on every instance
features = hiddenFeatures(theta, hiddenSize, visibleSize, instances, lambda, beta);
% store a{2} and theta to a file
model.hiddenFeatures = features;
model.theta = theta;
model.hiddenSize = hiddenSize;
model.visibleSize = visibleSize;

save('../dataset/model.mat', 'model');

end
%%======================================================================
%% STEP 5: Visualization 

if VISUAL == true
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
%disp(W1);
display_network(W1, 12); 

print -djpeg weights.jpg   % save the visualization to a file 
end

end

function sigm = sigmoid(x)
      sigm = 1 ./ (1 + exp(-x));
end
