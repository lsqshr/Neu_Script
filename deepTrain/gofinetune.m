function [opttheta] = gofinetune(T, softmaxModel, lhidden, sparsityParam, LAMBDA, BETA, data, labels)
	%% gather all the relevant theta into a vector
	opttheta = gatherVector(T, lhidden, inputSize);

    %% train finetune
    addpath ../sparseAutoencoder/minFunc/;
	options.Method = 'lbfgs'; 
	options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
	[opttheta, ~] = minFunc( @(x) finetune(x, softmaxModel, ...
							lhidden, sparsityParam, LAMBDA, BETA, data, labels), ...
		                    opttheta, options);
end