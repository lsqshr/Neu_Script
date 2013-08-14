function [opttheta] = gofinetune(T, softmaxModel, lhidden, sparsityParam, LAMBDA, BETA, data, labels)
	%% gather all the relevant theta into a vector
	thetaW = [];
	thetaB = [];
	
	for i = 1 : length(T)
		hiddenSize = lhidden(i);

		if i == 1	
			visibleSize = size(data, 1);
		else
			visibleSize = lhidden(i - 1);
		end

		W = T{i}(1 : hiddenSize * visibleSize);
		b = T{i}(2 * hiddenSize * visibleSize + 1 :...
						 2 * hiddenSize * visibleSize + hiddenSize);

		thetaW = [thetaW ; W];
		thetaB = [thetaB ; b];
	end

	opttheta = [thetaW ; thetaB];

    %% train finetune
    addpath ../sparseAutoencoder/minFunc/;
	options.Method = 'lbfgs'; 
	options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
	[opttheta, ~] = minFunc( @(x) finetune(x, softmaxModel, ...
							lhidden, sparsityParam, LAMBDA, BETA, data, labels), ...
		                    opttheta, options);
end