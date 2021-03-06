function [opttheta] = gofinetune(T, softmaxModel, lhidden, LAMBDASM, noiseRatio, MAXITER, data, labels, DEBUG, lossmode)
	%% gather all the relevant theta into a vector
	opttheta = gatherVector(T, lhidden, size(data, 1));
    opttheta = [softmaxModel.optTheta(:); opttheta];
    
    
    %% check finetune deriv
    if DEBUG
        disp('going to compute numerical gradient');
        
        % check your cost function and derivative calculations
        % for the sparse autoencoder.
        [~, grad] = finetune(opttheta, softmaxModel, ...
							lhidden, LAMBDASM, MAXITER, data, labels);
        numgrad = computeNumericalGradient( @(x) finetune(x, softmaxModel, ...
							lhidden, LAMBDASM, data, labels), opttheta);
        
        % compare side by side
        disp([numgrad grad]);
        % Compare numerically computed gradients with the ones obtained from backpropagation
        diff = norm(numgrad-grad)/norm(numgrad+grad);
        
        disp(diff); % Should be small. In our implementation, these values are
        % usually less than 1e-9.
        
        % When you got this working, Congratulations!!!
        pause();
    end
    %% train finetune
    addpath ../sparseAutoencoder/minFunc/;
	options.Method = 'lbfgs'; 
	options.maxIter = MAXITER;	  % Maximum number of iterations of L-BFGS to run 
	[opttheta, ~] = minFunc( @(x) finetune(x, softmaxModel, ...
							lhidden, LAMBDASM, noiseRatio, data, labels, lossmode), ...
		                    opttheta, options);
end