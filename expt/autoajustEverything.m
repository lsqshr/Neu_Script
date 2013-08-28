%% an example expirement script, which is being used by the author.

clear;
addpath ../deepTrain;
LAMBDA = 0.5e-3;
LAMBDASM = 1e-4;
BETA = 3;
sparsityParam = 0.1;
DEBUG = false;
MAXITER = 350;

maxpatience = 5;

features = ['VOLUME', 'SOLIDITY', 'CONVEXITY'];

%% auto adjust the number of hidden units



%% auto adjust iter
iter = 100;
bestacc = 0;
patience = 0;
bestiter = iter;
while(true)
    acc = godeep([1050 1050], 'bio', features, 0, LAMBDA, LAMBDASM, BETA, sparsityParam, iter, DEBUG, false);
    if bestacc < acc
        bestacc = acc;
        bestiter = iter;
        iter = iter + 50;
        patience = 0;
    elseif patience < maxpatience;
        patience = patience + 1;
    else
        break;
    end
end
iter = bestiter;

%% auto adjust LAMBDA
bestacc = 0;
patience = 0;
LAMBDA = 0.5e-4;
bestLAMBDA = LAMBDA;
while(true)
    acc = godeep([1050 1050], 'bio', features, 0, LAMBDA, LAMBDASM, BETA, sparsityParam, iter, DEBUG, false);
    if bestacc < acc
        bestacc = acc;
        bestLAMBDA = LAMBDA;
        LAMBDA = LAMBDA * 1.2 ;
        patience = 0;
    elseif patience < maxpatience;
        patience = patience + 1;
    else
        break;
    end
end

LAMBDA = bestLAMBDA;

%% auto adjust LAMBDA
bestacc = 0;
patience = 0;
LAMBDASM = 0.5e-4;
bestLAMBDASM = LAMBDASM;

while(true)
    acc = godeep([1050 1050], 'bio', features, 0, LAMBDA, LAMBDASM, BETA, sparsityParam, iter, DEBUG, false);
    if bestacc < acc
        bestacc = acc;
        bestLAMBDASM = LAMBDASM;
        LAMBDASM = LAMBDASM * 1.2 ;
        patience = 0;
    elseif patience < maxpatience;
        patience = patience + 1;
    else
        break;
    end
end

LAMBDASM = bestLAMBDASM;


%% auto adjust BETA
bestacc = 0;
patience = 0;
BETA = 1;
bestBETA = BETA;

while(true)
    acc = godeep([1050 1050], 'bio', features, 0, LAMBDA, LAMBDASM, BETA, sparsityParam, iter, DEBUG, false);
    if bestacc < acc
        bestacc = acc;
        bestBETA = BETA;
        BETA = BETA + 0.5 ;
        patience = 0;
    elseif patience < maxpatience;
        patience = patience + 1;
    else
        break;
    end
end

BETA = bestBETA;


%% auto adjust sparsityParam
bestacc = 0;
patience = 0;
sparsityParam = 0.02;
bestsparsityParam = sparsityParam;

while(true)
    acc = godeep([1050 1050], 'bio', features, 0, LAMBDA, LAMBDASM, BETA, sparsityParam, iter, DEBUG, false);
    if bestacc < acc
        bestacc = acc;
        bestsparsityParam = sparsityParam;
        sparsityParam = sparsityParam * 1.2;
        patience = 0;
    elseif patience < maxpatience;
        patience = patience + 1;
    else
        break;
    end
end

sparsityParam = bestsparsityParam;

%% display results
disp('the best combination of hyper-parameters:');
disp('MAXITER\tLAMBDA\tLAMBDASM\tBETA\tsparsityParam');
sprintf('%d\t%f\t%f\t%f\t%f', iter, LAMBDA, LAMBDASM, BETA, sparsityParam);
