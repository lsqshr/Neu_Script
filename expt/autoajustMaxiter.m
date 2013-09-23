%% an example expirement script, which is being used by the author.

clear;
addpath ../deepTrain;

LAMBDA = 0.5e-3;
LAMBDASM = 1e-4;
BETA = 3;
sparsityParam = 0.1;
DEBUG = false;
iter = 100;

features = ['VOLUME', 'SOLIDITY', 'CONVEXITY'];
bestacc = 0;
patience = 0;
bestiter = iter;
while(true)
    acc = godeep([59 59], 'bio', features, 0, LAMBDA, LAMBDASM, BETA, sparsityParam, iter, DEBUG, false);
    if bestacc < acc
        bestacc = acc;
        bestiter = iter;
        iter = iter + 50;
    elseif patience < 3;
        patience = patience + 1;
        break;
    else
        disp('best iter for now:');
        disp(bestiter);
    end
end
