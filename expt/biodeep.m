%% an example expirement script, which is being used by the author.

clear;
addpath ../deepTrain;

lhiddenlist = {[1000 1000]};
LAMBDA = 1e-3;
LAMBDASM = 1e-4;
BETA = 3;
sparsityParam = 0.1;
DEBUG = false;
MAXITER = 10;

% * the best feature combination:
% features = ['VOLUME', 'SOLIDITY', 'CONVEXITY', 'MeanIndex', 'FisherIndex', 'CMRGLC'];

% ISBI feature set
features = ['VOLUME', 'SOLIDITY', 'CONVEXITY'];

%% grid tune hidden unites
alist = [];
hiddenSize = 4.0;
for i = 1 : 10
    rSize = round(hiddenSize);
    acc = godeep( [ rSize rSize], 'gold331', features, 0, LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false);
    alist = [alist; hiddenSize acc];
    hiddenSize = hiddenSize + hiddenSize ^ (log(hiddenSize) / log2(hiddenSize));
end

%% plot and save to file
f = plot(alist(:, 1), alist(:, 2));
saveas(f, 'betaplot.png');

%% grid tune BETA
% alist = [];
% for BETA = 1 : 1 : 1
%     acc = godeep([59 59], 'gold331', features, 0, LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false);
%     alist = [alist; BETA acc];
% end
%
%% plot and save to file
% saveas(plot(alist(:, 1), alist(:, 2)), 'betaplot.png');


%% grid tune LAMBDA
% lambdaList = [3e-3 2e-3 1e-3 4e-3 0.5e-3 3e-4 ];
% for i = 1 : numel(lambdaList)
%     godeep([1050 1050], 'gold331', features, 0, lambdaList(i), LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false);
% end

%% grid tune sparsityParam

