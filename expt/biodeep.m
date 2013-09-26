%% an example expirement script, which is being used by the author.

clear;
addpath ../deepTrain;

lossmode = 'crossentropy';

hiddenSize = 10;
LAMBDA = 9.5e-7;
LAMBDASM = 1.75e-5;
BETA = 9;
sparsityParam = 2.2e-7;
DEBUG = false;
MAXITER = 400;
timestr = datestr(clock);
noiseRatio = 8;
datapath = '../dataset/super331.mat';

% * the best feature combination: plus PET
% features = ['VOLUME', 'SOLIDITY', 'CONVEXITY', 'MeanIndex', 'FisherIndex', 'CMRGLC'];

% ISBI feature set
%features = {'CONVEXITY','VOLUME', 'SOLIDITY','CURVATURE', 'ShapeIndex', 'LGI'};
features = {'CONVEXITY','VOLUME', 'SOLIDITY'};

% load data and labels
[data, labels] = loaddata(datapath, features);

LAMBDA = 0; % try not use weight decay
% acc = godeep( [hiddenSize hiddenSize], data, labels,...
%           LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio,MAXITER, DEBUG, false, lossmode);
% sq = 0;
% sc = 0;
% for i = 1: 5
%  acc = godeep( [hiddenSize hiddenSize], data, labels,...
%           LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false, 'squared');
%  sq = sq + acc;
%  acc = godeep( [hiddenSize hiddenSize], data, labels,...
%           LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false, 'crossentropy');
%  sc = sc + acc;
% end 
      
%% grid tune hidden unites by logarithm domain
% alist = [];
% 
% domain.start = 6;
% domain.end = 15;
% numTrials = 10;
% 
% trials = round(generateTrials(domain, numTrials));
% for i = 1 : numTrials
%     rSize = round(hiddenSize);
%     acc = godeep( [ hiddenSize hiddenSize], data, labels,...
%         LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, false, lossmode);
%     alist = [alist; trials(i) acc];
%     curlist(i) = acc;
% end
% 
% l = sortrows(alist, 1);
% f = plot(l(:, 1), l(:, 2));


%% grid tune BETA
% alist = [];
% 
% domain.start = 3;
% domain.end = 15;
% numTrials = 10;
% 
% trials = generateTrials(domain, numTrials);
% for i = 1 : numTrials
%    BETA = trials(i);
%     acc = godeep( [ hiddenSize hiddenSize], data, labels,...
%         LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, false, lossmode);
%    alist = [alist; trials(i) acc];
% end
% 
% % plot and save to file
% l = sortrows(alist, 1);
% plot(l(:, 1), l(:, 2));

%% grid tune LAMBDA using logarithm domain
% alist = [];
% 
% domain.start = 4e-3;
% domain.end = 5e-3;
% numTrials = 10;
% 
% trials = generateTrials(domain, numTrials);
% for i = 1 : numTrials
%     LAMBDA = trials(i);
%     acc = godeep( [ hiddenSize hiddenSize], data, labels,...
%          LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, false, lossmode);
%     alist = [alist; trials(i) acc];
% end
% 
% % plot and save to file
% l = sortrows(alist, 1);
% plot(l(:, 1), l(:, 2));


%% grid tune sparsityParam
% alist = [];
% 
% domain.start = 1e-7;
% domain.end = 1e-5;
% numTrials = 10;
% 
% trials = generateTrials(domain, numTrials);
% for i = 1 : numTrials
%     sparsityParam = trials(i);
%     acc = godeep( [ hiddenSize hiddenSize], data, labels,...
%          LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, false, lossmode);
%     alist = [alist; trials(i) acc];
% end
% 
% % plot and save to file
% l = sortrows(alist, 1);
% plot(l(:, 1), l(:, 2));


%% LAMBDASM
% alist = [];
% 
% domain.start = 1e-6;
% domain.end = 1e-4;
% numTrials = 10;
% 
% trials = generateTrials(domain, numTrials);
% for i = 1 : numTrials
%     LAMBDASM = trials(i);
%     acc = godeep( [ hiddenSize hiddenSize], data, labels,...
%          LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, false, lossmode);
%     alist = [alist; trials(i) acc];
% end
% 
% % plot and save to file
% l = sortrows(alist, 1);
% plot(l(:, 1), l(:, 2));

%% noiseRatio
% alist = [];
% 
% domain.start = 5;
% domain.end = 15;
% numTrials = 20;
% 
% trials = generateTrials(domain, numTrials);
% for i = 1 : numTrials
%     noiseRatio = trials(i);
%     acc = godeep( [ hiddenSize hiddenSize], data, labels,...
%         LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, false, lossmode);
%     alist = [alist; trials(i) acc];
% end
% 
% % plot and save to file
% l = sortrows(alist, 1);
% plot(l(:, 1), l(:, 2));

beep;