%% an example expirement script, which is being used by the author.

clear;
addpath ../deepTrain;

hiddenSize = 8;
LAMBDA = 9.5e-7;
LAMBDASM = 4.5e-5;
BETA = 13.92;
sparsityParam = 1e-5;
DEBUG = false;
MAXITER = 400;
timestr = datestr(clock);
datapath = '../dataset/NCAD331.mat';

% * the best feature combination: plus PET
% features = ['VOLUME', 'SOLIDITY', 'CONVEXITY', 'MeanIndex', 'FisherIndex', 'CMRGLC'];

% ISBI feature set
%features = {'CONVEXITY','VOLUME', 'SOLIDITY','CURVATURE', 'ShapeIndex', 'LGI'};
features = {'CONVEXITY','VOLUME', 'SOLIDITY'};

% load data and labels
[data, labels] = loaddata(datapath, features);

 acc = godeep( [hiddenSize hiddenSize], data, labels,...
          LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false);

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
%     acc = godeep( [ trials(i) trials(i)], data, labels,...
%         LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false);
%     alist = [alist; trials(i) acc];
%     curlist(i) = acc;
% end
% 
% plot and save to file
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
%    acc = godeep( [ hiddenSize hiddenSize], data, labels,...
%         LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false);
%    alist = [alist; trials(i) acc];
% end
% 
% % plot and save to file
% l = sortrows(alist, 1);
% plot(l(:, 1), l(:, 2));

%% grid tune LAMBDA using logarithm domain
% alist = [];
% 
% domain.start = 1e-7;
% domain.end = 1e-6;
% numTrials = 10;
% 
% trials = generateTrials(domain, numTrials);
% for i = 1 : numTrials
%     LAMBDA = trials(i);
%     acc = godeep( [ hiddenSize hiddenSize], data, labels,...
%         LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false);
%     alist = [alist; trials(i) acc];
% end
% 
% % plot and save to file
% l = sortrows(alist, 1);
% plot(l(:, 1), l(:, 2));


%% grid tune sparsityParam
% alist = [];
% 
% domain.start = 1e-6;
% domain.end = 1e-5;
% numTrials = 10;
% 
% trials = generateTrials(domain, numTrials);
% for i = 1 : numTrials
%     sparsityParam = trials(i);
%     acc = godeep( [ hiddenSize hiddenSize], data, labels,...
%         LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false);
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
%         LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false);
%     alist = [alist; trials(i) acc];
% end
% 
% % plot and save to file
% l = sortrows(alist, 1);
% plot(l(:, 1), l(:, 2));

beep;