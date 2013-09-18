%% an example expirement script, which is being used by the author.

clear;
addpath ../deepTrain;

hiddenSize = 11;
LAMBDA = 4.9733e-7;
LAMBDASM = 2.6157e-5;
BETA = 11;
sparsityParam = 0.3644;
DEBUG = false;
MAXITER = 400;
timestr = datestr(clock);
dataset = 'NCAD';

% * the best feature combination:
% features = ['VOLUME', 'SOLIDITY', 'CONVEXITY', 'MeanIndex', 'FisherIndex', 'CMRGLC'];

% ISBI feature set
features = {'CONVEXITY','VOLUME', 'SOLIDITY','CURVATURE', 'ShapeIndex', 'LGI'};

acc = godeep( [hiddenSize hiddenSize], dataset, features, 0,...
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
%     acc = godeep( [ trials(i) trials(i)], dataset, features,...
%         0, LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false);
%     alist = [alist; trials(i) acc];
%     curlist(i) = acc;
% end
% 
% % plot and save to file
% l = sortrows(alist, 1);
% f = plot(l(:, 1), l(:, 2));
% fname = strcat('hiddenSize', strrep(timestr,':','-'), '.png');
% saveas(f, fname);


%% grid tune hidden unites by linear step size
% alist = [];
% for hiddenSize = 63 : 66
%     acc = godeep( [ hiddenSize hiddenSize], 'super331', features, 0, LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false);
%     alist = [alist; hiddenSize acc];
% end
% 
% %% plot and save to file
% f = plot(alist(:, 1), alist(:, 2));
% saveas(f, 'hiddenSizePlot.png');

%% see what result can be produced by 65
% alist = [];
% for i = 1 : 10
%     acc = godeep( [65 65], 'super331', features, 0, LAMBDA, LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false);
%     alist = [alist; acc];
% end

% %% plot and save to file
% f = plot(alist(:));
% saveas(f, 'hiddenSizePlot.png');

%% grid tune BETA
% alist = [];
% 
% domain.start = 3;
% domain.end = 15;
% numTrials = 10;
% 
% trials = generateTrials(domain, numTrials);
% for i = 1 : numTrials
%     acc = godeep( [hiddenSize hiddenSize ], dataset, features,...
%         0, LAMBDA, LAMBDASM, trials(i), sparsityParam, MAXITER, DEBUG, false);
%     alist = [alist; trials(i) acc];
% end
% 
% % plot and save to file
% l = sortrows(alist, 1);
% f = plot(l(:, 1), l(:, 2));
% fname = strcat('beta', strrep(timestr,':','-'), '.png');
% saveas(f, fname);

%% grid tune LAMBDA using logarithm domain
% alist = [];
% 
% domain.start = 1e-7;
% domain.end = 1e-6;
% numTrials = 10;
% 
% trials = generateTrials(domain, numTrials);
% for i = 1 : numTrials
%     acc = godeep( [ hiddenSize hiddenSize ], dataset, features,...
%         0, trials(i), LAMBDASM, BETA, sparsityParam, MAXITER, DEBUG, false);
%     alist = [alist; trials(i) acc];
% end
% 
% % plot and save to file
% l = sortrows(alist, 1);
% f = plot(l(:, 1), l(:, 2));
% fname = strcat('lambda', strrep(timestr,':','-'), '.png');
% saveas(f, fname);

%% grid tune sparsityParam
% alist = [];
% 
% domain.start = 0.36;
% domain.end = 0.37;
% numTrials = 50;
% 
% trials = generateTrials(domain, numTrials);
% for i = 1 : numTrials
%     acc = godeep( [ hiddenSize hiddenSize ], dataset, features,...
%         0, LAMBDA, LAMBDASM, BETA, trials(i), MAXITER, DEBUG, false);
%     alist = [alist; trials(i) acc];
% end
% 
% % plot and save to file
% l = sortrows(alist, 1);
% f = plot(l(:, 1), l(:, 2));
% fname = strcat('lambdasm', strrep(timestr,':','-'), '.png');
% saveas(f, fname);

%% MAXITER

%% LAMBDASM
% alist = [];
% 
% domain.start = 1e-5;
% domain.end = 1e-4;
% numTrials = 50;
% 
% trials = generateTrials(domain, numTrials);
% for i = 1 : numTrials
%     acc = godeep( [ hiddenSize hiddenSize ], dataset, features,...
%         0, LAMBDA, trials(i), BETA, sparsityParam, MAXITER, DEBUG, false);
%     alist = [alist; trials(i) acc];
% end
% 
% % plot and save to file
% l = sortrows(alist, 1);
% f = plot(l(:, 1), l(:, 2));
% fname = strcat('lambdasm', strrep(timestr,':','-'), '.png');
% saveas(f, fname);