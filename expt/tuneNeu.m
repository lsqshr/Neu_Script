% script to autotune the deep learning archietecture using the autotune
% tool I wrote
% author: Siqi
clear;
%% preset parameters

hiddenSize      = 200;
LAMBDA          = 2.7e-6;
LAMBDASM        = 3e-5;
BETA            = 8;
sparsityParam   = 0.05;
DEBUG           = false;
MAXITER         = 400;
timestr         = datestr(clock);
noiseRatio      = 8;
autoencodertype = 'traditional';
lossmode        = 'square';
datapath        = 'new758.mat';
features        = {'CONVEXITY','VOLUME', 'LGI', 'CURVATURE', 'ShapeIndex', 'SOLIDITY'};
%features        = {'VOLUME'};
validation      = true;
if strcmp(autoencodertype, 'denoising')
    LAMBDA = 0;
end

[data, labels] = loaddata(datapath, features);
load elastic
pvalue = elasticresult.b(:, elasticresult.fitinfo.IndexMinMSE);
idx    = find(pvalue ~= 0);
data   = data(idx, :);
% labels = labels(idx);


godeep( [ hiddenSize hiddenSize],  autoencodertype, data, labels,...
         LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, false, lossmode, validation);

options.nsample = 6;
options.depth   = 2;

% % Tune hiddenSize
% domain.start    = 30;
% domain.end      = 100;
% 
% options.round   = true;
% [para, acc] = singletune(@(hiddenSize)godeep( [ hiddenSize hiddenSize ],  autoencodertype, data, labels,...
%         LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, false, lossmode, validation), domain, options);
% 
% hiddenSize = para;
% 
% % Tune LAMBDA
% domain.start    = 1e-7;
% domain.end      = 3e-5;
% options.round   = false;
% [para, acc] = singletune(@(LAMBDA)godeep( [ hiddenSize hiddenSize ],  autoencodertype, data, labels,...
%         LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, false, lossmode, validation), domain, options);
% 
% disp(['tuned LAMBDA : ', para, 'acc:', acc]);
% LAMBDA = para;

%Tune LAMBDASM
% 
% domain.start    = 1e-7;
% domain.end      = 3e-2;
% options.round   = false;
% [para, acc] = singletune(@(LAMBDASM)godeep( [hiddenSize hiddenSize], autoencodertype, data, labels, LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, false, lossmode,validation), domain, options);
% 
% disp(['tuned LAMBDASM : ', para, 'acc:', acc]);
% LAMBDASM = para;
% 
% %% Tune BETA
% 
% domain.start    = 0.5;
% domain.end      = 10;
% options.round   = false;
% [para, acc] = singletune(@(BETA)godeep( [ hiddenSize hiddenSize ],  autoencodertype, data, labels,...
%         LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, false, lossmode, validation), domain, options);
% 
% disp(['tuned BETA : ', para, 'acc:', acc]);
% BETA = para;

% % Tune sparsityParam
% 
% domain.start    = 1e-4;
% domain.end      = 5e-1;
% options.round   = false;
% [para, acc] = singletune(@(sparsityParam)godeep( [ hiddenSize hiddenSize ],  autoencodertype, data, labels,...
%         LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, false, lossmode, validation), domain, options);
% 
% disp(['tuned sparsityParam : ', para, 'acc:', acc]);
% sparsityParam = para;

%% final evaluation using 10 trials
% for i = 1 : 10
%     [acc, classacc, classf1score, sumperf, lperf, softmaxModel] = godeep( [ hiddenSize hiddenSize ],...
%                                                                           autoencodertype, data, labels,...
%                                                                           LAMBDA, LAMBDASM, BETA, sparsityParam,...
%                                                                           noiseRatio, MAXITER, DEBUG, false, lossmode, validation);
%     perf.acc(i)    = sumperf(1);
%     perf.sen(i)    = sumperf(2);
%     perf.spe(i)    = sumperf(3);
%     
%     for j = 1 : numel(classacc)
%         name        = strcat('klass', int2str(j));
%         perf.(name)(i) = classacc(j);
%     end    
% end
% 
% fn = fieldnames(perf);
% for i = 1 : numel(fn)
%     name         = fn{i};    
%     perf.(strcat(name,'mean'))  = mean(perf.(name));
%     perf.(strcat(name,'std'))   = std(perf.(name));    
% end
% 
% disp('tuned parameters:****************');
% disp('hidden size:');
% disp(hiddenSize);
% disp('beta:');
% disp(BETA);
% disp('sparsity');
% disp(sparsityParam);
% disp('LAMBDA');
% disp(LAMBDA);
% disp('LAMBDASM');
% disp(LAMBDASM);
% disp('*********************************');
% disp 'final perf:';

% for i = 1 : numel(fn)
%     name = strcat(fn{i}, 'mean');
%     disp(strcat(name, ':'));
%     disp(perf.(name));
%     
%     name = strcat(fn{i},'std');
%     disp(strcat(name, ':'));
%     disp(perf.(name));
% end

beep;