%% preset parameters
clear;
hiddenSize      =120;
LAMBDA          = 3e-7;
LAMBDASM        = 6e-6;
BETA            = 15;
sparsityParam   = 0.05;
DEBUG           = false;
MAXITER         = 400;
timestr         = datestr(clock);
noiseRatio      = 2;
autoencodertype = 'traditional';
lossmode        = 'square';
datapath        = 'NCMCI578.mat';
% features        = {'CONVEXITY','VOLUME', 'SOLIDITY','CURVATURE', 'ShapeIndex', 'LGI'};
features        = {'VOLUME'};
if strcmp(autoencodertype, 'denoising')
    LAMBDA = 0;
end

[data, labels] = loaddata(datapath, features);

% [acc, classacc, classf1score, sumperf, lperf, softmaxModel] = godeep( [ hiddenSize hiddenSize ],...
%                                                                           autoencodertype, data, labels,...
%                                                                           LAMBDA, LAMBDASM, BETA, sparsityParam,...
%                                                                           noiseRatio, MAXITER, DEBUG, false, lossmode);

%% final evaluation using 10 trials
for i = 1 : 10
    [acc, classacc, classf1score, sumperf, lperf, softmaxModel] = godeep( [ hiddenSize hiddenSize ],...
                                                                          autoencodertype, data, labels,...
                                                                          LAMBDA, LAMBDASM, BETA, sparsityParam,...
                                                                          noiseRatio, MAXITER, DEBUG, false, lossmode);
    perf.acc(i)    = sumperf(1);
    perf.sen(i)    = sumperf(2);
    perf.spe(i)    = sumperf(3);
    
    for j = 1 : numel(classacc)
        name        = strcat('klass', int2str(j));
        perf.(name)(i) = classacc(j);
    end    
end

fn = fieldnames(perf);
for i = 1 : numel(fn)
    name         = fn{i};    
    perf.(strcat(name,'mean'))  = mean(perf.(name));
    perf.(strcat(name,'std'))   = std(perf.(name));    
end

disp('tuned parameters:****************');
disp('hidden size:');
disp(hiddenSize);
disp('beta:');
disp(BETA);
disp('sparsity');
disp(sparsityParam);
disp('LAMBDA');
disp(LAMBDA);
disp('LAMBDASM');
disp(LAMBDASM);
disp('*********************************');
disp 'final perf:';

for i = 1 : numel(fn)
    name = strcat(fn{i}, 'mean');
    disp(strcat(name, ':'));
    disp(perf.(name));
    
    name = strcat(fn{i},'std');
    disp(strcat(name, ':'));
    disp(perf.(name));
end
                                                                      
beep;