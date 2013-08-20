clear;
addpath ../deepTrain;
%lhiddenlist = {[4,4],[50,50],[100,100],[200,200],[300,300],[500,500],[600,600],[700,700],[800,800],[900,900],[1000,1000],[1050,1050]};
%lhiddenlist = {[300 300]};
LAMBDA = 3e-3;
LAMBDASM = 1e-4;
BETA = 3;
sparsityParam = 0.1;
DEBUG = false;
MAXITER = 400;
% for i = 1 : numel(lhiddenlist)
%     godeep(lhiddenlist{i},'bio',0, LAMBDA, BETA, sparsityParam, MAXITER, DEBUG, false);
% end

lambdaList = [3e-3 2e-3 1e-3 4e-3 0.5e-3 3e-4 ];
for i = 1 : numel(lambdaList)
    godeep([300 300],'bio',0, lambdaList(i), BETA, sparsityParam, MAXITER, DEBUG, false);
end
