testImages  = loadMNISTImages('t10k-images.idx3-ubyte');
testLabels  = loadMNISTLabels('t10k-labels.idx1-ubyte');
trainImages = loadMNISTImages('train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('train-labels.idx1-ubyte');

data        = [trainImages testImages];
labels      = [trainLabels; testLabels];

data        = data(:, 1:5000);
labels      = labels(1:5000);
labels      = labels + 1;

hiddenSize      = 200;
LAMBDA          = 2e-4;
LAMBDASM        = 2e-3;
BETA            = 4;
sparsityParam   = 0.05;
DEBUG           = false;
MAXITER         = 400;
timestr         = datestr(clock);
noiseRatio      = 8;
autoencodertype = 'traditional';
lossmode        = 'square';
validation      = true;
if strcmp(autoencodertype, 'denoising')
    LAMBDA = 0;
end

godeep( [ hiddenSize hiddenSize],  autoencodertype, data, labels,...
          LAMBDA, LAMBDASM, BETA, sparsityParam, noiseRatio, MAXITER, DEBUG, false, lossmode, validation);