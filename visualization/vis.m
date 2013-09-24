clear;
load W;
%% calculate the visualization of the features
W1 = W{1}; % we only use the weights of first layer.
x = bsxfun(@rdivide, W1, sum(W1.^2));

% split each view

CONVEXITY  = x(:, 1 : 83);
VOLUME     = x(:, 84 : 166);
SOLIDITY   = x(:, 167 : 249);
CURVATURE  = x(:, 250 : 332);
ShapeIndex = x(:, 333 : 415);
LGI        = x(:, 416 : end);

sCONVEXITY  = sum(CONVEXITY);
sVOLUME     = sum(CONVEXITY);
sSOLIDITY   = sum(CONVEXITY);
sCURVATURE  = sum(CONVEXITY);
sShapeIndex = sum(CONVEXITY);
sLGI        = sum(CONVEXITY);


COMBI      = CONVEXITY + VOLUME + SOLIDITY + CURVATURE + ShapeIndex + LGI;
SUPERCOMBI = sum(COMBI, 1);
save('activationVis.mat');
subplot(2,4,1), bar(CONVEXITY);
subplot(2,4,2), bar(VOLUME);
subplot(2,4,3), bar(SOLIDITY);
subplot(2,4,4), bar(CURVATURE);
subplot(2,4,5), bar(ShapeIndex);
subplot(2,4,6), bar(LGI);
subplot(2,4,7), bar(SUPERCOMBI);
