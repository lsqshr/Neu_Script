% sample random patches from the given whitened images

function patches = sampleIMAGES(dest, numpatches, patchsize)
% sampleIMAGES
% Returns 10000 patches for training

load(dest);    % load images from disk 

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);

%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1

height = size(IMAGES, 1);
width = size(IMAGES, 2);
iheight = height - patchsize + 1;
iwidth = width - patchsize + 1;
indices = randi(10, numpatches,1);
for i = 1 : numpatches

	spx = randi(iwidth);
	spy = randi(iheight);
	image = IMAGES(spy : spy + patchsize - 1, spx : spx + patchsize - 1, indices(i));
	patches(:, i) = reshape(image.',[],1);  %'# Collect the row contents into a column vector
end

%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches);
end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
