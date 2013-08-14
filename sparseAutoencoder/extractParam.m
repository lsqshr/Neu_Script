function [W, b] = extractParam(theta, lhidden, inputSize)
	%% extract cells of W and b accross all the layers including the input and output layer
    nlayer = length(lhidden) + 1;
    sizes = [inputSize lhidden];
    nb = sum(lhidden);
    
    % split the vector into two halves
    wTheta = theta(1 : (end - nb));
    bTheta = theta((end - nb) + 1 : end);


    % the start idx and end idx of w and b in the splitted vector contains
    % only w or b
    wstart = 1;
    wend = wstart + sizes(1) * sizes(2) - 1;
    bstart = 1;
    bend = bstart + sizes(2) - 1;


    % init W and b for returning
    W = cell(nlayer - 1, 1);
    b = cell(nlayer - 1, 1);


    for i = 1 : nlayer - 1
	    W{i} = reshape(wTheta(wstart : wend ),...
								 sizes(i + 1), sizes(i));
		b{i} = bTheta(bstart : bend);


        % update the start and end points
        if i ~= nlayer - 1
            wstart = wstart + sizes(i) * sizes(i + 1);
            wend = wstart + sizes(i + 1) * sizes(i + 2) - 1;
            bstart = bstart + sizes(i + 1);
            bend = bstart + sizes(i + 2) - 1;
        end
    end
end