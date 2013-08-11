function [W, b] = extractParam(theta, lhidden, inputSize)
	% get the number of digits in theta is for W
	s = 0;
	for i = 1 : length(lhidden)
		if i == 1
			s = s + inputSize * lhidden(i);
		else
			s = s + lhidden(i) * lhidden(i - 1);
		end
	end

	bstart = s + 1;
	bTheta = theta(bstart : end);

	for i = 1 : length(lhidden) 
		hiddenSize = lhidden(i);

		if i == 1	
			visibleSize = inputSize;
		else
			visibleSize = lhidden(i - 1);
		end

		W{i} = reshape(theta((hiddenSize * visibleSize) * (i - 1) + 1 :...
								 hiddenSize * visibleSize * i),...
								 hiddenSize, visibleSize);
		b{i} = bTheta((i - 1) * hiddenSize + 1 : ...
					  i * hiddenSize);
	end
end