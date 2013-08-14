function opttheta = gatherVector(T, lhidden, inputSize)
%% gather all the relevant theta into a vector
	thetaW = [];
	thetaB = [];
	
	for i = 1 : length(T)
		hiddenSize = lhidden(i);

		if i == 1	
			visibleSize = inputSize;
		else
			visibleSize = lhidden(i - 1);
		end

		[W, b] = extractParam(T{i}, hiddenSize, visibleSize);

		thetaW = [thetaW ; W{1}(:)];
		thetaB = [thetaB ; b{1}(:)];
	end

	opttheta = [thetaW ; thetaB];