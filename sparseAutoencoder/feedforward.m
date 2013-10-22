function [y, a, hp] =  feedforward(data, W, b)
	% This method support nlayer feedforward
	% W: a cell list of weight matrices
	% b: a cell list of bias term vectors
	% hp: the sparsity parameter, a cell of $nlayer vectors
    nlayer = length(W) + 1;
    ndata = size(data, 2);
    z = cell(nlayer, 1);
    a = cell(nlayer, 1);
    z{1} = data; % the first layer is just the inputs
    a{1} = data; 
    hp = cell(nlayer, 1);

    for l = 1 : nlayer - 1
        z{l + 1} = W{l} * a{l} + repmat(b{l}, 1, ndata);
        a{l + 1} = sigmoid(z{l + 1});
    end

    for l = 2 : nlayer - 1
    		sp = sum(a{l}, 2);
    		hp{l} = sp/ ndata;
    end

     y = a{nlayer};
end
