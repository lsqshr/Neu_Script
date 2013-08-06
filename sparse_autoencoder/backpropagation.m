function [dW, db] = backpropagation(labels, W, b, a, hp, BETA, sparsityParam)
    % The labels here are the initial data in unsupervised machine

    nlayer = length(W) + 1;
    ninstance = size(labels, 2);
    hypothesis = a{nlayer};

    % calc the error terms

    % first layer
    errterms = cell(nlayer, 1);
    errterms{nlayer} = costfun(hypothesis, labels);

    % hidden layers
    for l = (nlayer - 1): -1 : 2
        sig_prime = (a{l} .* (1 - a{l}));
        sparsity_delta = -(sparsityParam ./ hp{l}) + (1 - sparsityParam) ./ (1 - hp{l});
        errterms{l} = ((W{l}' * errterms{l + 1})  + BETA * repmat(sparsity_delta, 1, ninstance)) .* sig_prime;
    end

    dW = cell(nlayer - 1, 1);
    db = cell(nlayer - 1, 1);
    for l = 1 : nlayer - 1
        % calc two partial dervatives
        dW{l} = errterms{l + 1} * a{l}';
        db{l} = sum(errterms{l + 1}, 2);
    end

end
