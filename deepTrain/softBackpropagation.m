function [dW, db] = softBackpropagation(labels, W,...
                     a, hp, BETA, sparsityParam, errfun)
    % The labels here are the initial data in unsupervised machine

    nlayer = length(W) + 1;
    ndata = size(labels, 2);
    a{nlayer} = a{nlayer};

    % calc the error terms

    % first layer
    errterms = cell(nlayer, 1);
    if BETA == 0
        output = a{nlayer + 1};
        sigPrime = a{nlayer} .* (1 - a{nlayer});
    else
        output = a{nlayer};
        sigPrime = output .* (1 - output);
    end
        
    dJ = errfun(labels, output);
    errterms{nlayer} = -dJ .* sigPrime;
    
    
    % hidden layers
    for l = (nlayer - 1): -1 : 2
        sigPrime = (a{l} .* (1 - a{l}));
        if BETA == 0
            sparsityterm = 0;
        else
            sparsityDelta = -(sparsityParam ./ hp{l}) +...
                            (1 - sparsityParam) ./ (1 - hp{l});
            sparsityterm = BETA * repmat(sparsityDelta, 1, ndata);
        end
        errterms{l} = ((W{l}' * errterms{l + 1}) + sparsityterm) .* sigPrime;
    end

    dW = cell(nlayer - 1, 1);
    db = cell(nlayer - 1, 1);
    for l = 1 : nlayer - 1
        % calc two partial dervatives
        dW{l} = errterms{l + 1} * a{l}';
        db{l} = sum(errterms{l + 1}, 2);
    end

end
