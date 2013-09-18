function trials = generateTrials(domain, numTrials)
    %% This function produce a number of trail hyperparameters in the domain between domain.start and domain.end
    a      = log10(domain.start);
    b      = log10(domain.end);
    rlog   = a + (b-a).*rand(numTrials, 1);
    trials = 10.^rlog; 
end