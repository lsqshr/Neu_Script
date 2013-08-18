clear;
lhiddenlist = {[1200 1200] [1100 1100]};
%lhiddenlist = {[2] [2] [2] [2]};
parfor i = 1 : numel(lhiddenlist)
    godeep(lhiddenlist{i},'bio',0);
end
